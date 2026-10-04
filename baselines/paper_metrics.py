"""Paper Table 4/5 metrics for the common person-date set.

The formulas follow daily_share_model.evaluate and the population-series
aggregation in tools/dashboard_app.py. This module changes evaluation only;
external baselines retain their own predictors and 18-share output contract.
"""
import numpy as np
import pandas as pd


def _kl(p, q, eps=1e-8):
    # Main model clamps each entry without renormalising after the clamp.
    a = np.clip(p, eps, 1.0)
    b = np.clip(q, eps, 1.0)
    return np.sum(a * np.log(a / b), axis=-1)


def _r2(y, p):
    ss_res = np.sum((y-p)**2, axis=0)
    ss_tot = np.sum((y-y.mean(axis=0))**2, axis=0)+1e-12
    return float(np.mean(1.0-ss_res/ss_tot))


def _macro(y, p):
    delta = p-y
    mae = np.mean(np.abs(delta), axis=0)
    rmse = np.sqrt(np.mean(delta**2, axis=0))
    bias = np.abs(np.mean(delta, axis=0))
    corr=[]
    for j in range(y.shape[1]):
        yc=y[:,j]-np.mean(y[:,j])
        pc=p[:,j]-np.mean(p[:,j])
        denom=float(np.sqrt(np.sum(yc*yc)*np.sum(pc*pc)))
        if denom==0:
            continue
        corr.append(float(np.sum(yc*pc)/denom))
    return {'MAE_h_day':float(np.mean(mae)),
            'RMSE_h_day':float(np.mean(rmse)),
            'Abs_Bias_h_day':float(np.mean(bias)),
            'Corr':float(np.mean(corr)) if corr else None,
            'n_series':int(y.shape[1]),'n_dates':int(y.shape[0])}


def paper_metrics(y, pred, n_cat, dates, day_hours, unknown_mode_index=5):
    """Evaluate joint shares in aligned date-major order; 5 known modes + 1 unknown.

    `pred` must already be projected to a unit simplex by its adapter.
    Truth and predicted mode shares are conditional on observed/predicted travel.
    """
    y=np.asarray(y,dtype=np.float64)
    p=np.asarray(pred,dtype=np.float64)
    dates=pd.to_datetime(np.asarray(dates))
    hours=np.asarray(day_hours,dtype=np.float64).reshape(-1)
    if y.shape!=p.shape or y.ndim!=2 or len(y)!=len(dates) or len(y)!=len(hours):
        raise ValueError('Inconsistent prediction, date or hour dimensions')
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(hours).all():
        raise ValueError('Non-finite evaluation inputs')
    if (hours<=0).any() or np.min(y)<-1e-7 or np.min(p)<-1e-7:
        raise ValueError('Invalid hour or share values')
    if not np.allclose(y.sum(1),1,atol=2e-5) or not np.allclose(p.sum(1),1,atol=2e-5):
        raise ValueError('Evaluation shares must satisfy the time budget')
    n_mode=y.shape[1]-n_cat
    if n_cat<1 or n_mode<2 or not 0<=unknown_mode_index<n_mode:
        raise ValueError('Invalid category/mode schema')
    known=[j for j in range(n_mode) if j!=unknown_mode_index]
    r=y[:,n_cat:].sum(1)
    t=p[:,n_cat:].sum(1)
    yc=np.divide(y[:,:n_cat],(1-r)[:,None],out=np.zeros_like(y[:,:n_cat]),where=(1-r)[:,None]>1e-8)
    pc=np.divide(p[:,:n_cat],(1-t)[:,None],out=np.zeros_like(p[:,:n_cat]),where=(1-t)[:,None]>1e-8)
    ym=np.divide(y[:,n_cat:],r[:,None],out=np.zeros_like(y[:,n_cat:]),where=r[:,None]>1e-8)
    pm=np.divide(p[:,n_cat:],t[:,None],out=np.zeros_like(p[:,n_cat:]),where=t[:,None]>1e-8)
    ym_known=ym[:,known]
    pm_known=pm[:,known]
    known_mass=ym_known.sum(1)
    yk=np.divide(ym_known,known_mass[:,None],out=np.zeros_like(ym_known),where=known_mass[:,None]>1e-8)
    pred_known_mass=pm_known.sum(1)
    pk=np.divide(pm_known,pred_known_mass[:,None],out=np.zeros_like(pm_known),where=pred_known_mass[:,None]>1e-8)
    w_day=hours/24.0
    w_mode=r*w_day
    mode_mask=(r>1e-8).astype(float)
    den_day=max(float(w_day.sum()),1e-8)
    den_mode=max(float(w_mode.sum()),1e-8)
    cat_h_true=y[:,:n_cat]*hours[:,None]
    cat_h_pred=p[:,:n_cat]*hours[:,None]
    mode_h_true=y[:,n_cat:]*hours[:,None]
    mode_h_pred=p[:,n_cat:]*hours[:,None]
    distribution={
        'wKL_cat':float(np.sum(_kl(yc,pc)*w_day)/den_day),
        'wKL_mode':float(np.sum(_kl(yk,pk)*w_mode*mode_mask*(known_mass>1e-8))/den_mode),
        'wKL_mode_all':float(np.sum(_kl(ym,pm)*w_mode*mode_mask)/den_mode),
        'wMSE_travel':float(np.sum((t-r)**2*w_day)/den_day),
        'wMSE_unknown_h':float(np.sum((mode_h_pred[:,unknown_mode_index]-mode_h_true[:,unknown_mode_index])**2*w_day)/den_day),
        'R2_cat_hours':_r2(cat_h_true,cat_h_pred),
        'R2_mode_hours':_r2(mode_h_true,mode_h_pred),
        'n_agent_days':int(len(y)),
    }
    # The paper averages agents first for each date and only then computes
    # each time-series metric, excluding unknown mode from the macro family.
    truth=pd.DataFrame(np.c_[cat_h_true,mode_h_true[:,known]])
    forecast=pd.DataFrame(np.c_[cat_h_pred,mode_h_pred[:,known]])
    truth['date']=dates
    forecast['date']=dates
    y_daily=truth.groupby('date',sort=True).mean().to_numpy()
    p_daily=forecast.groupby('date',sort=True).mean().to_numpy()
    macro={'POI_categories':_macro(y_daily[:,:n_cat],p_daily[:,:n_cat]),
           'travel_modes_known':_macro(y_daily[:,n_cat:],p_daily[:,n_cat:])}
    return {'distribution':distribution,'macro':macro,
            'protocol':'Paper held-out formulas; main-model 5 known modes and mode_5 unknown'}
