"""Causal panel preparation shared by the external baselines.

Target: unconditional shares [activity_0..K-1, travel_mode_0..M-1].
No target-day mobility, coverage, or observed rolling summaries enter inputs.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd


def simplex(values):
    """Euclidean simplex projection, with exact zeros; no learned output gates."""
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite predictions')
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    u = np.sort(x, axis=1)[:, ::-1]
    css = u.cumsum(1) - 1
    count = (u - css / np.arange(1, x.shape[1]+1) > 0).sum(1)
    theta = css[np.arange(len(x)), count-1] / count
    return np.maximum(x-theta[:, None], 0).reshape(shape).astype(np.float32)


def joint_target(frame, cats, modes, cat_contract):
    c = frame[cats].to_numpy(float)
    q = frame[modes].to_numpy(float)
    r = frame.travel_frac.to_numpy(float)
    if not np.isfinite(np.c_[c, q, r]).all():
        raise ValueError('Targets contain missing/non-finite values')
    if np.min(np.c_[c, q, r]) < -1e-7 or np.max(r) > 1+1e-6:
        raise ValueError('Invalid negative shares or travel fraction')
    if cat_contract == 'conditional':
        c = c * (1-r[:, None])
    if not np.allclose(c.sum(1)+r, 1, atol=2e-5):
        raise ValueError('Category budget mismatch: select the correct --cat-contract')
    active = r > 1e-8
    if not np.allclose(q[active].sum(1), 1, atol=2e-5):
        raise ValueError('Mode shares must sum to 1 on travel days')
    q = np.maximum(q, 0)
    q = q / np.maximum(q.sum(1, keepdims=True), 1e-12)
    y = np.c_[c, r[:, None]*q]
    return (y / y.sum(1, keepdims=True)).astype(np.float32)


@dataclass
class Panel:
    frame: pd.DataFrame
    y: np.ndarray
    context: np.ndarray
    dates: np.ndarray
    agents: list
    cat_cols: list
    mode_cols: list
    context_names: list
    train_days: np.ndarray
    val_days: np.ndarray
    test_days: np.ndarray
    history: int

    def examples(self, days, buffer=None):
        """Date-major causal histories; start-of-panel days get zero left padding."""
        b = self.y if buffer is None else buffer
        def window(values,d):
            start=max(0,d-self.history)
            part=values[start:d]
            if len(part)<self.history:
                pad=np.zeros((self.history-len(part),)+values.shape[1:],dtype=values.dtype)
                part=np.concatenate([pad,part],axis=0)
            return part.transpose(1,0,2)
        h = np.concatenate([window(b,d) for d in days])
        # Static demographics/IDs are already supplied in current. Avoid repeating
        # a 911-dimensional one-hot ID for every historical day.
        past = np.concatenate([window(self.context[:,:,:12],d) for d in days])
        current = np.concatenate([self.context[d] for d in days])
        return h.astype(np.float32), past.astype(np.float32), current.astype(np.float32)

    def targets(self, days):
        return self.y[days].reshape(-1, self.y.shape[-1])

    def records(self, days):
        return self.frame[self.frame.date.isin(self.dates[days])].copy().reset_index(drop=True)

    def split_manifest(self):
        return {name: [str(pd.Timestamp(self.dates[d]).date()) for d in days]
                for name, days in [('train',self.train_days),('validation',self.val_days),('test',self.test_days)]}


def prepare(csv_path, history=14, val_days=14, test_days=28, cat_contract='unconditional',
            max_agents=None, calendar=None, include_agent_id=False, split_manifest=None,
            split_mode='global_temporal'):
    df = pd.read_csv(csv_path, dtype={'agent_id':str})
    df['date'] = pd.to_datetime(df.date)
    cats = sorted([c for c in df if c.startswith('cat_') and c[4:].isdigit()],key=lambda c:int(c[4:]))
    modes = sorted([c for c in df if c.startswith('mode_') and c[5:].isdigit()],key=lambda c:int(c[5:]))
    if not cats or not modes:
        raise ValueError('No cat_* or mode_* columns')
    if df.duplicated(['date','agent_id']).any():
        raise ValueError('Duplicate person-date records')
    agents = sorted(df.agent_id.unique())
    if max_agents is not None:
        agents = agents[:max_agents]
        df = df[df.agent_id.isin(agents)].copy()
    dates = np.sort(df.date.unique())
    if split_mode not in ('global_temporal', 'phase_last_week_test'):
        raise ValueError(f'Unknown split mode: {split_mode}')
    if split_mode == 'global_temporal' and len(dates) < history+val_days+test_days+1 and split_manifest is None:
        raise ValueError('Not enough dates for history and three temporal splits')
    if len(dates)>1 and not np.all(np.diff(dates)==np.timedelta64(1,'D')):
        raise ValueError('Dates must be daily and contiguous; no implicit future imputation')
    df = df.sort_values(['date','agent_id']).reset_index(drop=True)
    if len(df) != len(dates)*len(agents):
        raise ValueError('Expected a complete daily panel; handle missing days explicitly first')
    y = joint_target(df,cats,modes,cat_contract).reshape(len(dates),len(agents),-1)
    if split_manifest is None:
        if split_mode == 'global_temporal':
            tr = np.arange(history,len(dates)-val_days-test_days)
            va = np.arange(len(dates)-val_days-test_days,len(dates)-test_days)
            te = np.arange(len(dates)-test_days,len(dates))
        else:
            # Match daily_share_model._split_phase_dates_last_week_test: the
            # final seven dates of each phase are test; it has no validation.
            by_date = df.groupby('date',sort=True).epi_phase.nunique()
            if (by_date != 1).any():
                raise ValueError('A policy phase must be common to all people on a date')
            phases = df.groupby('date',sort=True).epi_phase.first().to_numpy()
            train, test = [], []
            for phase in sorted(np.unique(phases)):
                ix = np.flatnonzero(phases == phase)
                if len(ix) < 2:
                    raise ValueError(f'Phase {phase} has too few dates for last-week test')
                n_test = min(7,len(ix)-1)
                train.extend(ix[:-n_test].tolist())
                test.extend(ix[-n_test:].tolist())
            tr = np.array(sorted(train),dtype=int)
            va = np.array([],dtype=int)
            te = np.array(sorted(test),dtype=int)
    else:
        splits = json.loads(Path(split_manifest).read_text(encoding='utf-8'))
        lookup = {str(pd.Timestamp(d).date()):i for i,d in enumerate(dates)}
        tr,va,te = [np.array([lookup[d] for d in splits[k]],dtype=int) for k in ['train','validation','test']]
    if len(tr)==0 or len(te)==0:
        raise ValueError('Train and test must be nonempty')
    if min(np.r_[tr,va,te])<0 or len(set(tr)&set(va)) or len(set(tr)&set(te)) or len(set(va)&set(te)):
        raise ValueError('Split overlaps or contains invalid dates')
    if split_mode == 'global_temporal':
        if tr.min()<history:
            raise ValueError('Global split requires complete pre-training history')
        if not len(va) or not (tr.max()<va.min() and va.max()<te.min()):
            raise ValueError('Global split must have ordered train, validation, test')
        for s in (tr,va,te):
            if len(s)>1 and not np.all(np.diff(s)==1):
                raise ValueError('Each global split must be contiguous')
        if va.min()!=tr.max()+1 or te.min()!=va.max()+1:
            raise ValueError('Gaps between global temporal splits are unsupported')
    elif len(va):
        raise ValueError('phase_last_week_test must have no validation dates, matching the main model')

    # Calendar contains only exogenous policy information. Negative countdowns to
    # future phase changes from the original calendar are intentionally not read.
    if calendar:
        cal = pd.read_csv(calendar)
        cal['date'] = pd.to_datetime(cal.date)
        if 'phase_id' in cal:
            cal = cal.rename(columns={'phase_id':'epi_phase'})
        if cal.date.duplicated().any():
            raise ValueError('Calendar contains duplicate dates')
        df = df.drop(columns=['epi_phase']).merge(cal[['date','epi_phase']],on='date',how='left',validate='many_to_one')
        if df.epi_phase.isna().any():
            raise ValueError('Calendar must cover every input date')
    # Public phase vocabulary is explicit (not fitted to test outcomes).
    phase = df.epi_phase.to_numpy(int)
    if not np.isin(phase,[0,1,2,3]).all():
        raise ValueError('Expected phase IDs 0..3; extend the declared vocabulary explicitly')
    by_date = df.groupby('date').epi_phase
    if (by_date.nunique()!=1).any():
        raise ValueError('A policy phase must be common to all people on a date')
    daily_phase = by_date.first()
    since = daily_phase.groupby(daily_phase.ne(daily_phase.shift()).cumsum()).cumcount()
    fields = {f'phase_{j}':(phase==j).astype(float) for j in range(4)}
    fields.update({f'weekday_{j}':(df.date.dt.weekday.to_numpy()==j).astype(float) for j in range(7)})
    fields['days_since_phase'] = df.date.map(since).to_numpy(float)/30
    for c in ['gender']+[f'age_{j}' for j in range(5)]:
        if c in df:
            s = pd.to_numeric(df[c],errors='coerce')
            if s.isna().any():
                raise ValueError(f'Non-numeric/missing static covariate: {c}')
            if df.assign(_v=s).groupby('agent_id')._v.nunique().max()>1:
                raise ValueError(f'{c} must be static per person')
            fields[c] = s.to_numpy(float)
    if include_agent_id:
        train_agents = sorted(df[df.date.isin(dates[tr])].agent_id.unique())
        for a in train_agents:
            fields[f'agent={a}'] = (df.agent_id==a).to_numpy(float)
    context = np.stack(list(fields.values()),axis=-1).astype(np.float32).reshape(len(dates),len(agents),-1)
    return Panel(df,y,context,dates,agents,cats,modes,list(fields),tr,va,te,history)


def flat_features(h,past,current):
    return np.concatenate([h.reshape(len(h),-1),past.reshape(len(h),-1),current],axis=1)


def prediction_frame(panel, days, joint):
    out = panel.records(days)
    k = len(panel.cat_cols)
    joint = simplex(joint)
    r = joint[:,k:].sum(1)
    q = joint[:,k:]/np.maximum(r[:,None],1e-12)
    out[panel.cat_cols] = joint[:,:k]
    out[panel.mode_cols] = q
    out['travel_frac'] = r
    keep = ['agent_id','date','epi_phase','day_hours','travel_frac']+panel.cat_cols+panel.mode_cols
    keep += [c for c in ['gender']+[f'age_{i}' for i in range(5)] if c in out]
    return out[keep]


def metrics(y,pred,k,hours=None):
    p = simplex(pred).astype(float)
    y = np.asarray(y,dtype=float)
    r,rt = p[:,k:].sum(1),y[:,k:].sum(1)
    def js(a,b):
        a = np.maximum(a,1e-8); a=a/a.sum(1,keepdims=True)
        b = np.maximum(b,1e-8); b=b/b.sum(1,keepdims=True)
        mid=(a+b)/2
        return float(np.mean(.5*(a*np.log(a/mid)).sum(1)+.5*(b*np.log(b/mid)).sum(1)))
    result = {'joint_share_mae':float(np.abs(p-y).mean()),'joint_share_mse':float(((p-y)**2).mean()),
              'activity_hours24_mae':float(np.abs(p[:,:k]-y[:,:k]).mean()*24),
              'mode_hours24_mae':float(np.abs(p[:,k:]-y[:,k:]).mean()*24),
              'travel_frac_mae':float(np.abs(r-rt).mean()),'joint_jsd':js(y,p),
              'max_budget_residual':float(np.abs(p.sum(1)-1).max()),'min_share':float(p.min()),
              'n':len(p)}
    active=rt>1e-8
    if active.any():
        # Smoothing makes predictions with zero travel explicit, not silently omitted.
        result['mode_jsd_on_observed_travel_days']=js(y[active,k:],p[active,k:])
    if hours is not None:
        result['observed_hours_mae']=float((np.abs(p-y)*np.asarray(hours)[:,None]).mean())
    return result
