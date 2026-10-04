"""Convert existing UrbanShare hour exports, never retrain or alter its architecture."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .data import joint_target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poi-hours',type=Path,required=True)
    p.add_argument('--mode-hours',type=Path,required=True)
    p.add_argument('--sample-keys',type=Path,required=True,help='Use one baseline test prediction CSV')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    def read(path):
        df=pd.read_csv(path,dtype={'agent_id':str})
        df['date']=pd.to_datetime(df.date)
        if df.duplicated(['agent_id','date']).any(): raise ValueError('Duplicate keys')
        return df.set_index(['agent_id','date'])
    poi,mode,template=[read(path) for path in [args.poi_hours,args.mode_hours,args.sample_keys]]
    if not template.index.isin(poi.index).all() or not template.index.isin(mode.index).all():
        raise ValueError('Original model is missing common test samples')
    poi,mode=poi.loc[template.index],mode.loc[template.index]
    cats=[c for c in template if c.startswith('cat_') and c[4:].isdigit()]
    modes=[c for c in template if c.startswith('mode_') and c[5:].isdigit()]
    h=poi.day_hours.to_numpy(float)
    if np.any(h<=0) or not np.allclose(h,mode.day_hours.to_numpy(float)):
        raise ValueError('Inconsistent/nonpositive hour budgets')
    ch=poi[[c+'h' for c in cats]].to_numpy(float)
    mh=mode[[c+'h' for c in modes]].to_numpy(float)
    if np.any(ch<0) or np.any(mh<0) or not np.allclose(ch.sum(1)+mh.sum(1),h,atol=2e-5):
        raise ValueError('Exported predicted hours violate the time budget')
    out=template.copy()
    out['day_hours']=h
    out[cats]=ch/h[:,None]
    out['travel_frac']=mh.sum(1)/h
    out[modes]=mh/np.maximum(mh.sum(1,keepdims=True),1e-12)
    # Hour files are rounded to 6 decimals; remove only that rounding residual.
    total=out[cats].sum(1)+out.travel_frac
    out[cats]=out[cats].div(total,axis=0)
    out['travel_frac']/=total
    joint_target(out,cats,modes,'unconditional')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    out.reset_index().to_csv(args.out,index=False)

if __name__=='__main__': main()
