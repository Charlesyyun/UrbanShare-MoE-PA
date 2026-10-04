"""Evaluate external model CSVs with the same metrics and exact sample matching."""
import argparse
from pathlib import Path
import pandas as pd
from .data import joint_target,metrics
from .paper_metrics import paper_metrics
from .run import json_write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--sample-keys',type=Path,required=True,help='A baseline prediction CSV defining the exact common evaluation people/dates')
    ap.add_argument('--reference-contract',choices=['conditional','unconditional'],default='unconditional')
    ap.add_argument('--prediction-contract',choices=['conditional','unconditional'],default='unconditional')
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    truth=pd.read_csv(a.reference,dtype={'agent_id':str})
    pred=pd.read_csv(a.predictions,dtype={'agent_id':str})
    keys=['agent_id','date']
    for frame in [truth,pred]:
        frame['date']=pd.to_datetime(frame.date)
        if frame.duplicated(keys).any(): raise ValueError('Duplicate evaluation keys')
    cats=sorted([c for c in truth if c.startswith('cat_') and c[4:].isdigit()],key=lambda c:int(c[4:]))
    modes=sorted([c for c in truth if c.startswith('mode_') and c[5:].isdigit()],key=lambda c:int(c[5:]))
    indexed=truth.set_index(keys)
    requested=pd.MultiIndex.from_frame(pred[keys])
    expected=pd.read_csv(a.sample_keys,dtype={'agent_id':str})
    expected['date']=pd.to_datetime(expected.date)
    if expected.duplicated(keys).any(): raise ValueError('Duplicate expected evaluation keys')
    wanted=pd.MultiIndex.from_frame(expected[keys])
    if len(requested)!=len(wanted) or not requested.isin(wanted).all():
        raise ValueError('Prediction sample set differs from --sample-keys; filter/export the common test split first')
    if not requested.isin(indexed.index).all(): raise ValueError('Prediction keys not present in reference')
    aligned=indexed.loc[requested].reset_index()
    y=joint_target(aligned,cats,modes,a.reference_contract)
    p=joint_target(pred,cats,modes,a.prediction_contract)
    result=metrics(y,p,len(cats),aligned.day_hours.to_numpy())
    if modes != [f'mode_{j}' for j in range(6)]:
        raise ValueError('Paper metrics require mode_0..mode_5, with mode_5 unknown')
    result['paper']=paper_metrics(y,p,len(cats),aligned.date.to_numpy(),
                                  aligned.day_hours.to_numpy(),unknown_mode_index=5)
    result['scope']='Exact prediction key set verified against --sample-keys; training leakage must be audited separately'
    a.out.parent.mkdir(parents=True,exist_ok=True)
    json_write(a.out,result)
    print(result)

if __name__=='__main__': main()
