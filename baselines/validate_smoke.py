"""Real-data integration smoke tests. These are not performance experiments."""
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'baselines/outputs'/('validation-'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    runs=[]
    def run(name,args,success=True):
        command=[sys.executable,'-m',*args]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (out/(name+'.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        passed=(result.returncode==0)==success
        runs.append(dict(name=name,passed=passed,exit_code=result.returncode,command=command))
        print(f'{name}: {"PASS" if passed else "FAIL"}',flush=True)
        if not passed: raise RuntimeError(result.stdout[-1000:]+result.stderr[-3000:])
    run('unit',['unittest','baselines.test_protocol','-v'])
    common=['--max-agents','12','--history','7','--val-days','7','--test-days','7',
            '--split-mode','global_temporal',
            '--epochs','2','--iterations','20','--width','32','--depth','2','--ensemble-size','4',
            '--mdcev-draws','16','--batch-size','128','--threads','2']
    models=['persistence','mean7','catboost','gru','lstm','transformer','tabm','timexer','logistic_normal','mdcev']
    for model in models:
        run(model,['baselines.run','--model',model,*common,'--out',str(out/model)])
    keys=None
    for model in models:
        for protocol in ['one-step','recursive']:
            df=pd.read_csv(out/model/f'predictions_{protocol}.csv.gz',dtype={'agent_id':str})
            cols=[f'cat_{j}' for j in range(12)]
            np.testing.assert_allclose(df[cols].sum(1)+df.travel_frac,1,atol=1e-6)
            if keys is None: keys=df[['agent_id','date']]
            pd.testing.assert_frame_equal(keys,df[['agent_id','date']])
    for model in ['catboost','gru','lstm','transformer','tabm','timexer','logistic_normal','mdcev']:
        run(model+'-reload',['baselines.run','--model',model,'--load-run',str(out/model),'--out',str(out/(model+'-reload')),'--threads','2'])
        for protocol in ['one-step','recursive']:
            original=pd.read_csv(out/model/f'predictions_{protocol}.csv.gz')
            loaded=pd.read_csv(out/(model+'-reload')/f'predictions_{protocol}.csv.gz')
            pd.testing.assert_frame_equal(original,loaded,atol=1e-6,rtol=1e-6)
    for model in ['catboost','timexer']:
        run(model+'-recursive-selection',['baselines.run','--model',model,*common,
            '--selection-protocol','recursive','--out',str(out/(model+'-recursive-selection'))])
    facts=pd.read_csv(ROOT/'data/behavior/observed.csv.gz')
    cal=facts[['date','epi_phase']].drop_duplicates().sort_values('date')
    origin=keys.date.min()
    cal.loc[cal.date>=origin,'epi_phase']=2
    cal.to_csv(out/'scenario.csv',index=False)
    run('scenario',['baselines.run','--model','timexer','--load-run',str(out/'timexer'),
           '--calendar',str(out/'scenario.csv'),'--out',str(out/'scenario'),'--threads','2'])
    scenario_metrics=json.loads((out/'scenario/metrics.json').read_text())
    assert scenario_metrics['scenario_only_no_counterfactual_ground_truth']
    assert 'recursive' not in scenario_metrics
    full=pd.read_csv(out/'scenario/seir_scenario_with_factual_prefix.csv.gz')
    assert len(full)==12*184 and (full.day_hours==24).all()
    cal.loc[cal.date<origin,'epi_phase']=2
    cal.to_csv(out/'invalid_scenario.csv',index=False)
    run('reject-preorigin-policy-change',['baselines.run','--model','timexer','--load-run',str(out/'timexer'),
           '--calendar',str(out/'invalid_scenario.csv'),'--out',str(out/'invalid-scenario')],success=False)
    run('external-evaluation',['baselines.evaluate','--reference',str(ROOT/'data/behavior/observed.csv.gz'),
          '--predictions',str(out/'tabm/predictions_recursive.csv.gz'),
          '--sample-keys',str(out/'persistence/predictions_recursive.csv.gz'),'--out',str(out/'external_metrics.json')])
    # Exercise compatibility with the real UrbanShare hour-file interface.
    source=pd.read_csv(out/'tabm/predictions_recursive.csv.gz',dtype={'agent_id':str})
    meta=['agent_id','date','epi_phase','day_hours']
    poi,mode=source[meta].copy(),source[meta].copy()
    for j in range(12): poi[f'cat_{j}h']=source[f'cat_{j}']*source.day_hours
    for j in range(6): mode[f'mode_{j}h']=source[f'mode_{j}']*source.travel_frac*source.day_hours
    poi.to_csv(out/'synthetic_poi_hours.csv',index=False)
    mode.to_csv(out/'synthetic_mode_hours.csv',index=False)
    run('urbanshare-hour-import',['baselines.import_urbanshare','--poi-hours',str(out/'synthetic_poi_hours.csv'),
        '--mode-hours',str(out/'synthetic_mode_hours.csv'),'--sample-keys',str(out/'tabm/predictions_recursive.csv.gz'),
        '--out',str(out/'converted.csv.gz')])
    pd.testing.assert_frame_equal(source,pd.read_csv(out/'converted.csv.gz',dtype={'agent_id':str}),atol=1e-6,rtol=1e-6)
    report={'scope':'functional smoke tests only, not tuned benchmark rankings','date_utc':stamp,
            'output_directory':str(out),'people':12,'days':184,'test_person_days':84,
            'neural_epochs':2,'catboost_iterations':20,'checks':runs,'all_passed':all(r['passed'] for r in runs)}
    (ROOT/'baselines/VALIDATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'All passed. Artifacts: {out}',flush=True)

if __name__=='__main__': main()
