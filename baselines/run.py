"""Run from repository root: python -m baselines.run --model tabm ..."""
import argparse
import copy
import hashlib
import json
import platform
import importlib.metadata
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .data import prepare,flat_features,simplex,prediction_frame,metrics
from .paper_metrics import paper_metrics

ROOT=Path(__file__).resolve().parents[1]


def json_write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def array_inputs(panel,days,buffer=None):
    h,p,c=panel.examples(days,buffer)
    return flat_features(h,p,c),h,p,c


class Predictor:
    def __init__(self,name,model=None,mean=None,scale=None,device='cpu',batch_size=512):
        self.name,self.model,self.mean,self.scale,self.device,self.batch_size=name,model,mean,scale,device,batch_size

    def predict(self,inputs):
        flat,h,p,c=inputs
        if self.name=='persistence':
            return simplex(h[:,-1,:])
        if self.name=='mean7':
            tail=h[:,-min(7,h.shape[1]):,:]
            valid=(tail.sum(-1)>1e-8)
            return simplex(tail.sum(1)/np.maximum(valid.sum(1,keepdims=True),1))
        if self.name=='catboost':
            return simplex(self.model.predict(flat))
        import torch
        self.model.eval()
        result=[]
        with torch.no_grad():
            for start in range(0,len(flat),self.batch_size):
                stop=start+self.batch_size
                args=[(flat[start:stop]-self.mean)/self.scale,h[start:stop],p[start:stop],c[start:stop]]
                args=[torch.as_tensor(x,dtype=torch.float32,device=self.device) for x in args]
                pred=self.model(*args).cpu().numpy()
                # For TabM: project each member before ensembling valid budgets.
                pred=simplex(pred)
                if pred.ndim==3:
                    pred=pred.mean(1)
                result.append(pred)
        return np.concatenate(result)


def evaluate_days(panel,predictor,days,protocol,scenario=False):
    if protocol=='one-step':
        if scenario:
            raise ValueError('Counterfactual calendar requires recursive protocol')
        return predictor.predict(array_inputs(panel,days))
    predictions=[]
    # Phase-stratified tests are four distinct seven-day episodes. Each block
    # starts from its factual pre-block history, with no labels used inside it.
    starts=np.r_[0,np.flatnonzero(np.diff(days)>1)+1]
    stops=np.r_[starts[1:],len(days)]
    for start,stop in zip(starts,stops):
        block=days[start:stop]
        buffer=panel.y.copy()
        buffer[block[0]:block[-1]+1]=np.nan
        for d in block:
            pred=predictor.predict(array_inputs(panel,[d],buffer))
            buffer[d]=pred
            predictions.append(pred)
    return np.concatenate(predictions)


def full_horizon_rollout(panel,predictor):
    """Factual or counterfactual 184-day closed loop, with zero pre-panel lag."""
    buffer=np.zeros_like(panel.y)
    predictions=[]
    for d in range(len(panel.dates)):
        pred=predictor.predict(array_inputs(panel,[d],buffer))
        buffer[d]=pred
        predictions.append(pred)
    return np.concatenate(predictions)


def fit(panel,args,out):
    if args.model in ('persistence','mean7'):
        return Predictor(args.model),[]
    inputs=array_inputs(panel,panel.train_days)
    y=panel.targets(panel.train_days)
    has_val=len(panel.val_days)>0
    val_inputs=array_inputs(panel,panel.val_days) if has_val else None
    vy=panel.targets(panel.val_days) if has_val else None
    if args.model=='catboost':
        from catboost import CatBoostRegressor
        model=CatBoostRegressor(iterations=args.iterations,depth=args.depth,learning_rate=args.lr,
                  loss_function='MultiRMSE',random_seed=args.seed,thread_count=args.threads,
                  allow_writing_files=False,verbose=False,l2_leaf_reg=args.weight_decay)
        if args.selection_protocol=='one-step' and has_val:
            model.fit(inputs[0],y,eval_set=(val_inputs[0],vy),early_stopping_rounds=args.patience,use_best_model=True)
        else:
            model.fit(inputs[0],y)
        predictor=Predictor(args.model,model)
        history=[]
        if args.selection_protocol=='recursive' and has_val:
            # Select boosting rounds on closed-loop validation, not test labels.
            best=float('inf'); best_model=None
            for trees in sorted(set(list(range(10,model.tree_count_+1,10))+[model.tree_count_])):
                candidate=model.copy(); candidate.shrink(trees)
                predictor.model=candidate
                pred=evaluate_days(panel,predictor,panel.val_days,'recursive')
                score=float(((pred-vy)**2).mean())
                history.append({'trees':trees,'val_joint_mse':score})
                if score<best:
                    best,best_model=score,candidate
            predictor.model=best_model
        predictor.model.save_model(str(out/'model.cbm'))
        return predictor,history
    import torch
    from .models import NEURAL
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    flat=inputs[0]
    mean=flat.mean(0); scale=flat.std(0)
    scale=np.where(scale>1e-5,scale,1).astype(np.float32)
    kwargs=dict(flat_dim=flat.shape[1],output_dim=y.shape[1],history=panel.history,
                ctx_dim=panel.context.shape[-1],width=args.width,depth=args.depth,k=args.ensemble_size,dropout=args.dropout)
    if args.model in ('gru','lstm','transformer','logistic_normal'):
        kwargs.update(past_ctx_dim=inputs[2].shape[-1])
    if args.model=='logistic_normal':
        kwargs.update(eps=args.composition_eps)
    if args.model=='mdcev':
        kwargs.update(draws=args.mdcev_draws,seed=args.seed)
    model=NEURAL[args.model](**kwargs).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    predictor=Predictor(args.model,model,mean,scale,args.device,args.batch_size)
    best=float('inf'); best_state=None; stale=0; log=[]
    rng=np.random.default_rng(args.seed)
    for epoch in range(1,args.epochs+1):
        model.train(); total=0
        order=rng.permutation(len(y))
        for offset in range(0,len(order),args.batch_size):
            idx=order[offset:offset+args.batch_size]
            batch=[(flat[idx]-mean)/scale,inputs[1][idx],inputs[2][idx],inputs[3][idx]]
            batch=[torch.as_tensor(x,dtype=torch.float32,device=args.device) for x in batch]
            target=torch.as_tensor(y[idx],device=args.device)
            if args.model=='mdcev':
                loss=model.nll(batch[0],target)
            elif args.model=='logistic_normal':
                loss=model.nll(*batch,target)
            else:
                pred=model(*batch)
                if pred.ndim==3:
                    target=target[:,None,:]
                loss=((pred-target)**2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); total+=float(loss.detach())*len(idx)
        selection_days=panel.val_days if has_val else panel.train_days
        selection_y=vy if has_val else y
        pred=evaluate_days(panel,predictor,selection_days,args.selection_protocol)
        score=float(((pred-selection_y)**2).mean())
        row={'epoch':epoch,'train_loss':total/len(y),
             ('val_joint_mse' if has_val else 'train_joint_mse'):score}
        log.append(row); print(json.dumps(row),flush=True)
        if score<best:
            best=score; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
        if stale>=args.patience:
            break
    model.load_state_dict(best_state)
    torch.save({'state_dict':best_state,'kwargs':kwargs,'name':args.model},out/'model.pt')
    np.savez(out/'scaler.npz',mean=mean,scale=scale)
    return predictor,log


def reload_predictor(args,out):
    if args.model in ('persistence','mean7'):
        return Predictor(args.model)
    if args.model=='catboost':
        from catboost import CatBoostRegressor
        model=CatBoostRegressor(); model.load_model(str(out/'model.cbm'))
        return Predictor(args.model,model)
    import torch
    from .models import NEURAL
    torch.set_num_threads(args.threads)
    ckpt=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    if ckpt['name']!=args.model:
        raise ValueError('Checkpoint model mismatch')
    model=NEURAL[args.model](**ckpt['kwargs']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    scaler=np.load(out/'scaler.npz')
    return Predictor(args.model,model,scaler['mean'],scaler['scale'],args.device,args.batch_size)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True,choices=['persistence','mean7','catboost','gru','lstm','transformer','tabm','timexer','logistic_normal','mdcev'])
    ap.add_argument('--data',type=Path,default=ROOT/'data/behavior/observed.csv.gz')
    ap.add_argument('--cat-contract',choices=['conditional','unconditional'],default='unconditional')
    ap.add_argument('--calendar',type=Path,help='Counterfactual calendar; only supported with --load-run')
    ap.add_argument('--history',type=int,default=14)
    ap.add_argument('--val-days',type=int,default=14)
    ap.add_argument('--test-days',type=int,default=28)
    ap.add_argument('--split-mode',choices=['global_temporal','phase_last_week_test'],
                    default='phase_last_week_test')
    ap.add_argument('--max-agents',type=int,help='Smoke-test subset, never label as full benchmark')
    ap.add_argument('--include-agent-id',action='store_true',help='Train-only one-hot ID vocabulary shared by all learners')
    ap.add_argument('--split-manifest',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--load-run',type=Path,help='Load trusted local checkpoint, do not retrain')
    ap.add_argument('--selection-protocol',choices=['one-step','recursive'],default='one-step')
    ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--iterations',type=int,default=1000)
    ap.add_argument('--patience',type=int,default=10)
    ap.add_argument('--batch-size',type=int,default=256)
    ap.add_argument('--width',type=int,default=128)
    ap.add_argument('--depth',type=int,default=2)
    ap.add_argument('--ensemble-size',type=int,default=16)
    ap.add_argument('--dropout',type=float,default=.1)
    ap.add_argument('--mdcev-draws',type=int,default=64)
    ap.add_argument('--composition-eps',type=float,default=1e-4,
                    help='Additive zero replacement used only by logistic_normal')
    ap.add_argument('--lr',type=float,default=None)
    ap.add_argument('--weight-decay',type=float,default=None)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--seed',type=int,default=1234)
    args=ap.parse_args()
    if args.calendar and not args.load_run:
        ap.error('--calendar requires --load-run: never train on a counterfactual calendar with factual labels')
    if min(args.history,args.val_days,args.test_days,args.epochs,args.iterations,args.batch_size,args.patience,args.threads)<=0:
        ap.error('History, split sizes, training counts, patience and threads must be positive')
    if args.max_agents is not None and args.max_agents<1:
        ap.error('--max-agents must be positive')
    if not 0<args.composition_eps<.1:
        ap.error('--composition-eps must lie between 0 and 0.1')
    if args.model in ('timexer','transformer') and args.width%4:
        ap.error('TimeXer/Transformer --width must be divisible by 4 attention heads')
    args.lr=args.lr if args.lr is not None else (.05 if args.model=='catboost' else .001)
    args.weight_decay=args.weight_decay if args.weight_decay is not None else (3. if args.model=='catboost' else .0001)
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():
        raise FileExistsError(f'Output already contains a run; choose a new directory: {out}')
    start=time.time()
    np.random.seed(args.seed)
    if args.load_run:
        saved=json.loads((args.load_run/'config.json').read_text(encoding='utf-8'))
        if saved['model']!=args.model:
            raise ValueError('--model differs from --load-run')
        for key in ['history','val_days','test_days','max_agents','include_agent_id','cat_contract']:
            setattr(args,key,saved[key])
        args.split_mode=saved.get('split_mode','global_temporal')
        args.split_manifest=args.load_run/'splits.json'
    if args.split_mode=='phase_last_week_test' and args.selection_protocol!='one-step' and not args.load_run:
        ap.error('The main-model phase split has no validation set; use --selection-protocol one-step')
    panel=prepare(args.data,args.history,args.val_days,args.test_days,args.cat_contract,args.max_agents,
                  args.calendar,args.include_agent_id,args.split_manifest,args.split_mode)
    if args.load_run:
        original=json.loads((args.load_run/'schema.json').read_text(encoding='utf-8'))
        if original['contexts']!=panel.context_names or original['agents']!=panel.agents or original['cats']!=panel.cat_cols or original['modes']!=panel.mode_cols:
            raise ValueError('Loaded run and input schema/agent order differ')
        if args.calendar and args.split_mode=='global_temporal':
            base=prepare(args.data,args.history,args.val_days,args.test_days,args.cat_contract,args.max_agents,
                         None,args.include_agent_id,args.split_manifest,args.split_mode)
            if not np.array_equal(base.context[:panel.test_days[0]],panel.context[:panel.test_days[0]]):
                raise ValueError('Counterfactual starts before test origin: create a factual run whose test origin is at/before policy divergence')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['input_sha256']=hashlib.sha256(args.data.read_bytes()).hexdigest()
    config['python']=platform.python_version()
    config['packages']={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','torch','catboost','rtdl_num_embeddings','reformer-pytorch'] if importlib.util.find_spec(name.replace('-','_')) is not None}
    source_lock=Path(__file__).with_name('sources.lock.json')
    if source_lock.exists():
        config['upstream_sources']=json.loads(source_lock.read_text(encoding='utf-8'))
    config['adapter_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['run.py','data.py','models.py','paper_metrics.py']}
    if args.calendar:
        config['calendar_sha256']=hashlib.sha256(args.calendar.read_bytes()).hexdigest()
    json_write(out/'config.json',config)
    json_write(out/'splits.json',panel.split_manifest())
    json_write(out/'schema.json',{'cats':panel.cat_cols,'modes':panel.mode_cols,'contexts':panel.context_names,
                  'agents':panel.agents,'target':'unconditional activity and unconditional mode shares; one simplex',
                  'innovations_excluded':['UrbanShare imports','MoE','phase FiLM','phase lag gate','PA','phase reset templates']})
    if panel.cat_cols!=[f'cat_{j}' for j in range(12)] or panel.mode_cols!=[f'mode_{j}' for j in range(6)]:
        raise ValueError('Paper metrics require cat_0..cat_11 and mode_0..mode_5')
    predictor,log=(reload_predictor(args,args.load_run),[]) if args.load_run else fit(panel,args,out)
    json_write(out/'training_history.json',log)
    results={}
    if args.calendar and args.split_mode=='phase_last_week_test':
        all_days=np.arange(len(panel.dates))
        pred=full_horizon_rollout(panel,predictor)
        records=prediction_frame(panel,all_days,pred)
        records['day_hours']=24.0
        records.to_csv(out/'seir_scenario_full_horizon.csv.gz',index=False)
        results['scenario_only_no_counterfactual_ground_truth']=True
        results['smoke_subset']=args.max_agents is not None
        results['elapsed_seconds']=time.time()-start
        json_write(out/'metrics.json',results)
        print(json.dumps(results,indent=2),flush=True)
        return
    for protocol in (['recursive'] if args.calendar else ['one-step','recursive']):
        pred=evaluate_days(panel,predictor,panel.test_days,protocol,bool(args.calendar))
        records=prediction_frame(panel,panel.test_days,pred)
        if args.calendar:
            records['day_hours']=24.0
        records.to_csv(out/f'predictions_{protocol}.csv.gz',index=False)
        if args.calendar:
            prefix_days=np.arange(panel.test_days[0])
            prefix=prediction_frame(panel,prefix_days,panel.targets(prefix_days))
            prefix['day_hours']=24.0
            pd.concat([prefix,records],ignore_index=True).to_csv(out/'seir_scenario_with_factual_prefix.csv.gz',index=False)
        if not args.calendar:
            results[protocol]=metrics(panel.targets(panel.test_days),pred,len(panel.cat_cols),records.day_hours.to_numpy())
            results[protocol]['paper']=paper_metrics(
                panel.targets(panel.test_days),pred,len(panel.cat_cols),
                panel.dates[panel.test_days].repeat(len(panel.agents)),
                records.day_hours.to_numpy(),unknown_mode_index=5)
            per_phase={}
            for phase in sorted(records.epi_phase.unique()):
                mask=(records.epi_phase.to_numpy()==phase)
                per_phase[str(phase)]=metrics(panel.targets(panel.test_days)[mask],pred[mask],len(panel.cat_cols))
                per_phase[str(phase)]['paper']=paper_metrics(
                    panel.targets(panel.test_days)[mask],pred[mask],len(panel.cat_cols),
                    records.date.to_numpy()[mask],records.day_hours.to_numpy()[mask],
                    unknown_mode_index=5)
            results[protocol]['by_phase']=per_phase
    if not args.calendar and args.split_mode=='phase_last_week_test':
        all_days=np.arange(len(panel.dates))
        pred=full_horizon_rollout(panel,predictor)
        records=prediction_frame(panel,all_days,pred)
        records.to_csv(out/'predictions_factual_full_horizon_recursive.csv.gz',index=False)
        results['full_horizon_recursive']=metrics(
            panel.targets(all_days),pred,len(panel.cat_cols),records.day_hours.to_numpy())
        results['full_horizon_recursive']['paper']=paper_metrics(
            panel.targets(all_days),pred,len(panel.cat_cols),
            panel.dates.repeat(len(panel.agents)),records.day_hours.to_numpy(),
            unknown_mode_index=5)
    results['elapsed_seconds']=time.time()-start
    results['scenario_only_no_counterfactual_ground_truth']=bool(args.calendar)
    results['smoke_subset']=args.max_agents is not None
    json_write(out/'metrics.json',results)
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    main()
