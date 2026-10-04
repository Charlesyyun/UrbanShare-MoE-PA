"""Regression checks for information boundaries, contracts and MDCEV math."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from .data import prepare,simplex,joint_target,prediction_frame
from .run import Predictor,evaluate_days,full_horizon_rollout,array_inputs,reload_predictor
from .paper_metrics import paper_metrics


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'panel.csv'
        rng=np.random.default_rng(9)
        rows=[]
        for d in pd.date_range('2020-01-01',periods=22):
            for a in ['a','b']:
                j=rng.dirichlet(np.ones(4))
                rows.append(dict(agent_id=a,date=d,epi_phase=int(d.day>12),day_hours=20,
                         travel_frac=j[2:].sum(),cat_0=j[0],cat_1=j[1],
                         mode_0=j[2]/j[2:].sum(),mode_1=j[3]/j[2:].sum(),
                         gender=float(a=='a'),total_trip_km=999,n_trips=999,prev_travel_frac=999))
        self.df=pd.DataFrame(rows)
        self.df.to_csv(self.path,index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def panel(self):
        return prepare(self.path,history=3,val_days=3,test_days=4)

    def test_projection_budget_and_zero_support(self):
        y=simplex(np.array([[-4,.2,2],[.2,.3,.5]]))
        np.testing.assert_allclose(y.sum(1),1,atol=1e-6)
        self.assertTrue((y>=0).all())
        self.assertEqual(y[0,0],0)
        np.testing.assert_allclose(y[1],[.2,.3,.5])

    def test_contract_roundtrip(self):
        p=self.panel()
        output=prediction_frame(p,p.test_days,p.targets(p.test_days))
        actual=joint_target(output,p.cat_cols,p.mode_cols,'unconditional')
        np.testing.assert_allclose(actual,p.targets(p.test_days),atol=1e-6)
        conditional=output.copy()
        conditional[p.cat_cols]=conditional[p.cat_cols].div(1-conditional.travel_frac,axis=0)
        np.testing.assert_allclose(joint_target(conditional,p.cat_cols,p.mode_cols,'conditional'),actual,atol=1e-6)
        with self.assertRaises(ValueError):
            joint_target(conditional,p.cat_cols,p.mode_cols,'unconditional')

    def test_recursive_never_reads_future_targets(self):
        p=self.panel()
        original=evaluate_days(p,Predictor('mean7'),p.test_days,'recursive')
        inputs=array_inputs(p,[p.test_days[0]])
        p.y[p.test_days]=np.array([.8,.1,.05,.05])
        changed=evaluate_days(p,Predictor('mean7'),p.test_days,'recursive')
        np.testing.assert_array_equal(original,changed)
        for a,b in zip(inputs,array_inputs(p,[p.test_days[0]])):
            np.testing.assert_array_equal(a,b)
        teacher=evaluate_days(p,Predictor('mean7'),p.test_days,'one-step')
        self.assertFalse(np.allclose(teacher,changed))

    def test_same_day_forbidden_features_excluded(self):
        p=self.panel()
        features=array_inputs(p,p.train_days)
        self.df['total_trip_km']=-300
        self.df['n_trips']=42
        self.df['prev_travel_frac']=.9
        self.df['day_hours']=5
        self.df.to_csv(self.path,index=False)
        p2=self.panel()
        for a,b in zip(features,array_inputs(p2,p2.train_days)):
            np.testing.assert_array_equal(a,b)

    def test_empty_duplicate_and_missing_panel_rejected(self):
        pd.concat([self.df,self.df.iloc[:1]]).to_csv(self.path,index=False)
        with self.assertRaises(ValueError): self.panel()
        self.df.iloc[1:].to_csv(self.path,index=False)
        with self.assertRaises(ValueError): self.panel()

    def test_split_order_and_history(self):
        p=self.panel()
        self.assertLess(max(p.train_days),min(p.val_days))
        self.assertLess(max(p.val_days),min(p.test_days))
        self.assertGreaterEqual(min(p.train_days),p.history)

    def test_phase_last_week_matches_main_model_dates(self):
        p=prepare(self.path,history=3,split_mode='phase_last_week_test')
        self.assertEqual(len(p.val_days),0)
        self.assertEqual(p.train_days[0],0)
        self.assertEqual(p.test_days.tolist(),list(range(5,12))+list(range(15,22)))
        self.assertTrue(set(p.train_days).isdisjoint(p.test_days))
        self.assertEqual(p.split_manifest()['test'][0],'2020-01-06')
        self.assertEqual(p.split_manifest()['test'][-1],'2020-01-22')
        result=evaluate_days(p,Predictor('mean7'),p.test_days,'recursive')
        self.assertEqual(result.shape,p.targets(p.test_days).shape)
        self.assertTrue(np.isfinite(result).all())

    def test_full_horizon_rollout_never_reads_observed_targets(self):
        p=prepare(self.path,history=3,split_mode='phase_last_week_test')
        before=full_horizon_rollout(p,Predictor('mean7'))
        p.y[:]=np.array([1.,0.,0.,0.])
        after=full_horizon_rollout(p,Predictor('mean7'))
        np.testing.assert_array_equal(before,after)
        self.assertEqual(len(before),22*2)
        np.testing.assert_allclose(before.sum(1),1,atol=1e-6)

    def test_paper_metrics_population_first_and_unknown_exclusion(self):
        # Two people and two dates; the +1/-1 person errors on day one cancel
        # only after population aggregation. Last mode is the unknown channel.
        y=np.array([[.5,0,.25,.25,0],
                    [.5,0,.25,.25,0],
                    [.5,0,.25,.25,0],
                    [.5,0,.25,.25,0]],dtype=float)
        p=y.copy()
        p[0]=[.6,0,.3,.1,0]
        p[1]=[.4,0,.2,.4,0]
        dates=pd.to_datetime(['2020-01-01']*2+['2020-01-02']*2)
        m=paper_metrics(y,p,2,dates,np.full(4,24.),unknown_mode_index=2)
        self.assertAlmostEqual(m['macro']['POI_categories']['MAE_h_day'],0)
        self.assertAlmostEqual(m['macro']['travel_modes_known']['MAE_h_day'],0)
        self.assertEqual(m['macro']['travel_modes_known']['n_series'],2)
        self.assertAlmostEqual(m['distribution']['wMSE_travel'],.005)
        self.assertAlmostEqual(m['distribution']['wMSE_unknown_h'],0)
        self.assertGreater(m['distribution']['wKL_mode_all'],0)

    def test_ordinary_model_uses_no_urban_share_code(self):
        import torch
        from .models import (TabMAdapter,TimeXerAdapter,GRUAdapter,LSTMAdapter,
                             TransformerAdapter,LogisticNormalARAdapter)
        args=dict(flat_dim=40,output_dim=4,history=7,ctx_dim=3,width=16,depth=1,k=4,dropout=0)
        x=[torch.randn(2,40),torch.rand(2,7,4),torch.rand(2,7,3),torch.rand(2,3)]
        for cls in (TabMAdapter,TimeXerAdapter,GRUAdapter,LSTMAdapter,
                    TransformerAdapter,LogisticNormalARAdapter):
            extra={'past_ctx_dim':3} if cls in (
                GRUAdapter,LSTMAdapter,TransformerAdapter,LogisticNormalARAdapter) else {}
            model=cls(**args,**extra)
            pred=model(*x)
            self.assertEqual(pred.shape[-1],4)
            self.assertTrue(torch.isfinite(pred).all())
            modules=[type(m).__module__ for m in model.modules()]
            self.assertFalse(any('daily_share_model' in m or 'preference_' in m for m in modules))
            (pred**2).mean().backward()

    def test_logistic_normal_simplex_zero_support_and_likelihood(self):
        import torch
        from .models import LogisticNormalARAdapter
        model=LogisticNormalARAdapter(20,4,3,2,past_ctx_dim=2,eps=1e-4)
        flat=torch.zeros(5,20)
        history=torch.tensor([
            [[0.,0.,0.,0.],[.7,.2,.1,0.],[.6,.2,.1,.1]],
        ]).expand(5,-1,-1).clone()
        past=torch.zeros(5,3,2)
        current=torch.zeros(5,2)
        target=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[.1,.2,.3,.4],
                             [0.,0.,0.,1.],[.25,.25,.25,.25]])
        pred=model(flat,history,past,current)
        np.testing.assert_allclose(pred.detach().sum(-1),1,atol=1e-6)
        self.assertTrue(torch.isfinite(pred).all())
        loss=model.nll(flat,history,past,current,target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_lstm_and_transformer_are_finite_simplex_models(self):
        import torch
        from .models import LSTMAdapter,TransformerAdapter
        args=dict(flat_dim=40,output_dim=4,history=7,ctx_dim=3,width=16,
                  depth=1,k=4,dropout=0,past_ctx_dim=3)
        x=[torch.randn(3,40),torch.rand(3,7,4),torch.rand(3,7,3),torch.rand(3,3)]
        for cls in (LSTMAdapter,TransformerAdapter):
            model=cls(**args)
            pred=model(*x)
            self.assertEqual(pred.shape,(3,4))
            self.assertTrue(torch.isfinite(pred).all())
            np.testing.assert_allclose(pred.detach().sum(-1),1,atol=1e-6)
            modules=[type(m).__module__ for m in model.modules()]
            self.assertFalse(any('daily_share_model' in m or 'preference_' in m
                                 for m in modules))
            pred.square().mean().backward()
        with self.assertRaises(ValueError):
            TransformerAdapter(**{**args,'width':15})

    def test_mdcev_kkt_budget_and_likelihood_normalization(self):
        import torch
        from scipy.integrate import quad
        from .models import MDCEVAdapter
        model=MDCEVAdapter(1,2,1,1,draws=8)
        gamma=torch.tensor([.2,.4],dtype=torch.float64)
        v=torch.tensor([[1.,.5]],dtype=torch.float64)
        allocation=model.allocate(v,gamma)
        np.testing.assert_allclose(allocation.sum(-1),1,atol=1e-7)
        marginal=gamma*torch.exp(v)/(allocation+gamma)
        active=allocation[0]>0
        self.assertLess(float(marginal[0,active].max()-marginal[0,active].min()),1e-7)
        with torch.no_grad(): model.log_gamma.copy_(torch.log(torch.tensor([.1,.1])))
        def density(a):
            with torch.no_grad():
                return float(torch.exp(-model.nll(torch.zeros(1,1),torch.tensor([[a,1-a]],dtype=torch.float32))))
        total=density(0)+density(1)+quad(density,0,1,epsabs=1e-5)[0]
        self.assertAlmostEqual(total,1,places=4)
        loss=model.nll(torch.zeros(3,1),torch.tensor([[1.,0.],[0.,1.],[.3,.7]]))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__': unittest.main()
