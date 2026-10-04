"""Adapters around official code plus an explicitly identified MDCEV variant."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import numpy as np
import torch
from torch import nn

UPSTREAM_SOURCE = Path(__file__).resolve().parent/'third_party'


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


class TabMAdapter(nn.Module):
    def __init__(self,flat_dim,output_dim,history,ctx_dim,width=128,depth=2,k=16,dropout=.1):
        super().__init__()
        upstream=load_file('dso_vendor_tabm',UPSTREAM_SOURCE/'tabm/tabm.py')
        self.model=upstream.TabM.make(n_num_features=flat_dim,d_out=output_dim,
                                    n_blocks=depth,d_block=width,k=k,dropout=dropout)

    def forward(self,flat,h,past,current):
        return self.model(flat)


class TimeXerAdapter(nn.Module):
    """Official multivariate TimeXer, one-day prediction.

    Covariate tokens = historical calendar/static information plus target-day
    known covariates repeated across the past window. This is a documented
    covariate-interface adaptation; encoder/attention/forecast head are upstream.
    """
    def __init__(self,flat_dim,output_dim,history,ctx_dim,width=128,depth=2,k=16,dropout=.1):
        super().__init__()
        sys.path.insert(0,str(UPSTREAM_SOURCE/'TimeXer'))
        upstream=load_file('dso_vendor_timexer',UPSTREAM_SOURCE/'TimeXer/models/TimeXer.py')
        cfg=SimpleNamespace(task_name='short_term_forecast',features='M',seq_len=history,
              pred_len=1,use_norm=False,patch_len=7 if history%7==0 else 1,
              enc_in=output_dim,d_model=width,dropout=dropout,embed='timeF',freq='d',
              factor=1,n_heads=4,d_ff=width*2,activation='gelu',e_layers=depth)
        self.model=upstream.Model(cfg)

    def forward(self,flat,h,past,current):
        marks=torch.cat([past,current[:,None,:].expand(-1,h.shape[1],-1)],dim=-1)
        return self.model(h,marks,None,None)[:,0,:]


class GRUAdapter(nn.Module):
    """Standard person-history GRU with a target-day covariate head.

    This intentionally contains no UrbanShare components.  The recurrent path
    reads only the person's lagged joint shares and lagged public calendar
    fields.  Static demographics/ID and known target-day calendar fields enter
    through a separate linear context embedding before the ordinary forecast
    head.  A softmax supplies the common one-simplex output contract.
    """
    def __init__(self,flat_dim,output_dim,history,ctx_dim,width=128,depth=2,k=16,
                 dropout=.1,past_ctx_dim=12):
        super().__init__()
        self.gru=nn.GRU(output_dim+past_ctx_dim,width,num_layers=depth,
                        batch_first=True,dropout=dropout if depth>1 else 0.0)
        self.context=nn.Sequential(nn.Linear(ctx_dim,width),nn.GELU())
        self.head=nn.Sequential(
            nn.LayerNorm(width*2),nn.Linear(width*2,width),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(width,output_dim))

    def forward(self,flat,h,past,current):
        sequence=torch.cat([h,past],dim=-1)
        _,state=self.gru(sequence)
        logits=self.head(torch.cat([state[-1],self.context(current)],dim=-1))
        return torch.softmax(logits,dim=-1)


class LSTMAdapter(nn.Module):
    """Standard person-history LSTM with the same covariate interface as GRU."""
    def __init__(self,flat_dim,output_dim,history,ctx_dim,width=128,depth=2,k=16,
                 dropout=.1,past_ctx_dim=12):
        super().__init__()
        self.lstm=nn.LSTM(output_dim+past_ctx_dim,width,num_layers=depth,
                          batch_first=True,dropout=dropout if depth>1 else 0.0)
        self.context=nn.Sequential(nn.Linear(ctx_dim,width),nn.GELU())
        self.head=nn.Sequential(
            nn.LayerNorm(width*2),nn.Linear(width*2,width),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(width,output_dim))

    def forward(self,flat,h,past,current):
        sequence=torch.cat([h,past],dim=-1)
        _,(hidden,_)=self.lstm(sequence)
        logits=self.head(torch.cat([hidden[-1],self.context(current)],dim=-1))
        return torch.softmax(logits,dim=-1)


class TransformerAdapter(nn.Module):
    """Vanilla Transformer encoder over each person's causal history window.

    This is deliberately a plain Transformer baseline: linear token embedding,
    fixed sinusoidal positions, standard self-attention/feed-forward encoder
    blocks, and an ordinary context-conditioned forecast head.
    """
    def __init__(self,flat_dim,output_dim,history,ctx_dim,width=128,depth=2,k=16,
                 dropout=.1,past_ctx_dim=12,n_heads=4):
        super().__init__()
        if width%n_heads:
            raise ValueError('Transformer width must be divisible by n_heads')
        self.token=nn.Linear(output_dim+past_ctx_dim,width)
        positions=torch.arange(history,dtype=torch.float32)[:,None]
        frequencies=torch.exp(
            torch.arange(0,width,2,dtype=torch.float32)*(-np.log(10000.0)/width))
        encoding=torch.zeros(history,width)
        encoding[:,0::2]=torch.sin(positions*frequencies)
        encoding[:,1::2]=torch.cos(positions*frequencies[:encoding[:,1::2].shape[1]])
        self.register_buffer('position',encoding[None,:,:],persistent=False)
        layer=nn.TransformerEncoderLayer(
            d_model=width,nhead=n_heads,dim_feedforward=width*2,
            dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=depth,norm=nn.LayerNorm(width))
        self.context=nn.Sequential(nn.Linear(ctx_dim,width),nn.GELU())
        self.head=nn.Sequential(
            nn.LayerNorm(width*2),nn.Linear(width*2,width),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(width,output_dim))

    def forward(self,flat,h,past,current):
        sequence=torch.cat([h,past],dim=-1)
        encoded=self.encoder(self.token(sequence)+self.position[:,:sequence.shape[1]])
        logits=self.head(torch.cat([encoded[:,-1],self.context(current)],dim=-1))
        return torch.softmax(logits,dim=-1)


class LogisticNormalARAdapter(nn.Module):
    """Zero-adjusted logistic-normal autoregression for compositional shares.

    Lagged compositions are mapped to additive log-ratios (ALR), then a linear
    autoregression with exogenous calendar/static covariates predicts the next
    ALR location.  A learned global diagonal scale defines a logistic-normal
    likelihood.  Point forecasts use the inverse ALR at the predicted location.
    Exact zeros are handled only by the declared additive smoothing constant;
    no learned gates, mixtures, phase modules, or UrbanShare code are used.
    """
    def __init__(self,flat_dim,output_dim,history,ctx_dim,width=128,depth=2,k=16,
                 dropout=.1,past_ctx_dim=12,eps=1e-4):
        super().__init__()
        if output_dim<2:
            raise ValueError('Logistic-normal output requires at least two parts')
        if not 0<eps<.1:
            raise ValueError('eps must lie between 0 and 0.1')
        self.output_dim=output_dim
        self.eps=float(eps)
        input_dim=history*((output_dim-1)+past_ctx_dim)+ctx_dim
        self.location=nn.Linear(input_dim,output_dim-1)
        self.log_scale=nn.Parameter(torch.zeros(output_dim-1))

    def alr(self,composition):
        smooth=(composition+self.eps)/(1.0+self.output_dim*self.eps)
        return torch.log(smooth[...,:-1]/smooth[...,-1:])

    def features(self,h,past,current):
        return torch.cat([self.alr(h).flatten(1),past.flatten(1),current],dim=-1)

    def alr_location(self,h,past,current):
        return self.location(self.features(h,past,current))

    @staticmethod
    def inverse_alr(location):
        logits=torch.cat([location,torch.zeros_like(location[...,:1])],dim=-1)
        return torch.softmax(logits,dim=-1)

    def forward(self,flat,h,past,current):
        return self.inverse_alr(self.alr_location(h,past,current))

    def nll(self,flat,h,past,current,target):
        location=self.alr_location(h,past,current)
        z=self.alr(target)
        log_scale=self.log_scale.clamp(-7.0,5.0)
        standardized=(z-location)/torch.exp(log_scale)
        # Terms that depend only on the observed composition (including the
        # inverse-ALR Jacobian) do not affect fitted parameters and are omitted.
        return (.5*standardized.square()+log_scale).sum(-1).mean()


class MDCEVAdapter(nn.Module):
    """Gamma-profile, no outside good, alpha=0, sigma=1, prices=1.

    Independent Python implementation of Bhat's MDCEV likelihood, not BU-MDCEV
    and not an invocation of the downloaded rmdcev R/Stan package. All 18 goods
    (including travel modes) share a unit budget. Utility location identified
    by fixing the first alternative's systematic utility to zero.
    """
    def __init__(self,flat_dim,output_dim,history,ctx_dim,width=128,depth=2,k=16,dropout=.1,
                 draws=64,seed=1234):
        super().__init__()
        self.utility=nn.Linear(flat_dim,output_dim-1)
        nn.init.zeros_(self.utility.weight)
        nn.init.zeros_(self.utility.bias)
        self.log_gamma=nn.Parameter(torch.full((output_dim,),-3.0))
        rng=np.random.default_rng(seed)
        self.register_buffer('shocks',torch.tensor(rng.gumbel(size=(draws,output_dim)),dtype=torch.float32))

    def parameters_for(self,flat):
        v=torch.cat([torch.zeros_like(flat[:,:1]),self.utility(flat)],dim=1)
        gamma=self.log_gamma.clamp(-9,5).exp()
        return v,gamma

    def nll(self,flat,y):
        v,gamma=self.parameters_for(flat)
        used=(y>0).to(y.dtype)
        m=used.sum(1)
        adjusted=v-torch.log1p(y/gamma)
        log_jacobian=-(used*torch.log(y+gamma)).sum(1)+torch.log((used*(y+gamma)).sum(1))
        ll=log_jacobian+(used*adjusted).sum(1)-m*torch.logsumexp(adjusted,dim=1)+torch.lgamma(m)
        return -ll.mean()

    @staticmethod
    def allocate(v,gamma,budget=1.0):
        # KKT: x_j = gamma_j * max(exp(V_j)/lambda - 1, 0).
        utility=torch.exp(v-v.amax(dim=-1,keepdim=True))
        lo=torch.zeros_like(utility[...,:1]); hi=torch.ones_like(lo)
        for _ in range(48):
            mid=(lo+hi)/2
            x=gamma*torch.relu(utility/mid.clamp_min(1e-15)-1)
            too_much=x.sum(-1,keepdim=True)>budget
            lo=torch.where(too_much,mid,lo); hi=torch.where(too_much,hi,mid)
        return gamma*torch.relu(utility/hi.clamp_min(1e-15)-1)

    def forward(self,flat,h,past,current):
        v,gamma=self.parameters_for(flat)
        # Common fixed Gumbel draws keep validation and comparisons deterministic.
        return self.allocate(v[:,None,:]+self.shocks[None,:,:],gamma).mean(1)


NEURAL={
    'tabm':TabMAdapter,
    'timexer':TimeXerAdapter,
    'gru':GRUAdapter,
    'lstm':LSTMAdapter,
    'transformer':TransformerAdapter,
    'logistic_normal':LogisticNormalARAdapter,
    'mdcev':MDCEVAdapter,
}
