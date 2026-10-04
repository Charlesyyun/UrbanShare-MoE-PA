
from __future__ import annotations
import argparse, json, math, os, random, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch 
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from preference_dataset import DailyShareDataset, EPS

class TopKMoEHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_experts: int = 4, top_k: int = 2,
                 hidden_dim: Optional[int] = None, dropout: float = 0.1,
                 gate_temperature: float = 1.0, gate_noise_std: float = 0.0,
                 gate_dropout: float = 0.0):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        self.num_experts = int(num_experts)
        self.top_k = int(max(1, min(top_k, num_experts)))
        self.out_features = int(out_dim)
        self.gate_temperature = float(max(gate_temperature, 1e-6))
        self.gate_noise_std = float(max(gate_noise_std, 0.0))
        self.gate_dropout = float(min(max(gate_dropout, 0.0), 0.95))
        expert_hidden = int(hidden_dim if hidden_dim is not None else max(in_dim // 2, out_dim * 2, 32))
        self.gate = nn.Linear(in_dim, self.num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, expert_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, out_dim),
            )
            for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor, return_info: bool = False):
        gate_logits = self.gate(x)
        if self.training and self.gate_noise_std > 0.0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.gate_noise_std
        gate_logits = gate_logits / self.gate_temperature
        if self.training and self.gate_dropout > 0.0 and self.num_experts > 1:
            drop_mask = (torch.rand_like(gate_logits) < self.gate_dropout)
            all_dropped = drop_mask.all(dim=-1, keepdim=True)
            rescue_idx = torch.argmax(gate_logits, dim=-1, keepdim=True)
            drop_mask = torch.where(all_dropped, torch.zeros_like(drop_mask), drop_mask)
            drop_mask.scatter_(-1, rescue_idx, False)
            gate_logits = gate_logits.masked_fill(drop_mask, float('-inf'))
        gate_probs = torch.softmax(gate_logits, dim=-1)

        topk_vals, topk_idx = torch.topk(gate_probs, k=self.top_k, dim=-1)
        topk_vals = topk_vals / torch.clamp(topk_vals.sum(dim=-1, keepdim=True), min=1e-8)

        sparse_weights = torch.zeros_like(gate_probs)
        sparse_weights.scatter_(-1, topk_idx, topk_vals)

        expert_out = torch.stack([exp(x) for exp in self.experts], dim=1)
        out = torch.sum(expert_out * sparse_weights.unsqueeze(-1), dim=1)

        # Encourage balanced routing across experts.
        importance = gate_probs.mean(dim=0)
        load = (sparse_weights > 0).float().mean(dim=0)
        uni = torch.full_like(importance, 1.0 / self.num_experts)
        aux = torch.mean((importance - uni) ** 2) + torch.mean((load - uni) ** 2)
        if return_info:
            info = {
                "gate_probs": gate_probs,
                "topk_idx": topk_idx,
                "topk_vals": topk_vals,
                "sparse_weights": sparse_weights,
            }
            return out, aux, info
        return out, aux

class PreferenceNet(nn.Module):
    def __init__(self, n_agents:int, n_phases:int, ctx_dim:int, n_cat:int, n_mode:int,
                 emb_dim:int=32, h_dim:int=128, dropout:float=0.1,
                 mode_known_idx: Optional[List[int]]=None, unk_mode_idx: Optional[int]=None,
                 moe_num_experts: int = 4, moe_top_k: int = 2,
                 moe_gate_temperature: float = 1.0,
                 moe_gate_noise_std: float = 0.0,
                 moe_gate_dropout: float = 0.0,
                 moe_cat_top_k: Optional[int] = None,
                 moe_mode_top_k: Optional[int] = None,
                 moe_cat_gate_temperature: Optional[float] = None,
                 moe_mode_gate_temperature: Optional[float] = None,
                 moe_cat_gate_noise_std: Optional[float] = None,
                 moe_mode_gate_noise_std: Optional[float] = None,
                 moe_cat_gate_dropout: Optional[float] = None,
                 moe_mode_gate_dropout: Optional[float] = None,
                 use_film: bool = True,
                 use_moe_heads: bool = True,
                 use_phase_lag_gate: bool = True,
                 detach_travel_head_context: bool = True):
        super().__init__()
        self.register_buffer(
            "mode_known_idx",
            torch.tensor(mode_known_idx or list(range(n_mode-1)), dtype=torch.long),
            persistent=False
        )
        self.unk_mode_idx = int(unk_mode_idx if unk_mode_idx is not None else (n_mode-1))
        self.use_film = bool(use_film)
        self.use_moe_heads = bool(use_moe_heads)
        self.use_phase_lag_gate = bool(use_phase_lag_gate)
        self.detach_travel_head_context = bool(detach_travel_head_context)
        self.agent_emb = nn.Embedding(n_agents, emb_dim)
        self.phase_emb = nn.Embedding(max(n_phases, 1), emb_dim)
        # FiLM: phase directly modulates the context representation multiplicatively
        self.phase_film_scale = nn.Linear(emb_dim, h_dim)
        self.phase_film_bias  = nn.Linear(emb_dim, h_dim)
        # Phase-gated lag feature suppression: phase controls how much lag features influence context
        # Applied at input level before context encoder, so phase can override behavioral inertia
        self.n_lag_features = 0  # set after construction via set_lag_feature_count
        self.phase_lag_gate = None  # initialized lazily via set_lag_feature_count
        self.moe_num_experts = int(moe_num_experts)
        self.moe_top_k = int(max(1, min(moe_top_k, self.moe_num_experts)))
        self.moe_gate_temperature = float(max(moe_gate_temperature, 1e-6))
        self.moe_gate_noise_std = float(max(moe_gate_noise_std, 0.0))
        self.moe_gate_dropout = float(min(max(moe_gate_dropout, 0.0), 0.95))
        self.moe_cat_top_k = int(max(1, min(moe_cat_top_k if moe_cat_top_k is not None else self.moe_top_k, self.moe_num_experts)))
        self.moe_mode_top_k = int(max(1, min(moe_mode_top_k if moe_mode_top_k is not None else self.moe_top_k, self.moe_num_experts)))
        self.moe_cat_gate_temperature = float(max(moe_cat_gate_temperature if moe_cat_gate_temperature is not None else self.moe_gate_temperature, 1e-6))
        self.moe_mode_gate_temperature = float(max(moe_mode_gate_temperature if moe_mode_gate_temperature is not None else self.moe_gate_temperature, 1e-6))
        self.moe_cat_gate_noise_std = float(max(moe_cat_gate_noise_std if moe_cat_gate_noise_std is not None else self.moe_gate_noise_std, 0.0))
        self.moe_mode_gate_noise_std = float(max(moe_mode_gate_noise_std if moe_mode_gate_noise_std is not None else self.moe_gate_noise_std, 0.0))
        self.moe_cat_gate_dropout = float(min(max(moe_cat_gate_dropout if moe_cat_gate_dropout is not None else self.moe_gate_dropout, 0.0), 0.95))
        self.moe_mode_gate_dropout = float(min(max(moe_mode_gate_dropout if moe_mode_gate_dropout is not None else self.moe_gate_dropout, 0.0), 0.95))
        self.moe_cat_aux_weight = 1.0
        self.moe_mode_aux_weight = 1.0
        self.ctx = nn.Sequential(
            nn.Linear(ctx_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        fused_dim = emb_dim*2 + h_dim
        self.fused_dim = fused_dim
        if self.use_moe_heads:
            self.head_cat = TopKMoEHead(
                fused_dim + 1, n_cat,
                num_experts=self.moe_num_experts,
                top_k=self.moe_cat_top_k,
                hidden_dim=h_dim,
                dropout=dropout,
                gate_temperature=self.moe_cat_gate_temperature,
                gate_noise_std=self.moe_cat_gate_noise_std,
                gate_dropout=self.moe_cat_gate_dropout,
            )
            self.head_mode = TopKMoEHead(
                fused_dim + 1, n_mode,
                num_experts=self.moe_num_experts,
                top_k=self.moe_mode_top_k,
                hidden_dim=h_dim,
                dropout=dropout,
                gate_temperature=self.moe_mode_gate_temperature,
                gate_noise_std=self.moe_mode_gate_noise_std,
                gate_dropout=self.moe_mode_gate_dropout,
            )
        else:
            self.head_cat = nn.Sequential(
                nn.Linear(fused_dim + 1, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h_dim, n_cat),
            )
            self.head_mode = nn.Sequential(
                nn.Linear(fused_dim + 1, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h_dim, n_mode),
            )
        self.head_travel = nn.Linear(fused_dim, 1)
        self.known_mass_gate = nn.Linear((emb_dim*2 + h_dim) + 1, 1)

    def set_lag_feature_indices(self, lag_indices: List[int]):
        if not self.use_phase_lag_gate:
            self.n_lag_features = 0
            return
        self.n_lag_features = len(lag_indices)
        if self.n_lag_features > 0:
            self.phase_lag_gate = nn.Linear(self.phase_emb.embedding_dim, self.n_lag_features)
            self.register_buffer("lag_feat_indices",
                                 torch.tensor(lag_indices, dtype=torch.long), persistent=True)

    def forward(self, agent_idx, phase_id, x_ctx,
                mode_temperature: float = 1.0,
                cat_temperature: float = 1.0,
                return_aux: bool = False,
                return_routing: bool = False):
        e = self.agent_emb(agent_idx)
        p = self.phase_emb(phase_id)
        # Phase-gated lag suppression: phase controls lag feature influence before encoding
        if self.use_phase_lag_gate and self.phase_lag_gate is not None and hasattr(self, "lag_feat_indices"):
            lag_gate = torch.sigmoid(self.phase_lag_gate(p))  # [B, n_lag]
            x_ctx = x_ctx.clone()
            x_ctx[:, self.lag_feat_indices] = x_ctx[:, self.lag_feat_indices] * lag_gate
        g = self.ctx(x_ctx)
        # FiLM modulation: phase gates context element-wise (scale + shift)
        # tanh keeps scale in (-1,1) → output in (0, 2)*g+bias, avoids exploding
        if self.use_film:
            film_scale = torch.tanh(self.phase_film_scale(p))   # [B, h_dim]
            film_bias  = self.phase_film_bias(p)                # [B, h_dim]
            g = g * (1.0 + film_scale) + film_bias
        z = torch.cat([e, p, g], dim=-1)

        travel = torch.sigmoid(self.head_travel(z))  # [B,1]

        # MODE head
        travel_context = travel.detach() if self.detach_travel_head_context else travel
        z_mode = torch.cat([z, travel_context], dim=-1)
        if self.use_moe_heads and return_routing:
            mode_logits, aux_mode, mode_info = self.head_mode(z_mode, return_info=True)
        elif self.use_moe_heads:
            mode_logits, aux_mode = self.head_mode(z_mode)
            mode_info = None
        else:
            mode_logits = self.head_mode(z_mode)
            aux_mode = z_mode.new_zeros(())
            mode_info = {
                "gate_probs": z_mode.new_ones((z_mode.size(0), 1)),
                "topk_idx": torch.zeros((z_mode.size(0), 1), dtype=torch.long, device=z_mode.device),
                "topk_vals": z_mode.new_ones((z_mode.size(0), 1)),
                "sparse_weights": z_mode.new_ones((z_mode.size(0), 1)),
            } if return_routing else None
        idx = self.mode_known_idx 

        
        alpha = torch.sigmoid(self.known_mass_gate(z_mode)) 

        # sparse distribution over known modes (sums to 1 over known), then scale by α
        logits_known = mode_logits.index_select(-1, idx)
        pm_known_dir = sparsemax(logits_known / max(1e-6, mode_temperature), dim=-1)  
        pm_known = pm_known_dir * alpha                                           

        # assemble full pm
        pm = torch.zeros_like(mode_logits)
        pm.scatter_(-1, idx.unsqueeze(0).expand(pm_known.size(0), -1), pm_known)
        # unknown gets the slack
        pm[..., self.unk_mode_idx] = (1.0 - alpha).squeeze(-1)  


        # category head sees stay gate
        stay_gate = (1.0 - travel).detach() if self.detach_travel_head_context else (1.0 - travel)
        z_cat = torch.cat([z, stay_gate], dim=-1)
        if self.use_moe_heads and return_routing:
            cat_logits, aux_cat, cat_info = self.head_cat(z_cat, return_info=True)
        elif self.use_moe_heads:
            cat_logits, aux_cat = self.head_cat(z_cat)
            cat_info = None
        else:
            cat_logits = self.head_cat(z_cat)
            aux_cat = z_cat.new_zeros(())
            cat_info = {
                "gate_probs": z_cat.new_ones((z_cat.size(0), 1)),
                "topk_idx": torch.zeros((z_cat.size(0), 1), dtype=torch.long, device=z_cat.device),
                "topk_vals": z_cat.new_ones((z_cat.size(0), 1)),
                "sparse_weights": z_cat.new_ones((z_cat.size(0), 1)),
            } if return_routing else None
        pc = sparsemax(cat_logits / max(1e-6, cat_temperature), dim=-1)


        aux_cat_w = 1.0
        aux_mode_w = 1.0
        if hasattr(self, "moe_cat_aux_weight"):
            aux_cat_w = self.moe_cat_aux_weight
            aux_mode_w = self.moe_mode_aux_weight
        aux = aux_mode_w * aux_mode + aux_cat_w * aux_cat
        if return_routing:
            routing = {
                "mode": mode_info,
                "cat": cat_info,
            }
            if return_aux:
                return pc, pm, travel, aux, routing
            return pc, pm, travel, routing
        if return_aux:
            return pc, pm, travel, aux
        return pc, pm, travel


# Loss & Metrics
def _renorm_known(t: torch.Tensor, idx: List[int], eps: float = 1e-8) -> torch.Tensor:
    x = t[..., idx]
    denom = torch.clamp(x.sum(dim=-1, keepdim=True), min=eps)
    return x / denom

def safe_weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8):
    num = torch.sum(values * weights)
    den = torch.sum(weights).clamp_min(eps)
    return num / den

def kl_divergence(p, q, eps=1e-8):
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * torch.log(p / q), dim=-1)

def r2_score(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    ss_res = np.sum((y_true - y_pred)**2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0))**2, axis=0) + 1e-12
    r2 = 1.0 - (ss_res / ss_tot)
    return float(np.mean(r2))


def sparsemax(logits, dim=-1):
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    k = torch.arange(1, z.size(dim)+1, device=z.device, dtype=z.dtype).view(
        *([1]*(z.dim()-1)), -1
    )
    cssv = torch.cumsum(z_sorted, dim=dim) - 1
    cond = z_sorted > cssv / k
    k_z = cond.sum(dim=dim, keepdim=True)
    tau = cssv.gather(dim, k_z-1) / k_z
    return torch.clamp(z - tau, min=0)



@dataclass
class TrainConfig:
    epochs:int=60
    batch_size:int=1024
    lr:float=1e-3
    weight_decay:float=1e-5
    emb_dim:int=32
    h_dim:int=128
    dropout:float=0.1
    val_days:int=14
    test_days:int=14
    split_mode:str="phase_last_week_test"
    holdout_phase:Optional[int]=None
    transition_backtest_idx:Optional[int]=None
    agent_val_frac:float=0.10
    agent_test_frac:float=0.10
    seed:int=1234
    num_workers:int=8
    device:str="cuda"
    warmup_epochs:int=5
    moe_num_experts:int=4
    moe_top_k:int=2
    moe_aux_weight:float=5e-2
    moe_gate_temperature:float=1.25
    moe_gate_noise_std:float=0.15
    moe_gate_dropout:float=0.02
    moe_cat_top_k:Optional[int]=None
    moe_mode_top_k:Optional[int]=None
    moe_cat_aux_weight:Optional[float]=None
    moe_mode_aux_weight:Optional[float]=None
    moe_cat_gate_temperature:Optional[float]=None
    moe_mode_gate_temperature:Optional[float]=None
    moe_cat_gate_noise_std:Optional[float]=None
    moe_mode_gate_noise_std:Optional[float]=None
    moe_cat_gate_dropout:Optional[float]=None
    moe_mode_gate_dropout:Optional[float]=None
    rollout_eval_every:int=5
    rollout_weight:float=1.0
    default_checkpoint_metric:str="balanced"
    use_film:bool=True
    use_moe_heads:bool=True
    use_phase_lag_gate:bool=True
    detach_travel_head_context:bool=True
    # Phase-sensitivity contrastive loss: penalise model if it produces
    # the same distribution for different phases given the same context.
    phase_sens_weight:float=0.10
    phase_sens_margin:float=0.05

def _normalize_device(device: str) -> str:
    return f"cuda:{device}" if device.isdigit() else device

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _loader_has_data(loader: Optional[DataLoader]) -> bool:
    if loader is None:
        return False
    dataset = getattr(loader, "dataset", None)
    try:
        return dataset is not None and len(dataset) > 0
    except TypeError:
        return False


def _split_phase_dates_last_week_test(dates: List):
    n_dates = len(dates)
    if n_dates == 0:
        return [], [], []
    if n_dates == 1:
        return list(dates), [], []
    n_test = min(7, max(1, n_dates - 1))
    test_cut = list(dates[-n_test:])
    train_cut = list(dates[:-n_test])
    if not train_cut:
        train_cut = [test_cut[0]]
        test_cut = test_cut[1:]
    return train_cut, [], test_cut

def _split_phase_dates_temporal(dates: List, val_days: int, test_days: int):
    n_dates = len(dates)
    if n_dates == 0:
        return [], [], []

    # Keep temporal order inside each phase while ensuring small phases still
    # contribute to validation/test when possible.
    if n_dates >= (val_days + test_days + 1):
        n_test = min(test_days, max(0, n_dates - 2)) if test_days > 0 else 0
        n_val = min(val_days, max(0, n_dates - n_test - 1)) if val_days > 0 else 0
    else:
        if n_dates == 1:
            return list(dates), [], []
        if n_dates == 2:
            return [dates[0]], [], [dates[1]]
        n_test = min(test_days, max(1, int(round(0.15 * n_dates)))) if test_days > 0 else 0
        n_val = min(val_days, max(1, int(round(0.15 * n_dates)))) if val_days > 0 else 0
        if n_test + n_val >= n_dates:
            overflow = n_test + n_val - (n_dates - 1)
            if n_val >= n_test and n_val > 0:
                n_val -= overflow
            else:
                n_test -= overflow
            n_test = max(n_test, 0)
            n_val = max(n_val, 0)

    test_cut = list(dates[-n_test:]) if n_test > 0 else []
    remain = list(dates[:-n_test]) if n_test > 0 else list(dates)
    val_cut = remain[-n_val:] if n_val > 0 else []
    train_cut = remain[:-n_val] if n_val > 0 else remain

    if not train_cut and val_cut:
        train_cut = [val_cut[0]]
        val_cut = val_cut[1:]
    if not train_cut and test_cut:
        train_cut = [test_cut[0]]
        test_cut = test_cut[1:]
    return train_cut, val_cut, test_cut

def _split_phase_dates_block_random(dates: List, val_days: int, test_days: int, seed: int, phase_token: int = 0):
    n_dates = len(dates)
    if n_dates == 0:
        return [], [], []
    if n_dates == 1:
        return list(dates), [], []
    if n_dates == 2:
        return [dates[0]], [], [dates[1]]

    ds = pd.Series(pd.to_datetime(pd.Series(dates))).dt.normalize()
    block_keys = ds.dt.to_period("W-SUN")
    block_to_dates: Dict[object, List] = {}
    for d, b in zip(ds.tolist(), block_keys.tolist()):
        block_to_dates.setdefault(b, []).append(d.date())
    blocks = list(block_to_dates.keys())
    n_blocks = len(blocks)
    if n_blocks == 1:
        return _split_phase_dates_temporal([d.date() for d in ds.tolist()], val_days, test_days)

    n_test = int(np.ceil(max(test_days, 0) / 7.0)) if test_days > 0 else 0
    n_val = int(np.ceil(max(val_days, 0) / 7.0)) if val_days > 0 else 0
    n_test = min(n_test, max(0, n_blocks - 2)) if n_blocks >= 3 else min(n_test, 1)
    n_val = min(n_val, max(0, n_blocks - n_test - 1))
    if n_test == 0 and test_days > 0 and n_blocks >= 3:
        n_test = 1
    if n_val == 0 and val_days > 0 and n_blocks - n_test >= 2:
        n_val = 1
    while n_test + n_val >= n_blocks and (n_test > 0 or n_val > 0):
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1

    rng = np.random.RandomState(int(seed) + int(phase_token) * 9973 + n_blocks * 31)
    perm = rng.permutation(n_blocks).tolist()
    test_blocks = set(perm[:n_test])
    val_blocks = set(perm[n_test:n_test + n_val])
    train_blocks = [i for i in range(n_blocks) if i not in test_blocks and i not in val_blocks]
    if not train_blocks:
        if val_blocks:
            moved = sorted(val_blocks)[0]
            val_blocks.remove(moved)
            train_blocks = [moved]
        elif test_blocks:
            moved = sorted(test_blocks)[0]
            test_blocks.remove(moved)
            train_blocks = [moved]

    train_cut, val_cut, test_cut = [], [], []
    for i, block in enumerate(blocks):
        target = train_cut if i in train_blocks else val_cut if i in val_blocks else test_cut
        target.extend(block_to_dates[block])
    return sorted(train_cut), sorted(val_cut), sorted(test_cut)


def _split_agent_ids(agent_ids: List[str], val_frac: float, test_frac: float, seed: int):
    agents = [str(a) for a in agent_ids]
    n_agents = len(agents)
    if n_agents == 0:
        return [], [], []
    if n_agents == 1:
        return agents, [], []

    test_frac = float(min(max(test_frac, 0.0), 0.9))
    val_frac = float(min(max(val_frac, 0.0), 0.9))
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n_agents).tolist()

    n_test = int(round(n_agents * test_frac))
    n_val = int(round(n_agents * val_frac))
    if test_frac > 0.0 and n_agents >= 3:
        n_test = max(1, n_test)
    if val_frac > 0.0 and n_agents >= 4:
        n_val = max(1, n_val)
    n_test = min(n_test, max(0, n_agents - 2))
    n_val = min(n_val, max(0, n_agents - n_test - 1))
    while n_test + n_val >= n_agents and (n_test > 0 or n_val > 0):
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1

    test_agents = [agents[i] for i in perm[:n_test]]
    val_agents = [agents[i] for i in perm[n_test:n_test + n_val]]
    train_agents = [agents[i] for i in perm[n_test + n_val:]]
    if not train_agents:
        if val_agents:
            train_agents = [val_agents.pop(0)]
        elif test_agents:
            train_agents = [test_agents.pop(0)]
    return sorted(train_agents), sorted(val_agents), sorted(test_agents)


def _summarize_split(df: pd.DataFrame, tr_idx: np.ndarray, va_idx: np.ndarray, te_idx: np.ndarray):
    def _subset_summary(indices: np.ndarray):
        subset = df.loc[indices] if len(indices) > 0 else df.iloc[0:0]
        summary = {
            "rows": int(len(subset)),
            "dates": int(subset["date"].nunique()) if "date" in subset.columns else 0,
            "phases": sorted(int(p) for p in subset["epi_phase"].dropna().unique().tolist()) if "epi_phase" in subset.columns else [],
        }
        if "agent_id" in subset.columns:
            summary["agents"] = int(subset["agent_id"].astype(str).nunique())
        return summary

    return {
        "train": _subset_summary(tr_idx),
        "val": _subset_summary(va_idx),
        "test": _subset_summary(te_idx),
    }


def _global_phase_transitions(df: pd.DataFrame) -> tuple[pd.DataFrame, List[Dict[str, object]]]:
    phase_df = df[["date", "epi_phase"]].copy()
    phase_df["date"] = pd.to_datetime(phase_df["date"]).dt.date
    phase_df = phase_df.dropna(subset=["epi_phase"])
    if phase_df.empty:
        return pd.DataFrame(columns=["date", "epi_phase"]), []
    timeline = (
        phase_df.groupby("date", as_index=False)["epi_phase"]
        .agg(lambda s: int(pd.Series(s).mode(dropna=False).iloc[0]))
        .sort_values("date")
        .reset_index(drop=True)
    )
    transitions: List[Dict[str, object]] = []
    prev_phase: Optional[int] = None
    for row in timeline.itertuples(index=False):
        cur_phase = int(row.epi_phase)
        if prev_phase is not None and cur_phase != prev_phase:
            transitions.append(
                {
                    "transition_idx": len(transitions),
                    "transition_date": row.date,
                    "from_phase": int(prev_phase),
                    "to_phase": cur_phase,
                }
            )
        prev_phase = cur_phase
    return timeline, transitions


def _resolve_transition_backtest(
    df: pd.DataFrame,
    val_days: int,
    test_days: int,
    transition_backtest_idx: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    timeline, transitions = _global_phase_transitions(df)
    if not transitions:
        raise ValueError("transition_backtest requires at least one global phase transition in the dataset.")

    selected_idx = len(transitions) - 1 if transition_backtest_idx is None else int(transition_backtest_idx)
    if selected_idx < 0 or selected_idx >= len(transitions):
        raise ValueError(
            f"transition_backtest_idx={selected_idx} is invalid for {len(transitions)} transitions."
        )
    selected = transitions[selected_idx]
    all_dates = timeline["date"].tolist()
    test_start = selected["transition_date"]
    n_test_days = max(1, int(test_days))
    n_val_days = max(1, int(val_days))

    pre_dates = [d for d in all_dates if d < test_start]
    test_cut = [d for d in all_dates if d >= test_start][:n_test_days]
    if not test_cut:
        raise ValueError(f"No test dates were found on or after transition date {test_start}.")
    val_cut = pre_dates[-n_val_days:] if pre_dates else []
    train_cut = [d for d in pre_dates if d not in set(val_cut)]
    if not train_cut and len(val_cut) > 1:
        train_cut = val_cut[:-1]
        val_cut = val_cut[-1:]
    if not train_cut:
        raise ValueError(
            f"transition_backtest at {test_start} produced an empty train split; "
            "choose an earlier transition or smaller val_days."
        )

    tr_idx = np.sort(df.index[df["date"].isin(train_cut)].to_numpy()) if train_cut else np.array([], dtype=int)
    va_idx = np.sort(df.index[df["date"].isin(val_cut)].to_numpy()) if val_cut else np.array([], dtype=int)
    te_idx = np.sort(df.index[df["date"].isin(test_cut)].to_numpy()) if test_cut else np.array([], dtype=int)
    detail = {
        "transition_idx": int(selected["transition_idx"]),
        "transition_date": str(test_start),
        "from_phase": int(selected["from_phase"]),
        "to_phase": int(selected["to_phase"]),
        "n_transitions_total": int(len(transitions)),
        "train_end_date": str(max(train_cut)) if train_cut else "",
        "val_start_date": str(min(val_cut)) if val_cut else "",
        "val_end_date": str(max(val_cut)) if val_cut else "",
        "test_start_date": str(min(test_cut)) if test_cut else "",
        "test_end_date": str(max(test_cut)) if test_cut else "",
    }
    return tr_idx, va_idx, te_idx, detail


def split_by_date_indices(
    daily_df: pd.DataFrame,
    val_days:int,
    test_days:int,
    split_mode:str="phase_block_random",
    seed:int=1234,
    holdout_phase: Optional[int] = None,
    transition_backtest_idx: Optional[int] = None,
    agent_val_frac: float = 0.10,
    agent_test_frac: float = 0.10,
):
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "agent_id" in df.columns:
        df["agent_id"] = df["agent_id"].astype(str)

    if split_mode == "global_temporal":
        all_dates = sorted(df["date"].unique().tolist())
        train_cut, val_cut, test_cut = _split_phase_dates_temporal(all_dates, val_days, test_days)
        tr_idx = np.sort(df.index[df["date"].isin(train_cut)].to_numpy()) if train_cut else np.array([], dtype=int)
        va_idx = np.sort(df.index[df["date"].isin(val_cut)].to_numpy()) if val_cut else np.array([], dtype=int)
        te_idx = np.sort(df.index[df["date"].isin(test_cut)].to_numpy()) if test_cut else np.array([], dtype=int)
        return tr_idx, va_idx, te_idx

    if split_mode == "leave_one_phase_out":
        phase_values = sorted(int(p) for p in df["epi_phase"].dropna().unique().tolist())
        if not phase_values:
            raise ValueError("leave_one_phase_out requires at least one non-null epi_phase value.")
        selected_phase = int(phase_values[-1] if holdout_phase is None else holdout_phase)
        if selected_phase not in phase_values:
            raise ValueError(f"holdout_phase={selected_phase} is not present in the dataset phases {phase_values}.")
        test_mask = df["epi_phase"] == selected_phase
        remainder = df.loc[~test_mask]
        rem_dates = sorted(remainder["date"].unique().tolist())
        train_cut, val_cut, _ = _split_phase_dates_temporal(rem_dates, val_days, 0)
        tr_idx = np.sort(remainder.index[remainder["date"].isin(train_cut)].to_numpy()) if train_cut else np.array([], dtype=int)
        va_idx = np.sort(remainder.index[remainder["date"].isin(val_cut)].to_numpy()) if val_cut else np.array([], dtype=int)
        te_idx = np.sort(df.index[test_mask].to_numpy())
        return tr_idx, va_idx, te_idx

    if split_mode == "transition_backtest":
        tr_idx, va_idx, te_idx, _ = _resolve_transition_backtest(
            df,
            val_days=val_days,
            test_days=test_days,
            transition_backtest_idx=transition_backtest_idx,
        )
        return tr_idx, va_idx, te_idx

    if split_mode == "unseen_agent":
        if "agent_id" not in df.columns:
            raise ValueError("unseen_agent split requires an agent_id column.")
        train_agents, val_agents, test_agents = _split_agent_ids(
            sorted(df["agent_id"].astype(str).unique().tolist()),
            val_frac=agent_val_frac,
            test_frac=agent_test_frac,
            seed=seed,
        )
        tr_idx = np.sort(df.index[df["agent_id"].isin(train_agents)].to_numpy()) if train_agents else np.array([], dtype=int)
        va_idx = np.sort(df.index[df["agent_id"].isin(val_agents)].to_numpy()) if val_agents else np.array([], dtype=int)
        te_idx = np.sort(df.index[df["agent_id"].isin(test_agents)].to_numpy()) if test_agents else np.array([], dtype=int)
        return tr_idx, va_idx, te_idx

    tr_parts = []
    va_parts = []
    te_parts = []

    for phase in sorted(df["epi_phase"].dropna().unique().tolist()):
        phase_df = df[df["epi_phase"] == phase]
        phase_dates = sorted(phase_df["date"].unique().tolist())
        phase_token = int(phase) if pd.notna(phase) else 0
        if split_mode == "phase_last_week_test":
            train_cut, val_cut, test_cut = _split_phase_dates_last_week_test(phase_dates)
        elif split_mode == "phase_temporal":
            train_cut, val_cut, test_cut = _split_phase_dates_temporal(phase_dates, val_days, test_days)
        else:
            train_cut, val_cut, test_cut = _split_phase_dates_block_random(phase_dates, val_days, test_days, seed=seed, phase_token=phase_token)

        if train_cut:
            tr_parts.append(phase_df.index[phase_df["date"].isin(train_cut)].to_numpy())
        if val_cut:
            va_parts.append(phase_df.index[phase_df["date"].isin(val_cut)].to_numpy())
        if test_cut:
            te_parts.append(phase_df.index[phase_df["date"].isin(test_cut)].to_numpy())

    # Any rows without a valid phase fall back to the same phase-aware date logic.
    remainder = df[df["epi_phase"].isna()]
    if not remainder.empty:
        rem_dates = sorted(remainder["date"].unique().tolist())
        if split_mode == "phase_last_week_test":
            train_cut, val_cut, test_cut = _split_phase_dates_last_week_test(rem_dates)
        elif split_mode == "phase_temporal":
            train_cut, val_cut, test_cut = _split_phase_dates_temporal(rem_dates, val_days, test_days)
        else:
            train_cut, val_cut, test_cut = _split_phase_dates_block_random(rem_dates, val_days, test_days, seed=seed, phase_token=99991)
        if train_cut:
            tr_parts.append(remainder.index[remainder["date"].isin(train_cut)].to_numpy())
        if val_cut:
            va_parts.append(remainder.index[remainder["date"].isin(val_cut)].to_numpy())
        if test_cut:
            te_parts.append(remainder.index[remainder["date"].isin(test_cut)].to_numpy())

    tr_idx = np.sort(np.concatenate(tr_parts)) if tr_parts else np.array([], dtype=int)
    va_idx = np.sort(np.concatenate(va_parts)) if va_parts else np.array([], dtype=int)
    te_idx = np.sort(np.concatenate(te_parts)) if te_parts else np.array([], dtype=int)
    return tr_idx, va_idx, te_idx

def make_loaders(dataset: DailyShareDataset, cfg: TrainConfig, out_dir: Path):
    tr_idx, va_idx, te_idx = split_by_date_indices(
        dataset.daily,
        cfg.val_days,
        cfg.test_days,
        split_mode=cfg.split_mode,
        seed=cfg.seed,
        holdout_phase=cfg.holdout_phase,
        transition_backtest_idx=cfg.transition_backtest_idx,
        agent_val_frac=cfg.agent_val_frac,
        agent_test_frac=cfg.agent_test_frac,
    )
    pos_index = pd.Index(dataset.daily.index)
    tr_pos = pos_index.get_indexer(tr_idx)
    va_pos = pos_index.get_indexer(va_idx)
    te_pos = pos_index.get_indexer(te_idx)
    assert np.all(tr_pos >= 0) and np.all(va_pos >= 0) and np.all(te_pos >= 0), \
        "Split indices could not be mapped to dataset positions."
    pd.DataFrame({"pos": tr_pos}).to_csv(out_dir/"train_indices_pos.csv", index=False)
    pd.DataFrame({"pos": va_pos}).to_csv(out_dir/"val_indices_pos.csv", index=False)
    pd.DataFrame({"pos": te_pos}).to_csv(out_dir/"test_indices_pos.csv", index=False)
    
    pd.DataFrame({"idx":tr_idx}).to_csv(out_dir/"train_indices.csv", index=False)
    pd.DataFrame({"idx":va_idx}).to_csv(out_dir/"val_indices.csv", index=False)
    pd.DataFrame({"idx":te_idx}).to_csv(out_dir/"test_indices.csv", index=False)
    split_config = {
        "split_mode": cfg.split_mode,
        "val_days": cfg.val_days,
        "test_days": cfg.test_days,
        "seed": cfg.seed,
        "holdout_phase": cfg.holdout_phase,
        "transition_backtest_idx": cfg.transition_backtest_idx,
        "agent_val_frac": cfg.agent_val_frac,
        "agent_test_frac": cfg.agent_test_frac,
    }
    split_summary = _summarize_split(dataset.daily.copy(), tr_idx, va_idx, te_idx)
    if cfg.split_mode == "transition_backtest":
        _, _, _, detail = _resolve_transition_backtest(
            dataset.daily.copy(),
            val_days=cfg.val_days,
            test_days=cfg.test_days,
            transition_backtest_idx=cfg.transition_backtest_idx,
        )
        split_config["transition_backtest"] = detail
        split_summary["transition_backtest"] = detail
    (out_dir / "split_config.json").write_text(json.dumps(split_config, indent=2))
    (out_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2))
    train_loader = DataLoader(Subset(dataset, tr_pos), batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(Subset(dataset, va_pos), batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(Subset(dataset, te_pos), batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader, test_loader

class RolloutLagRoller:
    def __init__(self, ctx_cols: List[str], home_pos: Optional[int]):
        self.ctx_cols = set(ctx_cols)
        self.home = []
        self.travel = []
        self.cat_rings: Dict[int, List[float]] = {}
        self.home_pos = home_pos

    def seed_from_row(self, row: pd.Series):
        if "prev_home_share" in self.ctx_cols:
            self.home = [float(row.get("prev_home_share", 0.0))] * 7
        if "prev_travel_frac" in self.ctx_cols:
            self.travel = [float(row.get("prev_travel_frac", 0.0))] * 7
        for c in row.index:
            if c.startswith("prev_cat_") and not c.startswith("prev7_"):
                try:
                    k = int(c.split("_")[-1])
                    self.cat_rings[k] = [float(row[c])] * 7
                except Exception:
                    pass

    def _push7(self, ring: List[float], v: float):
        ring.append(float(v))
        if len(ring) > 7:
            ring.pop(0)

    def update_next_row(self, next_ctx: pd.Series, pc_row: np.ndarray, t_val: float):
        if "prev_travel_frac" in self.ctx_cols:
            next_ctx["prev_travel_frac"] = float(t_val)
            self._push7(self.travel, t_val)
            if "prev7_travel_frac_mean" in self.ctx_cols:
                next_ctx["prev7_travel_frac_mean"] = float(np.mean(self.travel))
        if "prev_home_share" in self.ctx_cols and self.home_pos is not None:
            h = float(pc_row[self.home_pos]) if self.home_pos < len(pc_row) else 0.0
            next_ctx["prev_home_share"] = h
            self._push7(self.home, h)
            if "prev7_home_share_mean" in self.ctx_cols:
                next_ctx["prev7_home_share_mean"] = float(np.mean(self.home))
        for c in list(next_ctx.index):
            if c.startswith("prev_cat_") and not c.startswith("prev7_"):
                try:
                    k = int(c.split("_")[-1])
                    v = float(pc_row[k]) if k < len(pc_row) else 0.0
                    next_ctx[c] = v
                    ring = self.cat_rings.setdefault(k, [])
                    self._push7(ring, v)
                    k7 = f"prev7_cat_{k}_mean"
                    if k7 in self.ctx_cols:
                        next_ctx[k7] = float(np.mean(ring))
                except Exception:
                    pass


def _subset_daily_frame(ds: DailyShareDataset, loader: DataLoader) -> pd.DataFrame:
    if loader is None or not _loader_has_data(loader):
        return ds.daily.iloc[0:0].copy()
    subset = loader.dataset
    if isinstance(subset, Subset):
        return ds.daily.iloc[list(subset.indices)].copy()
    return ds.daily.copy()


@torch.no_grad()
def evaluate_rollout_fidelity(model: PreferenceNet, ds: DailyShareDataset, loader: DataLoader, device: str,
                              mode_temperature: float = 1.0, cat_temperature: float = 1.0):
    device = _normalize_device(device)
    model.eval()
    eval_df = _subset_daily_frame(ds, loader)
    if eval_df.empty:
        return {
            "rollout_mae_travel_daily": float("nan"),
            "rollout_mae_home_daily": float("nan"),
            "rollout_mae_cat_daily": float("nan"),
            "daily_share_rollout_score": float("inf"),
        }

    schema = ds.schema()
    home_cat_idx = int(getattr(getattr(ds, "meta", {}), "get", lambda *_: -1)("home_cat_idx", -1)) if hasattr(ds, "meta") else -1
    home_col = f"cat_{home_cat_idx}"
    home_pos = schema.cat_cols.index(home_col) if home_col in schema.cat_cols else None

    eval_df = eval_df.copy()
    eval_df["date"] = pd.to_datetime(eval_df["date"]).dt.date
    eval_df = eval_df.sort_values(["agent_id", "date"]).reset_index(drop=True)

    rows = []
    for agent_id, g in eval_df.groupby("agent_id", sort=False):
        g = g.sort_values("date").copy()
        roller = RolloutLagRoller(schema.ctx_cols, home_pos)
        roller.seed_from_row(g.iloc[0])

        for i in range(len(g)):
            row = g.iloc[i]
            x_np = row[schema.ctx_cols].to_numpy(dtype=np.float32)
            a = torch.tensor([ds.agent2idx.get(str(agent_id), 0)], dtype=torch.long, device=device)
            p = torch.tensor([int(row["epi_phase"])], dtype=torch.long, device=device)
            x = torch.tensor(x_np[None, :], dtype=torch.float32, device=device)
            pc, pm, t = model(a, p, x, mode_temperature=mode_temperature, cat_temperature=cat_temperature)
            pc_row = pc.squeeze(0).cpu().numpy()
            t_val = float(t.squeeze(0).item())

            rec = {
                "date": row["date"],
                "travel_true": float(row.get("travel_frac", 0.0)),
                "travel_pred": t_val,
            }
            for j, col in enumerate(schema.cat_cols):
                rec[f"true_{col}"] = float(row.get(col, 0.0))
                rec[f"pred_{col}"] = float(pc_row[j])
            if home_pos is not None:
                rec["home_true"] = float(row.get(home_col, 0.0))
                rec["home_pred"] = float(pc_row[home_pos])
            rows.append(rec)

            if i + 1 < len(g):
                next_idx = g.index[i + 1]
                next_ctx = g.loc[next_idx, schema.ctx_cols].copy()
                roller.update_next_row(next_ctx, pc_row=pc_row, t_val=t_val)
                for k, v in next_ctx.items():
                    g.at[next_idx, k] = np.float32(v)

    pred_df = pd.DataFrame(rows)
    by_date = pred_df.groupby("date", as_index=False).mean(numeric_only=True)
    travel_mae = float(np.mean(np.abs(by_date["travel_pred"] - by_date["travel_true"])))

    cat_maes = []
    for col in schema.cat_cols:
        cat_maes.append(np.abs(by_date[f"pred_{col}"] - by_date[f"true_{col}"]).mean())
    cat_mae = float(np.mean(cat_maes)) if cat_maes else float("nan")

    if home_pos is not None and "home_true" in by_date.columns and "home_pred" in by_date.columns:
        home_mae = float(np.mean(np.abs(by_date["home_pred"] - by_date["home_true"])))
        score = travel_mae + home_mae + 0.5 * cat_mae
    else:
        home_mae = float("nan")
        score = travel_mae + 0.5 * cat_mae

    return {
        "rollout_mae_travel_daily": travel_mae,
        "rollout_mae_home_daily": home_mae,
        "rollout_mae_cat_daily": cat_mae,
        "daily_share_rollout_score": float(score),
    }


def evaluate(model:PreferenceNet, loader:DataLoader, device:str,
             cat_cols:Optional[List[str]],
             mode_known_idx:List[int],
             mode_temperature: float = 1.0,
             cat_temperature:  float = 1.0):
    device = _normalize_device(device)
    model.eval()
    kl_cat_sum = 0.0
    kl_mode_sum = 0.0
    mse_trav_sum = 0.0
    w_day_sum = 0.0
    w_mode_sum = 0.0
    kl_mode_all_sum = 0.0
    mse_unk_h_sum = 0.0 
    n_mode_out = model.head_mode.out_features
    known_set = set(mode_known_idx)
    unk_idx = list(set(range(n_mode_out)) - known_set)
    unk_idx = unk_idx[0] if len(unk_idx) == 1 else None
    if not _loader_has_data(loader):
        return {
            "wKL_cat": float("inf"),
            "wKL_mode": float("inf"),
            "wKL_mode_all": float("inf"),
            "wMSE_travel": float("inf"),
            "wMSE_unknown_h": float("inf") if unk_idx is not None else None,
            "R2_cat_hours": float("nan"),
            "R2_mode_hours": float("nan"),
            "n_batches": 0,
        }
    all_cat_h_true = []; all_cat_h_pred = []
    all_mode_h_true = []; all_mode_h_pred = []

    with torch.no_grad():
        for batch in loader:
            a,p,x = batch["agent_idx"].to(device), batch["phase_id"].to(device), batch["x_ctx"].to(device)
            y_cat, y_mode = batch["y_cat"].to(device), batch["y_mode"].to(device)
            trav = batch["travel_frac"].to(device)  
            dh = batch["day_hours"].to(device)    
            mask = batch["mode_mask"].to(device)  

            pc, pm, t = model(
                a, p, x,
                mode_temperature=mode_temperature,
                cat_temperature=cat_temperature
            )

            # ---- weights ----
            w_day  = (dh.squeeze(-1) / 24.0)                
            w_mode = (trav.squeeze(-1) * w_day)             


            KLc = kl_divergence(y_cat, pc)             
            kl_cat_sum += torch.sum(KLc * w_day).item()


            y_mode_k = _renorm_known(y_mode, mode_known_idx)     
            pm_k = _renorm_known(pm, mode_known_idx) 
            
            known_mass = torch.sum(y_mode[..., mode_known_idx], dim=-1)
            mask_known = (known_mass > 1e-8).float()

            KLm_known = kl_divergence(y_mode_k, pm_k) * mask.squeeze(-1) * mask_known
            kl_mode_sum += torch.sum(KLm_known * w_mode).item()
            KLm_all = kl_divergence(y_mode, pm) * mask.squeeze(-1)
            kl_mode_all_sum += torch.sum(KLm_all * w_mode).item()
            
            #  travel MSE 
            mse_trav_sum += torch.sum(torch.mean((t - trav)**2, dim=-1) * w_day).item()

            #  book keeping 
            w_day_sum += torch.sum(w_day).item()
            w_mode_sum += torch.sum(w_mode).item()

            # R² (hours): keep using ALL modes for hours accounting
            stay_h_true = (1.0 - trav) * dh
            stay_h_pred = (1.0 - t) * dh
            cat_h_true = y_cat * stay_h_true
            cat_h_pred = pc * stay_h_pred
            all_cat_h_true.append(cat_h_true.cpu())
            all_cat_h_pred.append(cat_h_pred.cpu())

            mode_h_true = y_mode * (trav * dh)
            mode_h_pred = pm * (t * dh)
            all_mode_h_true.append(mode_h_true.cpu())
            all_mode_h_pred.append(mode_h_pred.cpu())
 
            if unk_idx is not None:
                true_unk_h = y_mode[..., unk_idx] * (trav * dh).squeeze(-1)
                pred_unk_h = pm[..., unk_idx] * (t * dh).squeeze(-1)
                mse_unk = (pred_unk_h - true_unk_h)**2
                mse_unk_h_sum += torch.sum(mse_unk * w_day).item()           

    if w_day_sum < 1e-9: w_day_sum  = 1.0
    if w_mode_sum < 1e-9: w_mode_sum = 1.0

    cat_true = torch.cat(all_cat_h_true, dim=0)
    cat_pred = torch.cat(all_cat_h_pred, dim=0)
    mode_true = torch.cat(all_mode_h_true, dim=0)
    mode_pred = torch.cat(all_mode_h_pred, dim=0)
    r2_cat = r2_score(cat_true,  cat_pred)
    r2_mode = r2_score(mode_true, mode_pred)

    return {
        "wKL_cat": kl_cat_sum / w_day_sum,
        "wKL_mode": kl_mode_sum / w_mode_sum,           
        "wKL_mode_all": kl_mode_all_sum / w_mode_sum,
        "wMSE_travel": mse_trav_sum / w_day_sum,
        "wMSE_unknown_h": (mse_unk_h_sum / w_day_sum) if unk_idx is not None else None,  # NEW
        "R2_cat_hours": r2_cat,
        "R2_mode_hours": r2_mode,
        "n_batches": len(loader)
    }

@torch.no_grad()
def find_best_mode_temperature(model, loader, device, mode_known_idx):
    grid = np.linspace(0.85, 1.15, 13)  
    best_tau, best = 1.0, (float("inf"), -float("inf"))
    for tau in grid:
        m = evaluate(model, loader, device, None, mode_known_idx,
                     mode_temperature=float(tau), cat_temperature=1.0)
        crit = (m["wKL_mode"] + 0.25*m["wKL_mode_all"], -m["R2_mode_hours"])
        if crit < best: 
            best, best_tau = crit, float(tau)
    return best_tau

@torch.no_grad()
def find_best_cat_temperature(model, loader, device, mode_known_idx):
    grid = np.linspace(0.85, 1.15, 13)
    best_tau, best = 1.0, (float("inf"), -float("inf"))
    for tau in grid:
        m = evaluate(model, loader, device, None, mode_known_idx,
                     mode_temperature=1.0, cat_temperature=float(tau))
        crit = (m["wKL_cat"], -m["R2_cat_hours"])
        if crit < best:
            best, best_tau = crit, float(tau)
    return best_tau


def build_checkpoint_payload(model, cfg, ds, schema):
    return {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "n_agents": len(ds.agents),
        "n_phases": int(ds.daily["epi_phase"].max()) + 1,
        "ctx_dim": len(schema.ctx_cols),
        "n_cat": ds.n_cat,
        "n_mode": ds.n_mode,
        "schema": asdict(schema),
    }


def evaluate_checkpoint(label: str, ckpt_path: Path, model: PreferenceNet, val_loader: DataLoader,
                        test_loader: DataLoader, device: str, mode_known_idx: List[int],
                        cat_cols: List[str]):
    device = _normalize_device(device)
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    if _loader_has_data(val_loader):
        best_tau_mode = find_best_mode_temperature(model, val_loader, device, mode_known_idx)
        best_tau_cat = find_best_cat_temperature(model, val_loader, device, mode_known_idx)
    else:
        best_tau_mode = 1.0
        best_tau_cat = 1.0
    test_metrics = evaluate(model, test_loader, device, cat_cols, mode_known_idx,
                            mode_temperature=best_tau_mode, cat_temperature=best_tau_cat)
    return {
        "checkpoint": label,
        "path": str(ckpt_path),
        "mode_temperature": best_tau_mode,
        "cat_temperature": best_tau_cat,
        "test_metrics": test_metrics,
    }


def train(cfg:TrainConfig, daily_shares:Path, feature_meta:Optional[Path], out_dir:Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    ds = DailyShareDataset(daily_shares, feature_meta)
    ds.agent_index_df().to_csv(out_dir/"agent_index.csv", index=False)
    schema = ds.schema()
    json.dump({
        "cat_cols": schema.cat_cols,
        "mode_cols_all": schema.mode_cols_all,
        "mode_cols_known": schema.mode_cols_known,
        "ctx_cols": schema.ctx_cols
    }, open(out_dir/"columns.json","w"), indent=2)

    torch.save(ds.daily, out_dir/"daily_shares_cache.pt")  

    device = _normalize_device(cfg.device) if torch.cuda.is_available() else "cpu"
    if not device.startswith("cuda"):
        device = "cpu"

    # Precompute per-phase first-week lag feature means for training augmentation
    lag_cols = [c for c in schema.ctx_cols if c.startswith("prev_") or c.startswith("prev7_")]
    lag_col_indices = [schema.ctx_cols.index(c) for c in lag_cols]
    phase_lag_means_train: dict = {}
    if lag_col_indices:
        for ph, g_ph in ds.daily.groupby("epi_phase"):
            g_sorted = g_ph.sort_values("date")
            first_dates = g_sorted["date"].unique()[:7]
            g_first = g_sorted[g_sorted["date"].isin(first_dates)]
            phase_lag_means_train[int(ph)] = g_first[lag_cols].mean().to_numpy(dtype=np.float32)

    loaders = make_loaders(ds, cfg, out_dir)
    train_loader, val_loader, test_loader = loaders
    has_val = _loader_has_data(val_loader)
    mode_known_idx = [ds.schema().mode_cols_all.index(c) for c in ds.schema().mode_cols_known]

    mode_known_idx = [ds.schema().mode_cols_all.index(c) for c in ds.schema().mode_cols_known]
    unk_idx = list(set(range(len(ds.schema().mode_cols_all))) - set(mode_known_idx))
    assert len(unk_idx) == 1, "Expected exactly one unknown mode."
    unk_idx = unk_idx[0]

    model = PreferenceNet(
        n_agents=len(ds.agents),
        n_phases=int(ds.daily["epi_phase"].max())+1,
        ctx_dim=len(schema.ctx_cols),
        n_cat=ds.n_cat,
        n_mode=ds.n_mode,
        emb_dim=cfg.emb_dim, h_dim=cfg.h_dim, dropout=cfg.dropout,
        mode_known_idx=mode_known_idx, unk_mode_idx=unk_idx,
        moe_num_experts=cfg.moe_num_experts, moe_top_k=cfg.moe_top_k,
        moe_gate_temperature=cfg.moe_gate_temperature,
        moe_gate_noise_std=cfg.moe_gate_noise_std,
        moe_gate_dropout=cfg.moe_gate_dropout,
        moe_cat_top_k=cfg.moe_cat_top_k,
        moe_mode_top_k=cfg.moe_mode_top_k,
        moe_cat_gate_temperature=cfg.moe_cat_gate_temperature,
        moe_mode_gate_temperature=cfg.moe_mode_gate_temperature,
        moe_cat_gate_noise_std=cfg.moe_cat_gate_noise_std,
        moe_mode_gate_noise_std=cfg.moe_mode_gate_noise_std,
        moe_cat_gate_dropout=cfg.moe_cat_gate_dropout,
        moe_mode_gate_dropout=cfg.moe_mode_gate_dropout,
        use_film=cfg.use_film,
        use_moe_heads=cfg.use_moe_heads,
        use_phase_lag_gate=cfg.use_phase_lag_gate,
        detach_travel_head_context=cfg.detach_travel_head_context,
    ).to(device)
    model.moe_cat_aux_weight = float(cfg.moe_cat_aux_weight if cfg.moe_cat_aux_weight is not None else cfg.moe_aux_weight)
    model.moe_mode_aux_weight = float(cfg.moe_mode_aux_weight if cfg.moe_mode_aux_weight is not None else cfg.moe_aux_weight)
    # Enable phase-gated lag suppression
    if lag_col_indices and cfg.use_phase_lag_gate:
        model.set_lag_feature_indices(lag_col_indices)
        model.phase_lag_gate = model.phase_lag_gate.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # warmup + cosine
    warm = max(0, int(cfg.warmup_epochs))
    cosine_epochs = max(1, cfg.epochs - warm)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warm) if warm > 0 else
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=1.0, total_iters=1),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_epochs, eta_min=cfg.lr * 0.1)
        ],
        milestones=[warm] if warm > 0 else [1],
    )
    best_scores = {
        "kl": float("inf"),
        "mode_r2": -float("inf"),
        "balanced": float("inf"),
        "sim": float("inf"),
    }
    best_paths = {
        "kl": out_dir/"best_kl_model.pt",
        "mode_r2": out_dir/"best_mode_r2_model.pt",
        "balanced": out_dir/"best_balanced_model.pt",
        "sim": out_dir/"best_sim_model.pt",
    }
    best_paths["default"] = out_dir/"model.pt"
    default_label = cfg.default_checkpoint_metric if cfg.default_checkpoint_metric in best_scores else "balanced"
    metrics_rows = []
    for epoch in range(1, cfg.epochs+1):
        model.train()
        loss_sum = 0.0
        w_sum = 0.0
        for batch in train_loader:
            a,p,x = batch["agent_idx"].to(device), batch["phase_id"].to(device), batch["x_ctx"]
            # Lag feature augmentation: replace lag features with prev-phase first-week means
            # at 80% probability for phase-transition samples, teaching phase responsiveness
            if lag_col_indices and phase_lag_means_train:
                has_prev = batch.get("has_prev_phase", torch.zeros(len(a))).numpy()
                prev_phases = batch.get("prev_phase_id", p.cpu()).numpy()
                x_np = x.numpy().copy()
                mask = (has_prev > 0.5) & (np.random.rand(len(a)) < 0.8)
                for bi in np.where(mask)[0]:
                    prev_ph = int(prev_phases[bi])
                    if prev_ph in phase_lag_means_train:
                        x_np[bi, lag_col_indices] = phase_lag_means_train[prev_ph]
                x = torch.tensor(x_np, dtype=torch.float32)
            x = x.to(device)
            y_cat, y_mode = batch["y_cat"].to(device), batch["y_mode"].to(device)
            trav = batch["travel_frac"].to(device)  
            dh = batch["day_hours"].to(device)     
            mask = batch["mode_mask"].to(device)   

            # temperature=1 during training
            pc, pm, t, aux_moe = model(a,p,x, return_aux=True)  

            w_day  = (dh / 24.0).squeeze(-1)         
            w_mode = (trav.squeeze(-1) * w_day) 

            # cat loss
            L_cat = safe_weighted_mean(kl_divergence(y_cat, pc), w_day)

            # mode KL on known subset 
            y_mode_k = _renorm_known(y_mode, mode_known_idx)       
            pm_k     = _renorm_known(pm,     mode_known_idx)       
            
            known_mass = torch.sum(y_mode[..., mode_known_idx], dim=-1)
            mask_known = (known_mass > 1e-8).float()

            KLm_known = kl_divergence(y_mode_k, pm_k) * mask.squeeze(-1) * mask_known
            L_modeKL_known = safe_weighted_mean(KLm_known, w_mode)

            KLm_all = kl_divergence(y_mode, pm) * mask.squeeze(-1)
            L_modeKL_all = safe_weighted_mean(KLm_all, w_mode)

            # travel loss
            L_trav = torch.mean((t - trav)**2 * (dh/24.0))

            y_travel_bin = (trav > 1e-6).float()    
            eps = 1e-6
            bce_vec = -(y_travel_bin * torch.log(t.clamp_min(eps)) +
                        (1.0 - y_travel_bin) * torch.log((1.0 - t).clamp_min(eps)))
            L_trav_bce = safe_weighted_mean(bce_vec.squeeze(-1), w_day)              

            # hours MSE on all modes 
            true_mode_h_all = y_mode * (trav * dh)
            pred_mode_h_all = pm  * (t* dh)
            L_modeH_all = safe_weighted_mean(((pred_mode_h_all - true_mode_h_all)**2).mean(dim=-1), w_day)
            unk = model.unk_mode_idx
            true_unk_h = (y_mode[..., unk] * (trav * dh).squeeze(-1))
            pred_unk_h = (pm[..., unk] * (t * dh).squeeze(-1))
            L_unkH = safe_weighted_mean((pred_unk_h - true_unk_h)**2, w_day)

            # observed hours per sample
            stay_h_true = (1.0 - trav) * dh       
            mode_h_true = y_mode * (trav * dh)     
            # predicted hours
            stay_h_pred = (1.0 - t) * dh        
            mode_h_pred = pm * (t * dh)           
            cat_h_pred  = pc * stay_h_pred         

            # zero-aware L1 on hours
            zero_cat_mask  = ((y_cat * stay_h_true) <= 1e-9).float()
            zero_mode_mask = ((y_mode * (trav * dh)) <= 1e-9).float()

            L_zero_cat  = safe_weighted_mean((cat_h_pred  * zero_cat_mask ).sum(dim=-1), w_day)
            L_zero_mode = safe_weighted_mean((mode_h_pred * zero_mode_mask).sum(dim=-1), w_day)


            # anneal weights
            ramp = min(1.0, epoch / max(1, cfg.epochs//3))
            mode_w = 0.25 + 0.5 * ramp    
            true_cat_h = y_cat * ((1.0 - trav) * dh)
            pred_cat_h = pc * ((1.0 - t) * dh)
            L_catH = safe_weighted_mean(((pred_cat_h - true_cat_h)**2).mean(dim=-1), w_day)           


            H_cat  = -torch.sum(pc.clamp_min(1e-8) * torch.log(pc.clamp_min(1e-8)), dim=-1)
            H_mode = -torch.sum(pm.clamp_min(1e-8) * torch.log(pm.clamp_min(1e-8)), dim=-1)
            L_entropy = safe_weighted_mean(H_cat, w_day) + safe_weighted_mean(H_mode, w_mode)

            # Phase-sensitivity loss: model should produce meaningfully different
            # category distributions when phase changes, given the same context.
            # We shuffle phases within the batch, compute predictions, then
            # penalise if KL(pc || pc_other_phase) < margin for cross-phase pairs.
            L_phase_sens = torch.zeros((), device=device)
            if cfg.phase_sens_weight > 0.0 and p.unique().numel() > 1:
                perm_idx = torch.randperm(p.size(0), device=device)
                p_other = p[perm_idx]
                diff_phase = (p != p_other).float()          # 1 where phases differ
                if diff_phase.sum() > 0:
                    with torch.no_grad():
                        pc_other, _, _ = model(a, p_other, x,
                                               mode_temperature=1.0,
                                               cat_temperature=1.0)
                    # KL(pc || pc_other): how much current pc differs from other-phase pc
                    phase_kl = kl_divergence(pc, pc_other.detach())
                    # Penalise insufficient divergence (below margin)
                    L_phase_sens = safe_weighted_mean(
                        F.relu(cfg.phase_sens_margin - phase_kl) * diff_phase,
                        w_day * diff_phase.clamp_min(1e-8)
                    )

            loss = (
                L_cat
                + 0.50 * L_trav
                + 0.05 * L_trav_bce
                + mode_w * L_modeKL_known
                + 0.20 * L_modeKL_all
                + 0.40 * L_modeH_all
                + 0.25 * L_catH
                + 0.01 * L_zero_cat
                + 0.05 * L_zero_mode
                + 0.00 * L_entropy
                + 0.15 * L_unkH
                + cfg.moe_aux_weight * aux_moe
                + cfg.phase_sens_weight * L_phase_sens
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            loss_sum += loss.item() * float(w_day.shape[0])
            w_sum += float(w_day.shape[0])
        sched.step()

        train_metrics = evaluate(model, train_loader, device, ds.schema().cat_cols, mode_known_idx,
                                mode_temperature=1.0, cat_temperature=1.0)
        selection_metrics = (
            evaluate(model, val_loader, device, ds.schema().cat_cols, mode_known_idx,
                     mode_temperature=1.0, cat_temperature=1.0)
            if has_val else train_metrics
        )
        do_rollout_eval = (epoch == 1) or (epoch == cfg.epochs) or (epoch % max(1, cfg.rollout_eval_every) == 0)
        rollout_metrics = evaluate_rollout_fidelity(model, ds, val_loader, device) if (do_rollout_eval and has_val) else {}
        metric_prefix = "val" if has_val else "sel"

        row = {"epoch":epoch, "train_loss": loss_sum/max(w_sum,1.0)}
        row.update({f"train_{k}":v for k,v in train_metrics.items()})
        row.update({f"{metric_prefix}_{k}":v for k,v in selection_metrics.items()})
        row.update({f"{metric_prefix}_{k}": v for k, v in rollout_metrics.items()})
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(out_dir/"metrics.csv", index=False)
        rollout_key = f"{metric_prefix}_daily_share_rollout_score"
        rollout_msg = f"  {metric_prefix}_rollout={row[rollout_key]:.4f}" if rollout_key in row else ""
        print(f"[epoch {epoch}] train_loss={row['train_loss']:.4f}  "
              f"{metric_prefix}_wKL_cat={row[f'{metric_prefix}_wKL_cat']:.4f}  "
              f"{metric_prefix}_R2_cat={row[f'{metric_prefix}_R2_cat_hours']:.3f}  "
              f"{metric_prefix}_R2_mode={row[f'{metric_prefix}_R2_mode_hours']:.3f}{rollout_msg}")

        kl_crit = selection_metrics["wKL_cat"] + selection_metrics["wKL_mode"]
        mode_r2_crit = selection_metrics["R2_mode_hours"]
        balanced_crit = (
            selection_metrics["wKL_cat"]
            + selection_metrics["wKL_mode"]
            - 0.30 * selection_metrics["R2_cat_hours"]
            - 0.50 * selection_metrics["R2_mode_hours"]
        )
        ckpt_payload = build_checkpoint_payload(model, cfg, ds, schema)
        sim_crit = balanced_crit
        if rollout_metrics:
            sim_crit = cfg.rollout_weight * rollout_metrics["daily_share_rollout_score"] + 0.25 * (selection_metrics["wKL_cat"] + selection_metrics["wKL_mode"])

        if kl_crit < best_scores["kl"]:
            best_scores["kl"] = kl_crit
            torch.save(ckpt_payload, best_paths["kl"])
            if default_label == "kl":
                torch.save(ckpt_payload, best_paths["default"])

        if mode_r2_crit > best_scores["mode_r2"]:
            best_scores["mode_r2"] = mode_r2_crit
            torch.save(ckpt_payload, best_paths["mode_r2"])
            if default_label == "mode_r2":
                torch.save(ckpt_payload, best_paths["default"])

        if balanced_crit < best_scores["balanced"]:
            best_scores["balanced"] = balanced_crit
            torch.save(ckpt_payload, best_paths["balanced"])
            if default_label == "balanced":
                torch.save(ckpt_payload, best_paths["default"])

        if sim_crit < best_scores["sim"]:
            best_scores["sim"] = sim_crit
            torch.save(ckpt_payload, best_paths["sim"])
            if default_label == "sim":
                torch.save(ckpt_payload, best_paths["default"])

    checkpoint_results = {}
    for label in ["kl", "mode_r2", "balanced", "sim"]:
        result = evaluate_checkpoint(
            label,
            best_paths[label],
            model,
            (val_loader if has_val else None),
            test_loader,
            device,
            mode_known_idx,
            ds.schema().cat_cols,
        )
        checkpoint_results[label] = result
        with open(out_dir/f"test_metrics_{label}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"[TEST:{label}] (temp-calibrated)", result["test_metrics"])

        with open(out_dir/"best_model_summary.json", "w") as f:
            json.dump({
                "best_scores": best_scores,
                "default_checkpoint": default_label,
                "results": checkpoint_results,
            }, f, indent=2)

    default_result = checkpoint_results[default_label]
    with open(out_dir/"calibration.json", "w") as f:
        json.dump({
            "mode_temperature": default_result["mode_temperature"],
            "cat_temperature": default_result["cat_temperature"],
            "checkpoint": default_label,
        }, f, indent=2)

    with open(out_dir/"test_metrics.json", "w") as f:
        json.dump(default_result["test_metrics"], f, indent=2)

    best = torch.load(best_paths[default_label], map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    emit_predictions(
        model,
        ds,
        device,
        out_dir,
        mode_temperature=default_result["mode_temperature"],
        cat_temperature=default_result["cat_temperature"],
    )
    emit_expert_analysis(
        model,
        ds,
        device,
        out_dir,
        mode_temperature=default_result["mode_temperature"],
        cat_temperature=default_result["cat_temperature"],
        checkpoint_label=default_label,
    )

def _entropy_from_probs(probs: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(probs, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def emit_expert_analysis(model:PreferenceNet, ds:DailyShareDataset, device:str, out_dir:Path,
                         mode_temperature: float = 1.0,
                         cat_temperature: float = 1.0,
                         checkpoint_label: str = "balanced"):
    device = _normalize_device(device)
    model.eval()
    schema = ds.schema()
    analysis_dir = out_dir / "expert_analysis"
    analysis_dir.mkdir(exist_ok=True, parents=True)
    if not getattr(model, "use_moe_heads", True):
        (analysis_dir / f"expert_analysis_{checkpoint_label}.txt").write_text(
            "Expert routing analysis is unavailable because MoE heads were disabled for this run.\n",
            encoding="utf-8",
        )
        return

    routing_rows = []
    context_cols = [
        "days_since_phase", "days_since_npi", "day_hours", "n_trips",
        "total_trip_km", "mean_trip_km", "max_trip_km",
        "prev_travel_frac", "prev7_travel_frac_mean",
        "prev_home_share", "prev7_home_share_mean",
        "gender", "age_0", "age_1", "age_2", "age_3", "age_4",
    ]
    context_cols = [c for c in context_cols if c in ds.daily.columns]

    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            row_raw = ds.daily.loc[ds.index[i]]
            a = sample["agent_idx"].to(device)
            p = sample["phase_id"].to(device)
            x = sample["x_ctx"].to(device)
            pc, pm, t, routing = model(
                a.unsqueeze(0), p.unsqueeze(0), x.unsqueeze(0),
                mode_temperature=mode_temperature,
                cat_temperature=cat_temperature,
                return_routing=True,
            )

            mode_probs = routing["mode"]["gate_probs"].squeeze(0).cpu().numpy()
            cat_probs = routing["cat"]["gate_probs"].squeeze(0).cpu().numpy()
            mode_topk_idx = routing["mode"]["topk_idx"].squeeze(0).cpu().tolist()
            cat_topk_idx = routing["cat"]["topk_idx"].squeeze(0).cpu().tolist()
            mode_topk_vals = routing["mode"]["topk_vals"].squeeze(0).cpu().tolist()
            cat_topk_vals = routing["cat"]["topk_vals"].squeeze(0).cpu().tolist()
            pm_np = pm.squeeze(0).cpu().numpy()
            pc_np = pc.squeeze(0).cpu().numpy()
            t_val = float(t.squeeze(0).item())
            meta = sample["meta"]
            record = {
                "checkpoint": checkpoint_label,
                "agent_id": meta["agent_id"],
                "date": meta["date"],
                "epi_phase": meta["epi_phase"],
                "mode_expert_top1": int(np.argmax(mode_probs)),
                "mode_expert_top1_prob": float(np.max(mode_probs)),
                "mode_expert_entropy": _entropy_from_probs(mode_probs),
                "mode_expert_topk": "|".join(map(str, mode_topk_idx)),
                "mode_expert_topk_prob": "|".join(f"{v:.6f}" for v in mode_topk_vals),
                "cat_expert_top1": int(np.argmax(cat_probs)),
                "cat_expert_top1_prob": float(np.max(cat_probs)),
                "cat_expert_entropy": _entropy_from_probs(cat_probs),
                "cat_expert_topk": "|".join(map(str, cat_topk_idx)),
                "cat_expert_topk_prob": "|".join(f"{v:.6f}" for v in cat_topk_vals),
                "travel_frac_true": float(sample["travel_frac"].item()),
                "travel_frac_pred": t_val,
                "day_hours": float(sample["day_hours"].item()),
                "mode_unknown_pred_share": float(pm_np[model.unk_mode_idx]),
            }
            for col in context_cols:
                record[col] = float(row_raw[col])
            for j, col in enumerate(schema.mode_cols_all):
                record[f"true_{col}"] = float(sample["y_mode"][j].item())
                record[f"pred_{col}"] = float(pm_np[j])
            for j, col in enumerate(schema.cat_cols):
                record[f"true_{col}"] = float(sample["y_cat"][j].item())
                record[f"pred_{col}"] = float(pc_np[j])
            routing_rows.append(record)

    routing_df = pd.DataFrame(routing_rows)
    routing_path = analysis_dir / f"per_sample_routing_{checkpoint_label}.csv"
    routing_df.to_csv(routing_path, index=False, float_format="%.6f")

    def summarize_by_expert(head_name: str, n_experts: int):
        expert_col = f"{head_name}_expert_top1"
        prob_col = f"{head_name}_expert_top1_prob"
        entropy_col = f"{head_name}_expert_entropy"
        summary_rows = []
        for expert_id in range(n_experts):
            sub = routing_df[routing_df[expert_col] == expert_id]
            row = {
                "head": head_name,
                "expert_id": expert_id,
                "n_samples": int(len(sub)),
                "sample_share": float(len(sub) / max(len(routing_df), 1)),
            }
            if len(sub) == 0:
                summary_rows.append(row)
                continue
            row.update({
                "avg_top1_prob": float(sub[prob_col].mean()),
                "avg_entropy": float(sub[entropy_col].mean()),
                "avg_travel_frac_true": float(sub["travel_frac_true"].mean()),
                "avg_travel_frac_pred": float(sub["travel_frac_pred"].mean()),
                "avg_day_hours": float(sub["day_hours"].mean()),
                "avg_mode_unknown_pred_share": float(sub["mode_unknown_pred_share"].mean()),
                "dominant_epi_phase": int(sub["epi_phase"].mode().iloc[0]),
            })
            for col in context_cols:
                row[f"avg_{col}"] = float(sub[col].mean())
            for col in schema.mode_cols_all:
                row[f"avg_true_{col}"] = float(sub[f"true_{col}"].mean())
                row[f"avg_pred_{col}"] = float(sub[f"pred_{col}"].mean())
            for col in schema.cat_cols:
                row[f"avg_true_{col}"] = float(sub[f"true_{col}"].mean())
                row[f"avg_pred_{col}"] = float(sub[f"pred_{col}"].mean())
            summary_rows.append(row)
        return pd.DataFrame(summary_rows)

    mode_summary = summarize_by_expert("mode", model.head_mode.num_experts)
    cat_summary = summarize_by_expert("cat", model.head_cat.num_experts)
    mode_summary.to_csv(analysis_dir / f"expert_profile_mode_{checkpoint_label}.csv", index=False, float_format="%.6f")
    cat_summary.to_csv(analysis_dir / f"expert_profile_cat_{checkpoint_label}.csv", index=False, float_format="%.6f")

    phase_rows = []
    for head_name in ["mode", "cat"]:
        expert_col = f"{head_name}_expert_top1"
        phase_share = (routing_df.groupby([expert_col, "epi_phase"]).size() / routing_df.groupby(expert_col).size()).reset_index(name="phase_share")
        phase_share.insert(0, "head", head_name)
        phase_share = phase_share.rename(columns={expert_col: "expert_id"})
        phase_rows.append(phase_share)
    pd.concat(phase_rows, ignore_index=True).to_csv(
        analysis_dir / f"expert_phase_distribution_{checkpoint_label}.csv",
        index=False,
        float_format="%.6f",
    )


def emit_predictions(model:PreferenceNet, ds:DailyShareDataset, device:str, out_dir:Path,
                     mode_temperature: float = 1.0,
                     cat_temperature: float = 1.0):
    device = _normalize_device(device)
    model.eval()
    schema = ds.schema()
    daily = ds.daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    poi_rows = []
    mode_rows = []

    # precompute mapping and the single unknown index+column
    mode_known_indices = [schema.mode_cols_all.index(c) for c in schema.mode_cols_known]
    all_mode_count = len(schema.mode_cols_all)
    unknown_all_idx = list(set(range(all_mode_count)) - set(mode_known_indices))
    assert len(unknown_all_idx) == 1, "Expected exactly one unknown mode."
    unk_all_idx = int(unknown_all_idx[0])
    unknown_col_name = schema.mode_cols_all[unk_all_idx] + "h"

    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            a = sample["agent_idx"].to(device)
            p = sample["phase_id"].to(device)
            x = sample["x_ctx"].to(device)
            pc, pm, t = model(
                a.unsqueeze(0), p.unsqueeze(0), x.unsqueeze(0),
                mode_temperature=mode_temperature,
                cat_temperature=cat_temperature
            )
            pc = pc.squeeze(0).cpu().numpy()
            pm = pm.squeeze(0).cpu().numpy()
            t = float(t.squeeze(0).item())

            dh = float(sample["day_hours"].item())
            meta = sample["meta"]

            # POI hours
            stay_h = (1.0 - t) * dh
            row_poi = {
                "agent_id": meta["agent_id"],
                "date": meta["date"],
                "epi_phase": meta["epi_phase"],
                "day_hours": dh,
                "total_stay_h": float(stay_h),
            }
            for j, col in enumerate(schema.cat_cols):
                row_poi[f"{col}h"] = float(pc[j] * stay_h)
            poi_rows.append(row_poi)

            # Mode hours
            unknown_h = float(pm[unk_all_idx] * (t * dh))
            row_mode = {
                "agent_id": meta["agent_id"],
                "date": meta["date"],
                "epi_phase": meta["epi_phase"],
                "day_hours": dh,
                "total_travel_h": float((pm * (t * dh)).sum()),
                unknown_col_name: unknown_h,  
            }
            for j, col in enumerate(schema.mode_cols_known):
                j_all = mode_known_indices[j]
                row_mode[f"{col}h"] = float(pm[j_all] * (t * dh))
            mode_rows.append(row_mode)

    poi_df  = pd.DataFrame(poi_rows)
    mode_df = pd.DataFrame(mode_rows)

    if unknown_col_name not in mode_df.columns:
        mode_df[unknown_col_name] = 0.0

    poi_cols  = ["agent_id","date","epi_phase","day_hours","total_stay_h"] \
                + [f"{c}h" for c in schema.cat_cols]
    mode_cols = ["agent_id","date","epi_phase","day_hours","total_travel_h", unknown_col_name] \
                + [f"{c}h" for c in schema.mode_cols_known]

    poi_df  = poi_df[poi_cols].sort_values(["agent_id","date"])
    mode_df = mode_df[mode_cols].sort_values(["agent_id","date"])


    (out_dir/"pred").mkdir(exist_ok=True, parents=True)
    poi_path = out_dir/"pred"/"per_agent_daily_poi_hours_pred.csv"
    mode_path = out_dir/"pred"/"per_agent_daily_mode_hours_pred.csv"

    poi_df.to_csv(poi_path, index=False, float_format="%.6f")
    mode_df.to_csv(mode_path, index=False, float_format="%.6f")

    # Per-agent phase means
    agg_poi = poi_df.groupby(["agent_id","epi_phase"], dropna=False)[[c for c in poi_cols if c.endswith("h")]].mean().reset_index()
    agg_mode = mode_df.groupby(["agent_id","epi_phase"], dropna=False)[[c for c in mode_cols if c.endswith("h")]].mean().reset_index()
    agent_phase = pd.merge(agg_poi, agg_mode, on=["agent_id","epi_phase"], how="outer").fillna(0.0)
    phase_path = out_dir/"pred"/"per_agent_phase_summary_pred.csv"
    agent_phase.to_csv(phase_path, index=False, float_format="%.6f")

    # Write a quick alignment manifest to help debug
    json.dump({
        "poi_pred_path": str(poi_path),
        "mode_pred_path": str(mode_path),
        "phase_pred_path": str(phase_path),
        "cat_cols": schema.cat_cols,
        "mode_cols_known": schema.mode_cols_known,
        "mode_temperature": mode_temperature,
        "cat_temperature":  cat_temperature,
        "unknown_mode_col": unknown_col_name,
    }, open(out_dir/"pred"/"manifest.json","w"), indent=2)

    cat_cols = [c for c in poi_df.columns if c.startswith("cat_") and c.endswith("h")]
    poi_chk = poi_df[["agent_id","date","day_hours","total_stay_h"] + cat_cols].copy()
    poi_chk["sum_cat_h"] = poi_chk[cat_cols].sum(axis=1)

    chk = poi_chk[["agent_id","date","day_hours","total_stay_h","sum_cat_h"]].merge(
        mode_df[["agent_id","date","total_travel_h"]],
        on=["agent_id","date"], how="left"
    )
    chk["lhs"] = chk["sum_cat_h"] + chk["total_travel_h"].fillna(0.0)
    chk["residual"] = chk["day_hours"] - chk["lhs"]
    chk.to_csv(out_dir / "pred" / "sums_check.csv", index=False)


    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily_shares", required=True)
    ap.add_argument("--feature_meta", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--h_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val_days", type=int, default=14)
    ap.add_argument("--test_days", type=int, default=14)
    ap.add_argument("--split_mode", choices=[
        "phase_block_random",
        "phase_temporal",
        "phase_last_week_test",
        "global_temporal",
        "leave_one_phase_out",
        "transition_backtest",
        "unseen_agent",
    ], default="phase_last_week_test")
    ap.add_argument("--holdout_phase", type=int, default=None,
                    help="Held-out epi_phase for leave_one_phase_out evaluation. Defaults to the latest available phase.")
    ap.add_argument("--transition_backtest_idx", type=int, default=None,
                    help="0-based global phase-transition index for transition_backtest. Defaults to the latest transition.")
    ap.add_argument("--agent_val_frac", type=float, default=0.10,
                    help="Validation-agent fraction for unseen_agent split mode.")
    ap.add_argument("--agent_test_frac", type=float, default=0.10,
                    help="Test-agent fraction for unseen_agent split mode.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--moe_num_experts", type=int, default=4)
    ap.add_argument("--moe_top_k", type=int, default=2)
    ap.add_argument("--moe_aux_weight", type=float, default=5e-2)
    ap.add_argument("--moe_gate_temperature", type=float, default=1.25)
    ap.add_argument("--moe_gate_noise_std", type=float, default=0.15)
    ap.add_argument("--moe_gate_dropout", type=float, default=0.02)
    ap.add_argument("--moe_cat_top_k", type=int, default=0)
    ap.add_argument("--moe_mode_top_k", type=int, default=0)
    ap.add_argument("--moe_cat_aux_weight", type=float, default=-1.0)
    ap.add_argument("--moe_mode_aux_weight", type=float, default=-1.0)
    ap.add_argument("--moe_cat_gate_temperature", type=float, default=0.0)
    ap.add_argument("--moe_mode_gate_temperature", type=float, default=0.0)
    ap.add_argument("--moe_cat_gate_noise_std", type=float, default=-1.0)
    ap.add_argument("--moe_mode_gate_noise_std", type=float, default=-1.0)
    ap.add_argument("--moe_cat_gate_dropout", type=float, default=-1.0)
    ap.add_argument("--moe_mode_gate_dropout", type=float, default=-1.0)
    ap.add_argument("--rollout_eval_every", type=int, default=5)
    ap.add_argument("--rollout_weight", type=float, default=1.0)
    ap.add_argument("--default_checkpoint_metric", choices=["kl", "mode_r2", "balanced", "sim"], default="balanced")
    ap.add_argument("--disable_film", action="store_true",
                    help="Disable FiLM phase modulation for ablation runs.")
    ap.add_argument("--disable_moe_heads", action="store_true",
                    help="Replace MoE heads with dense heads for ablation runs.")
    ap.add_argument("--disable_phase_lag_gate", action="store_true",
                    help="Disable phase-gated lag suppression for ablation runs.")
    ap.add_argument("--disable_travel_head_stopgrad", action="store_true",
                    help="Allow travel-head gradients to flow into category/mode conditioning inputs.")
    ap.add_argument("--phase_sens_weight", type=float, default=0.10,
                    help="Weight for cross-phase sensitivity contrastive loss")
    ap.add_argument("--phase_sens_margin", type=float, default=0.05,
                    help="KL margin below which the model is penalised for insensitivity to phase change")
    args = ap.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, emb_dim=args.emb_dim, h_dim=args.h_dim,
        dropout=args.dropout, val_days=args.val_days, test_days=args.test_days,
        split_mode=args.split_mode,
        holdout_phase=args.holdout_phase,
        transition_backtest_idx=args.transition_backtest_idx,
        agent_val_frac=args.agent_val_frac,
        agent_test_frac=args.agent_test_frac,
        seed=args.seed, num_workers=args.num_workers,
        device="cuda" if torch.cuda.is_available() else "cpu",
        warmup_epochs=args.warmup_epochs,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_aux_weight=args.moe_aux_weight,
        moe_gate_temperature=args.moe_gate_temperature,
        moe_gate_noise_std=args.moe_gate_noise_std,
        moe_gate_dropout=args.moe_gate_dropout,
        moe_cat_top_k=(args.moe_cat_top_k if args.moe_cat_top_k > 0 else None),
        moe_mode_top_k=(args.moe_mode_top_k if args.moe_mode_top_k > 0 else None),
        moe_cat_aux_weight=(args.moe_cat_aux_weight if args.moe_cat_aux_weight >= 0.0 else None),
        moe_mode_aux_weight=(args.moe_mode_aux_weight if args.moe_mode_aux_weight >= 0.0 else None),
        moe_cat_gate_temperature=(args.moe_cat_gate_temperature if args.moe_cat_gate_temperature > 0.0 else None),
        moe_mode_gate_temperature=(args.moe_mode_gate_temperature if args.moe_mode_gate_temperature > 0.0 else None),
        moe_cat_gate_noise_std=(args.moe_cat_gate_noise_std if args.moe_cat_gate_noise_std >= 0.0 else None),
        moe_mode_gate_noise_std=(args.moe_mode_gate_noise_std if args.moe_mode_gate_noise_std >= 0.0 else None),
        moe_cat_gate_dropout=(args.moe_cat_gate_dropout if args.moe_cat_gate_dropout >= 0.0 else None),
        moe_mode_gate_dropout=(args.moe_mode_gate_dropout if args.moe_mode_gate_dropout >= 0.0 else None),
        rollout_eval_every=args.rollout_eval_every,
        rollout_weight=args.rollout_weight,
        default_checkpoint_metric=args.default_checkpoint_metric,
        use_film=not args.disable_film,
        use_moe_heads=not args.disable_moe_heads,
        use_phase_lag_gate=not args.disable_phase_lag_gate,
        detach_travel_head_context=not args.disable_travel_head_stopgrad,
        phase_sens_weight=args.phase_sens_weight,
        phase_sens_margin=args.phase_sens_margin,
    )

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/"train_config.json","w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    daily_shares = Path(args.daily_shares)
    feat_meta = Path(args.feature_meta) if args.feature_meta else daily_shares.parent/"feature_meta.json"

    train(cfg, daily_shares, feat_meta, out_dir)




