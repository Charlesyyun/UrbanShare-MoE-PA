from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

EPS = 1e-8


@dataclass
class PreferenceScoreCfg:
    s_dim: int
    a_dim: int
    h_dim: int = 128
    dropout: float = 0.1


class PreferenceScoreNet(nn.Module):
    def __init__(self, cfg: PreferenceScoreCfg):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.s_dim + cfg.a_dim, cfg.h_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.h_dim, cfg.h_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.h_dim, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)


def pack_actions(travel_frac: torch.Tensor, p_cat: torch.Tensor, p_mode: torch.Tensor) -> torch.Tensor:
    return torch.cat([travel_frac, p_cat, p_mode], dim=-1)


def kl_categorical(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = (p + EPS) / (p.sum(dim=-1, keepdim=True) + EPS)
    q = (q + EPS) / (q.sum(dim=-1, keepdim=True) + EPS)
    return (p * (p.add(EPS).log() - q.add(EPS).log())).sum(dim=-1)
