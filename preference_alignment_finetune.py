from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
import uuid
import importlib.util as _ilu
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from preference_scorer import PreferenceScoreCfg, PreferenceScoreNet, pack_actions, kl_categorical
from preference_dataset import DailyShareDataset


EPS = 1e-8


def _import_from_path(py_path: Path, base_name: str):
    unique_name = f"{base_name}_{uuid.uuid4().hex}"
    spec = _ilu.spec_from_file_location(unique_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for: {py_path}")
    mod = _ilu.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_batches(df: pd.DataFrame, batch_size: int, shuffle: bool = False, seed: int = 0) -> List[pd.DataFrame]:
    if df.empty:
        return []
    order = np.arange(len(df))
    if shuffle and len(order) > 1:
        rng = np.random.RandomState(int(seed))
        rng.shuffle(order)
    return [df.iloc[order[i:i + batch_size]] for i in range(0, len(order), batch_size)]


def _build_known_unknown_idx(mode_cols_all: List[str], mode_cols_known: List[str]) -> Tuple[List[int], int]:
    idx_known = [mode_cols_all.index(c) for c in mode_cols_known]
    all_idx = set(range(len(mode_cols_all)))
    unk = list(all_idx - set(idx_known))
    assert len(unk) == 1, f"Expected exactly one unknown mode; got {unk}"
    return idx_known, int(unk[0])


def _normalize_dist(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(EPS)
    return x / x.sum(dim=-1, keepdim=True)


def _model_call(policy_model, a, p, ctx, mode_temperature=1.0, cat_temperature=1.0):
    pc, pm, travel = policy_model(
        a, p, ctx,
        mode_temperature=mode_temperature,
        cat_temperature=cat_temperature
    )
    pc = _normalize_dist(pc)
    pm = _normalize_dist(pm)
    return travel, pc, pm


def _resolve_policy_init_kwargs(policy_cls, cfg: Dict, base_kwargs: Dict) -> Dict:
    sig = inspect.signature(policy_cls.__init__)
    kwargs = dict(base_kwargs)
    for name in sig.parameters:
        if name in ("self",):
            continue
        if name not in kwargs and name in cfg:
            kwargs[name] = cfg[name]
    return kwargs


def _shuffle_rows(x: torch.Tensor) -> torch.Tensor:
    if x.size(0) <= 1:
        return x
    idx = torch.randperm(x.size(0), device=x.device)
    return x[idx]


def build_negative_actions(t_exp, pc_exp, pm_exp, t_pol, pc_pol, pm_pol, perturb_scale: float = 0.15):
    t_shuf = _shuffle_rows(t_exp)
    pc_shuf = _shuffle_rows(pc_exp)
    pm_shuf = _shuffle_rows(pm_exp)

    t_neg = torch.clamp(0.65 * t_pol + 0.20 * t_shuf + 0.15 * torch.rand_like(t_pol), 0.0, 1.0)
    pc_mix = 0.65 * pc_pol + 0.20 * pc_shuf + 0.15 * torch.rand_like(pc_pol)
    pm_mix = 0.65 * pm_pol + 0.20 * pm_shuf + 0.15 * torch.rand_like(pm_pol)

    if perturb_scale > 0.0:
        pc_mix = pc_mix + perturb_scale * torch.randn_like(pc_mix).abs()
        pm_mix = pm_mix + perturb_scale * torch.randn_like(pm_mix).abs()

    pc_neg = _normalize_dist(pc_mix)
    pm_neg = _normalize_dist(pm_mix)
    return t_neg, pc_neg, pm_neg


def _phase_transition_weights(days_since_phase: np.ndarray, boost: float, scale: float) -> np.ndarray:
    if boost <= 0:
        return np.ones(len(days_since_phase), dtype=np.float32)
    ds = np.nan_to_num(days_since_phase.astype(np.float32), nan=0.0)
    ds = np.maximum(ds, 0.0)
    return (1.0 + boost * np.exp(-ds / max(scale, 1e-6))).astype(np.float32)


def _compute_context_norm_stats(df: pd.DataFrame, ctx_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if df.empty:
        mean = np.zeros((1, len(ctx_cols)), dtype=np.float32)
        std = np.ones((1, len(ctx_cols)), dtype=np.float32)
        return mean, std
    ctx = df[ctx_cols].to_numpy(dtype=np.float32)
    mean = ctx.mean(axis=0, keepdims=True)
    std = ctx.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _build_phase_reference_actions(
    df: pd.DataFrame,
    ctx_cols: List[str],
    cat_cols: List[str],
    mode_cols_all: List[str],
    ctx_mean: np.ndarray,
    ctx_std: np.ndarray,
) -> Dict[int, Dict[str, np.ndarray]]:
    refs: Dict[int, Dict[str, np.ndarray]] = {}
    for phase_id, g in df.groupby("epi_phase", dropna=False):
        phase = int(phase_id)
        ctx = g[ctx_cols].to_numpy(dtype=np.float32)
        refs[phase] = {
            "ctx": ((ctx - ctx_mean) / ctx_std).astype(np.float32),
            "travel_rows": g["travel_frac"].to_numpy(dtype=np.float32).reshape(-1, 1),
            "pc_rows": g[cat_cols].to_numpy(dtype=np.float32),
            "pm_rows": g[mode_cols_all].to_numpy(dtype=np.float32),
        }
    return refs


def _safe_np_dist(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
    s = float(x.sum())
    if s <= EPS:
        return np.full_like(x, 1.0 / max(1, len(x)))
    return x / s


def _phase_order(refs: Dict[int, Dict[str, np.ndarray]]) -> List[int]:
    return sorted(int(k) for k in refs.keys())


def _pick_other_phase(phase: int, ordered_phases: List[int]) -> int:
    if not ordered_phases:
        return phase
    if len(ordered_phases) == 1:
        return ordered_phases[0]
    if phase not in ordered_phases:
        return ordered_phases[0]
    pos = ordered_phases.index(phase)
    return ordered_phases[(pos + 1) % len(ordered_phases)]


def build_reference_actions_from_source_phases(
    source_phases: np.ndarray,
    ctx_np: np.ndarray,
    phase_refs: Dict[int, Dict[str, np.ndarray]],
    cat_dim: int,
    mode_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    t_rows, pc_rows, pm_rows, src_phases = [], [], [], []
    uniform_pc = np.full(cat_dim, 1.0 / max(1, cat_dim), dtype=np.float32)
    uniform_pm = np.full(mode_dim, 1.0 / max(1, mode_dim), dtype=np.float32)
    for i, src_phase in enumerate(np.asarray(source_phases, dtype=np.int64)):
        ref = phase_refs.get(int(src_phase))
        if ref is None:
            t_rows.append(np.array([[0.5]], dtype=np.float32))
            pc_rows.append(uniform_pc)
            pm_rows.append(uniform_pm)
        else:
            ref_ctx = ref.get("ctx")
            if ref_ctx is not None and len(ref_ctx) > 0:
                q = np.asarray(ctx_np[i], dtype=np.float32)
                dists = np.sum((ref_ctx - q) ** 2, axis=1)
                best = int(np.argmin(dists))
                t_rows.append(ref["travel_rows"][best : best + 1])
                pc_rows.append(_safe_np_dist(ref["pc_rows"][best]))
                pm_rows.append(_safe_np_dist(ref["pm_rows"][best]))
            else:
                t_rows.append(np.array([[0.5]], dtype=np.float32))
                pc_rows.append(uniform_pc)
                pm_rows.append(uniform_pm)
        src_phases.append(int(src_phase))
    t = torch.tensor(np.concatenate(t_rows, axis=0), dtype=torch.float32, device=device)
    pc = torch.tensor(np.stack(pc_rows, axis=0), dtype=torch.float32, device=device)
    pm = torch.tensor(np.stack(pm_rows, axis=0), dtype=torch.float32, device=device)
    return t, pc, pm, np.asarray(src_phases, dtype=np.int64)


def build_phase_negative_actions(
    phases: np.ndarray,
    ctx_np: np.ndarray,
    phase_refs: Dict[int, Dict[str, np.ndarray]],
    cat_dim: int,
    mode_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    ordered = _phase_order(phase_refs)
    src_phases = np.asarray([_pick_other_phase(int(ph), ordered) for ph in phases], dtype=np.int64)
    return build_reference_actions_from_source_phases(src_phases, ctx_np, phase_refs, cat_dim, mode_dim, device)


def _weighted_mean(loss_vec: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (loss_vec * weights).sum() / weights.sum().clamp_min(EPS)


def _weighted_mean_if_any(loss_vec: torch.Tensor, weights: torch.Tensor, device: torch.device) -> torch.Tensor:
    if weights.numel() == 0 or float(weights.sum().detach().item()) <= EPS:
        return torch.zeros((), dtype=torch.float32, device=device)
    return _weighted_mean(loss_vec, weights)


def _attach_phase_boundary_context(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "epi_phase" not in df.columns or "date" not in df.columns:
        return df.copy()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    phase_dates = (
        out[["epi_phase", "date"]]
        .drop_duplicates()
        .sort_values(["epi_phase", "date"])
        .reset_index(drop=True)
    )
    if phase_dates.empty:
        out["prev_phase_id_global"] = out["epi_phase"].astype(int)
        out["has_prev_phase_global"] = 0.0
        return out

    phase_order_df = (
        phase_dates.groupby("epi_phase", as_index=False)["date"]
        .min()
        .sort_values(["date", "epi_phase"])
        .reset_index(drop=True)
    )
    ordered_phases = [int(x) for x in phase_order_df["epi_phase"].tolist()]

    prev_map: Dict[int, int] = {}
    has_prev_map: Dict[int, float] = {}
    for i, phase in enumerate(ordered_phases):
        prev_map[phase] = ordered_phases[i - 1] if i > 0 else phase
        has_prev_map[phase] = 1.0 if i > 0 else 0.0
    out["prev_phase_id_global"] = out["epi_phase"].map(prev_map).fillna(out["epi_phase"]).astype(int)
    out["has_prev_phase_global"] = out["epi_phase"].map(has_prev_map).fillna(0.0).astype(float)
    return out


def _subset_dataframe_from_loader(df: pd.DataFrame, loader) -> pd.DataFrame:
    subset = getattr(loader, "dataset", None)
    indices = getattr(subset, "indices", None)
    if indices is None:
        return df.copy()
    return df.iloc[list(indices)].copy()


def _alignment_selection_score(metrics: Dict[str, float]) -> float:
    return (
        float(metrics["wKL_cat"])
        + float(metrics["wKL_mode"])
        - 0.30 * float(metrics["R2_cat_hours"])
        - 0.50 * float(metrics["R2_mode_hours"])
    )


PROFILE_DEFAULTS: Dict[str, Dict[str, float | int | str]] = {
    "factual": {
        "selection_mode": "train_loss",
        "lambda_pref": 0.30,
        "lambda_phase_pref": 0.40,
        "lambda_bc_cat": 0.70,
        "lambda_bc_mode": 0.70,
        "lambda_bc_travel": 0.70,
        "lambda_bc_cat_h": 0.70,
        "lambda_bc_mode_h": 0.70,
        "lambda_anchor_cat": 0.05,
        "lambda_anchor_mode": 0.05,
        "lambda_anchor_travel": 0.05,
        "lambda_rollout_cat": 0.25,
        "lambda_rollout_mode": 0.25,
        "lambda_rollout_travel": 0.20,
        "rollout_eval_weight": 1.0,
    },
    "counterfactual": {
        "selection_mode": "last",
        "score_epochs": 80,
        "finetune_epochs": 24,
        "batch_size": 1024,
        "lr_policy": 1e-4,
        "phase_rank_weight": 1.00,
        "pol_phase_weight": 0.75,
        "phase_neg_weight": 0.0,
        "phase_transition_boost": 3.0,
        "phase_transition_scale": 3.0,
        "lambda_pref": 0.30,
        "lambda_phase_pref": 0.50,
        "lambda_bc_cat": 0.25,
        "lambda_bc_mode": 0.25,
        "lambda_bc_travel": 0.25,
        "lambda_bc_cat_h": 0.20,
        "lambda_bc_mode_h": 0.20,
        "lambda_anchor_cat": 0.20,
        "lambda_anchor_mode": 0.20,
        "lambda_anchor_travel": 0.20,
        "lambda_rollout_cat": 0.0,
        "lambda_rollout_mode": 0.0,
        "lambda_rollout_travel": 0.0,
        "lambda_pre_policy_cat": 0.0,
        "lambda_pre_policy_mode": 0.0,
        "lambda_pre_policy_travel": 0.0,
        # post_phase_response: penalise model if it fails to change behaviour
        # significantly at phase transitions (data-driven, no hard rules).
        # Margin = minimum KL distance from prev-phase behaviour required.
        "lambda_post_phase_response_cat": 0.40,
        "lambda_post_phase_response_mode": 0.30,
        "lambda_post_phase_response_travel": 0.30,
        "post_phase_response_cat_margin": 0.28,
        "post_phase_response_mode_margin": 0.20,
        "post_phase_response_travel_margin": 0.10,
        # phase_wrong: penalise model if its output for phase P is indistinguishable
        # from reference behaviour of a wrong phase (scored by the preference net).
        "lambda_phase_wrong_cat": 0.20,
        "lambda_phase_wrong_mode": 0.15,
        "lambda_phase_wrong_travel": 0.15,
        "phase_wrong_cat_margin": 0.30,
        "phase_wrong_mode_margin": 0.22,
        "phase_wrong_travel_margin": 0.10,
    },
}


def _resolve_home_cat_position(feature_meta_path: Path, cat_cols: List[str]) -> int | None:
    if not feature_meta_path.exists():
        return None
    try:
        meta = json.loads(feature_meta_path.read_text())
        home_cat_idx = int(meta.get("home_cat_idx", -1))
    except Exception:
        return None
    home_col = f"cat_{home_cat_idx}"
    return cat_cols.index(home_col) if home_col in cat_cols else None


def _attach_one_step_rollout_targets(
    df: pd.DataFrame,
    ctx_cols: List[str],
    cat_cols: List[str],
    mode_cols_all: List[str],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["agent_id", "date"]).reset_index(drop=True)
    grp = out.groupby("agent_id", sort=False)
    next_date = grp["date"].shift(-1)
    contiguous = next_date.notna() & ((next_date - out["date"]).dt.days == 1)

    out["rollout_has_next"] = contiguous.astype(np.float32)
    out["next_phase_id"] = grp["epi_phase"].shift(-1).where(contiguous, 0).fillna(0).astype(int)
    out["next_day_hours"] = grp["day_hours"].shift(-1).where(contiguous, 24.0).fillna(24.0).astype(float)
    out["next_travel_frac"] = grp["travel_frac"].shift(-1).where(contiguous, 0.0).fillna(0.0).astype(float)

    for col in cat_cols + mode_cols_all:
        out[f"next_target__{col}"] = grp[col].shift(-1).where(contiguous, 0.0).fillna(0.0).astype(float)
    for c in ctx_cols:
        out[f"next_ctx__{c}"] = grp[c].shift(-1).where(contiguous, 0.0).fillna(0.0).astype(float)

    out["date"] = out["date"].dt.date
    return out


def _one_step_rollout_losses(
    batch_df: pd.DataFrame,
    *,
    policy_model: nn.Module,
    a_cur: torch.Tensor,
    t_cur: torch.Tensor,
    pc_cur: torch.Tensor,
    ctx_cols: List[str],
    cat_cols: List[str],
    mode_cols_all: List[str],
    home_cat_pos: int | None,
    device_t: torch.device,
) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=torch.float32, device=device_t)
    if batch_df.empty or "rollout_has_next" not in batch_df.columns:
        return {"cat": zero, "mode": zero, "travel": zero}

    mask_np = pd.to_numeric(batch_df["rollout_has_next"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) > 0.5
    if not mask_np.any():
        return {"cat": zero, "mode": zero, "travel": zero}

    next_ctx_cols = [f"next_ctx__{c}" for c in ctx_cols]
    next_target_cat_cols = [f"next_target__{c}" for c in cat_cols]
    next_target_mode_cols = [f"next_target__{c}" for c in mode_cols_all]
    required_cols = next_ctx_cols + next_target_cat_cols + next_target_mode_cols + ["next_phase_id", "next_day_hours", "next_travel_frac"]
    if any(c not in batch_df.columns for c in required_cols):
        return {"cat": zero, "mode": zero, "travel": zero}

    mask = torch.tensor(mask_np, dtype=torch.bool, device=device_t)
    if int(mask.sum().item()) == 0:
        return {"cat": zero, "mode": zero, "travel": zero}

    x_next = torch.tensor(batch_df[next_ctx_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_t)[mask].clone()
    a_next = a_cur[mask]
    p_next = torch.tensor(batch_df["next_phase_id"].to_numpy(dtype=np.int64), dtype=torch.long, device=device_t)[mask]

    t_cur_sel = t_cur[mask]
    pc_cur_sel = pc_cur[mask]
    current_true_travel = torch.tensor(batch_df["travel_frac"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_t)[mask]
    current_true_cat = torch.tensor(batch_df[cat_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_t)[mask]

    ctx_pos = {c: i for i, c in enumerate(ctx_cols)}
    if "prev_travel_frac" in ctx_pos:
        x_next[:, ctx_pos["prev_travel_frac"]] = t_cur_sel.squeeze(-1)
    if "prev7_travel_frac_mean" in ctx_pos:
        x_next[:, ctx_pos["prev7_travel_frac_mean"]] = (
            x_next[:, ctx_pos["prev7_travel_frac_mean"]] + (t_cur_sel.squeeze(-1) - current_true_travel) / 7.0
        )
    if home_cat_pos is not None and 0 <= home_cat_pos < pc_cur_sel.size(-1):
        home_pred = pc_cur_sel[:, home_cat_pos]
        home_true = current_true_cat[:, home_cat_pos]
        if "prev_home_share" in ctx_pos:
            x_next[:, ctx_pos["prev_home_share"]] = home_pred
        if "prev7_home_share_mean" in ctx_pos:
            x_next[:, ctx_pos["prev7_home_share_mean"]] = (
                x_next[:, ctx_pos["prev7_home_share_mean"]] + (home_pred - home_true) / 7.0
            )
    for j, cat_col in enumerate(cat_cols):
        try:
            cat_id = int(cat_col.split("_")[1])
        except Exception:
            continue
        prev_key = f"prev_cat_{cat_id}"
        prev7_key = f"prev7_cat_{cat_id}_mean"
        if prev_key in ctx_pos:
            x_next[:, ctx_pos[prev_key]] = pc_cur_sel[:, j]
        if prev7_key in ctx_pos:
            x_next[:, ctx_pos[prev7_key]] = (
                x_next[:, ctx_pos[prev7_key]] + (pc_cur_sel[:, j] - current_true_cat[:, j]) / 7.0
            )

    t_next, pc_next, pm_next = _model_call(policy_model, a_next, p_next, x_next)
    t_next_exp = torch.tensor(batch_df["next_travel_frac"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_t)[mask].unsqueeze(-1)
    pc_next_exp = torch.tensor(batch_df[next_target_cat_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_t)[mask]
    pm_next_exp = torch.tensor(batch_df[next_target_mode_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_t)[mask]
    next_day_hours = torch.tensor(batch_df["next_day_hours"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_t)[mask].unsqueeze(-1)
    w_day = (next_day_hours / 24.0).squeeze(-1)
    w_mode = t_next_exp.squeeze(-1) * w_day

    loss_cat = _weighted_mean(kl_categorical(pc_next_exp, pc_next), w_day)
    loss_mode = _weighted_mean(kl_categorical(pm_next_exp, pm_next), torch.clamp(w_mode, min=1e-4))
    loss_travel = _weighted_mean(((t_next - t_next_exp) ** 2).squeeze(-1), w_day)
    return {"cat": loss_cat, "mode": loss_mode, "travel": loss_travel}


def _choose_selection_mode(profile: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if profile == "counterfactual":
        return "last"
    return "val"


def _loader_has_data(loader) -> bool:
    if loader is None:
        return False
    dataset = getattr(loader, "dataset", None)
    try:
        return dataset is not None and len(dataset) > 0
    except TypeError:
        return False


def _apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    default_map = {
        "score_epochs": 10,
        "finetune_epochs": 2,
        "batch_size": 4096,
        "lr_policy": 5e-5,
        "phase_transition_boost": 1.0,
        "phase_transition_scale": 7.0,
        "lambda_pref": 0.10,
        "lambda_phase_pref": 0.15,
        "lambda_bc_cat": 1.0,
        "lambda_bc_mode": 1.0,
        "lambda_bc_travel": 1.0,
        "lambda_anchor_cat": 0.15,
        "lambda_anchor_mode": 0.15,
        "lambda_anchor_travel": 0.15,
        "lambda_bc_cat_h": 0.25,
        "lambda_bc_mode_h": 0.40,
        "lambda_rollout_cat": 0.0,
        "lambda_rollout_mode": 0.0,
        "lambda_rollout_travel": 0.0,
        "rollout_eval_weight": 0.0,
        "lambda_pre_policy_cat": 0.0,
        "lambda_pre_policy_mode": 0.0,
        "lambda_pre_policy_travel": 0.0,
        "lambda_post_phase_response_cat": 0.0,
        "lambda_post_phase_response_mode": 0.0,
        "lambda_post_phase_response_travel": 0.0,
        "post_phase_response_cat_margin": 0.20,
        "post_phase_response_mode_margin": 0.15,
        "post_phase_response_travel_margin": 0.08,
        "phase_neg_weight": 0.0,
        "lambda_phase_wrong_cat": 0.0,
        "lambda_phase_wrong_mode": 0.0,
        "lambda_phase_wrong_travel": 0.0,
        "phase_wrong_cat_margin": 0.20,
        "phase_wrong_mode_margin": 0.15,
        "phase_wrong_travel_margin": 0.08,
        "selection_mode": "auto",
    }
    if args.alignment_profile not in PROFILE_DEFAULTS:
        return args
    for key, value in PROFILE_DEFAULTS[args.alignment_profile].items():
        if hasattr(args, key) and getattr(args, key) == default_map.get(key):
            setattr(args, key, value)
    return args


def train_preference_scorer(
    policy_model: nn.Module,
    df: pd.DataFrame,
    ctx_cols: List[str],
    cat_cols: List[str],
    mode_cols_all: List[str],
    device: str,
    epochs: int = 10,
    batch_size: int = 4096,
    lr: float = 1e-3,
    h_dim: int = 128,
    dropout: float = 0.1,
    perturb_scale: float = 0.10,
    phase_rank_weight: float = 1.00,
    pol_phase_weight: float = 0.75,
    pol_neg_weight: float = 0.25,
    phase_neg_weight: float = 0.0,
    phase_transition_boost: float = 3.0,
    phase_transition_scale: float = 3.0,
) -> Tuple[PreferenceScoreNet, Dict]:
    # Normalize device string: '0' -> 'cuda:0', '1' -> 'cuda:1', etc.
    if device.isdigit():
        device = f'cuda:{device}'
    device_t = torch.device(device)
    s_dim = len(ctx_cols)
    a_dim = 1 + len(cat_cols) + len(mode_cols_all)
    snet = PreferenceScoreNet(PreferenceScoreCfg(s_dim=s_dim, a_dim=a_dim, h_dim=h_dim, dropout=dropout)).to(device_t)
    opt = torch.optim.AdamW(snet.parameters(), lr=lr)
    ctx_mean, ctx_std = _compute_context_norm_stats(df, ctx_cols)
    phase_refs = _build_phase_reference_actions(df, ctx_cols, cat_cols, mode_cols_all, ctx_mean, ctx_std)
    history = {
        "pairwise_loss": [],
        "phase_loss": [],
        "hier_pol_phase_loss": [],
        "hier_pol_neg_loss": [],
        "hier_phase_neg_loss": [],
    }

    policy_model.eval()

    for ep in range(1, epochs + 1):
        batches = make_batches(df, batch_size, shuffle=True, seed=ep)
        ep_loss = 0.0
        ep_phase = 0.0
        ep_pol_phase = 0.0
        ep_pol_neg = 0.0
        ep_phase_neg = 0.0
        for bi, b in enumerate(batches):
            s = torch.tensor(b[ctx_cols].values, dtype=torch.float32, device=device_t)
            s_np = b[ctx_cols].to_numpy(dtype=np.float32)
            s_np_norm = ((s_np - ctx_mean) / ctx_std).astype(np.float32)
            phase_vals = b["epi_phase"].astype(int).to_numpy()
            sample_weights = torch.tensor(
                _phase_transition_weights(
                    b.get("days_since_phase", pd.Series(np.zeros(len(b), dtype=np.float32))).to_numpy(dtype=np.float32),
                    phase_transition_boost,
                    phase_transition_scale,
                ),
                dtype=torch.float32,
                device=device_t,
            )
            t_exp = torch.tensor(b["travel_frac"].values, dtype=torch.float32, device=device_t).unsqueeze(-1)
            pc_exp = torch.tensor(b[cat_cols].values, dtype=torch.float32, device=device_t)
            pm_exp = torch.tensor(b[mode_cols_all].values, dtype=torch.float32, device=device_t)

            a = torch.tensor(b["agent_idx"].values, dtype=torch.long, device=device_t)
            p = torch.tensor(b["epi_phase"].values, dtype=torch.long, device=device_t)
            x = torch.tensor(s_np, dtype=torch.float32, device=device_t)
            with torch.no_grad():
                t_pol, pc_pol, pm_pol = _model_call(policy_model, a, p, x)
            t_neg, pc_neg, pm_neg = build_negative_actions(t_exp, pc_exp, pm_exp, t_pol, pc_pol, pm_pol, perturb_scale=perturb_scale)
            t_phase_other, pc_phase_other, pm_phase_other, _ = build_phase_negative_actions(
                phase_vals,
                s_np_norm,
                phase_refs,
                len(cat_cols),
                len(mode_cols_all),
                device_t,
            )
            has_prev_phase_np = (
                pd.to_numeric(b.get("has_prev_phase_global", pd.Series(np.zeros(len(b)))), errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
                > 0.5
            )
            if has_prev_phase_np.any():
                prev_phase_ids = b.get("prev_phase_id_global", pd.Series(phase_vals)).to_numpy(dtype=np.int64)
                t_phase_prev, pc_phase_prev, pm_phase_prev, _ = build_reference_actions_from_source_phases(
                    prev_phase_ids,
                    s_np_norm,
                    phase_refs,
                    len(cat_cols),
                    len(mode_cols_all),
                    device_t,
                )
                prev_mask_2d = torch.tensor(has_prev_phase_np[:, None], dtype=torch.bool, device=device_t)
                prev_mask_1d = torch.tensor(has_prev_phase_np, dtype=torch.bool, device=device_t)
                t_phase = torch.where(prev_mask_2d, t_phase_prev, t_phase_other)
                pc_phase = torch.where(prev_mask_1d.unsqueeze(-1), pc_phase_prev, pc_phase_other)
                pm_phase = torch.where(prev_mask_1d.unsqueeze(-1), pm_phase_prev, pm_phase_other)
            else:
                t_phase, pc_phase, pm_phase = t_phase_other, pc_phase_other, pm_phase_other

            a_exp = pack_actions(t_exp, pc_exp, pm_exp)
            a_pol = pack_actions(t_pol, pc_pol, pm_pol)
            a_neg = pack_actions(t_neg, pc_neg, pm_neg)
            a_phase = pack_actions(t_phase, pc_phase, pm_phase)

            score_exp = snet(s, a_exp)
            score_pol = snet(s, a_pol)
            score_neg = snet(s, a_neg)
            score_phase = snet(s, a_phase)

            loss_pol = _weighted_mean(F.softplus(-(score_exp - score_pol)), sample_weights)
            loss_neg = _weighted_mean(F.softplus(-(score_exp - score_neg)), sample_weights)
            loss_phase = _weighted_mean(F.softplus(-(score_exp - score_phase)), sample_weights)
            loss_pol_phase = _weighted_mean(F.softplus(-(score_pol - score_phase)), sample_weights)
            loss_pol_neg = _weighted_mean(F.softplus(-(score_pol - score_neg)), sample_weights)
            loss_phase_neg = _weighted_mean(F.softplus(-(score_phase - score_neg)), sample_weights)
            loss_rank = (
                0.5 * (loss_pol + loss_neg)
                + phase_rank_weight * loss_phase
                + pol_phase_weight * loss_pol_phase
                + pol_neg_weight * loss_pol_neg
                + phase_neg_weight * loss_phase_neg
            )
            loss_reg = 1e-4 * ((score_exp ** 2).mean() + (score_pol ** 2).mean() + (score_neg ** 2).mean() + (score_phase ** 2).mean())
            loss = loss_rank + loss_reg

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(snet.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss)
            ep_phase += float(loss_phase)
            ep_pol_phase += float(loss_pol_phase)
            ep_pol_neg += float(loss_pol_neg)
            ep_phase_neg += float(loss_phase_neg)

            if (bi % 100) == 0:
                print(
                    f"[PrefScore] epoch {ep} batch {bi}/{len(batches)} "
                    f"loss={float(loss):.4f} phase_loss={float(loss_phase):.4f} "
                    f"pol>phase={float(loss_pol_phase):.4f} phase_vs_neg={float(loss_phase_neg):.4f}"
                )

        ep_loss /= max(1, len(batches))
        ep_phase /= max(1, len(batches))
        ep_pol_phase /= max(1, len(batches))
        ep_pol_neg /= max(1, len(batches))
        ep_phase_neg /= max(1, len(batches))
        history["pairwise_loss"].append(ep_loss)
        history["phase_loss"].append(ep_phase)
        history["hier_pol_phase_loss"].append(ep_pol_phase)
        history["hier_pol_neg_loss"].append(ep_pol_neg)
        history["hier_phase_neg_loss"].append(ep_phase_neg)
        print(
            f"[PrefScore] epoch {ep} mean loss={ep_loss:.4f} phase_loss={ep_phase:.4f} "
            f"pol>phase={ep_pol_phase:.4f} pol>neg={ep_pol_neg:.4f} phase_vs_neg={ep_phase_neg:.4f}"
        )

    return snet, history


def _sample_frame(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if df.empty or max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    rng = np.random.RandomState(int(seed))
    picked = rng.choice(len(df), size=int(max_rows), replace=False)
    return df.iloc[np.sort(picked)].copy()


def _action_matrix_from_df(df: pd.DataFrame, cat_cols: List[str], mode_cols_all: List[str]) -> np.ndarray:
    if df.empty:
        return np.zeros((0, 1 + len(cat_cols) + len(mode_cols_all)), dtype=np.float32)
    return np.concatenate(
        [
            df[["travel_frac"]].to_numpy(dtype=np.float32),
            df[cat_cols].to_numpy(dtype=np.float32),
            df[mode_cols_all].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )


def _nearest_action_distance(query: np.ndarray, reference: np.ndarray, query_chunk: int = 128) -> np.ndarray:
    if len(query) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(reference) == 0:
        return np.full((len(query),), np.nan, dtype=np.float32)
    out = np.full((len(query),), np.inf, dtype=np.float32)
    for start in range(0, len(query), max(1, int(query_chunk))):
        stop = min(len(query), start + max(1, int(query_chunk)))
        q = query[start:stop].astype(np.float32, copy=False)
        diff = q[:, None, :] - reference[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float32))
        out[start:stop] = dists.min(axis=1)
    return out


@torch.no_grad()
def evaluate_preference_diagnostics(
    *,
    policy_model: nn.Module,
    scorer: PreferenceScoreNet,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    ctx_cols: List[str],
    cat_cols: List[str],
    mode_cols_all: List[str],
    device: str,
    max_eval_rows: int = 4096,
    max_support_rows: int = 4096,
    seed: int = 1234,
) -> tuple[Dict[str, float], pd.DataFrame]:
    if train_df.empty or eval_df.empty:
        return {}, pd.DataFrame()

    if device.isdigit():
        device = f"cuda:{device}"
    device_t = torch.device(device)

    train_df = _attach_phase_boundary_context(train_df)
    eval_df = _attach_phase_boundary_context(eval_df)
    train_ref = _sample_frame(train_df, max_support_rows, seed)
    eval_ref = _sample_frame(eval_df, max_eval_rows, seed + 17)
    if train_ref.empty or eval_ref.empty:
        return {}, pd.DataFrame()

    ctx_mean, ctx_std = _compute_context_norm_stats(train_ref, ctx_cols)
    phase_refs = _build_phase_reference_actions(train_ref, ctx_cols, cat_cols, mode_cols_all, ctx_mean, ctx_std)
    s_np = eval_ref[ctx_cols].to_numpy(dtype=np.float32)
    s_np_norm = ((s_np - ctx_mean) / ctx_std).astype(np.float32)
    phase_vals = eval_ref["epi_phase"].astype(int).to_numpy()

    s = torch.tensor(s_np, dtype=torch.float32, device=device_t)
    a = torch.tensor(eval_ref["agent_idx"].to_numpy(dtype=np.int64), dtype=torch.long, device=device_t)
    p = torch.tensor(phase_vals, dtype=torch.long, device=device_t)
    t_exp = torch.tensor(eval_ref["travel_frac"].to_numpy(dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=device_t)
    pc_exp = torch.tensor(eval_ref[cat_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_t)
    pm_exp = torch.tensor(eval_ref[mode_cols_all].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_t)

    policy_model.eval()
    scorer.eval()
    t_pol, pc_pol, pm_pol = _model_call(policy_model, a, p, s)
    t_neg, pc_neg, pm_neg = build_negative_actions(t_exp, pc_exp, pm_exp, t_pol, pc_pol, pm_pol, perturb_scale=0.10)
    t_phase_other, pc_phase_other, pm_phase_other, _ = build_phase_negative_actions(
        phase_vals,
        s_np_norm,
        phase_refs,
        len(cat_cols),
        len(mode_cols_all),
        device_t,
    )
    has_prev_phase_np = (
        pd.to_numeric(eval_ref.get("has_prev_phase_global", pd.Series(np.zeros(len(eval_ref)))), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
        > 0.5
    )
    if has_prev_phase_np.any():
        prev_phase_ids = eval_ref.get("prev_phase_id_global", pd.Series(phase_vals)).to_numpy(dtype=np.int64)
        t_phase_prev, pc_phase_prev, pm_phase_prev, _ = build_reference_actions_from_source_phases(
            prev_phase_ids,
            s_np_norm,
            phase_refs,
            len(cat_cols),
            len(mode_cols_all),
            device_t,
        )
        prev_mask_2d = torch.tensor(has_prev_phase_np[:, None], dtype=torch.bool, device=device_t)
        prev_mask_1d = torch.tensor(has_prev_phase_np, dtype=torch.bool, device=device_t)
        t_phase = torch.where(prev_mask_2d, t_phase_prev, t_phase_other)
        pc_phase = torch.where(prev_mask_1d.unsqueeze(-1), pc_phase_prev, pc_phase_other)
        pm_phase = torch.where(prev_mask_1d.unsqueeze(-1), pm_phase_prev, pm_phase_other)
    else:
        t_phase, pc_phase, pm_phase = t_phase_other, pc_phase_other, pm_phase_other

    score_exp = scorer(s, pack_actions(t_exp, pc_exp, pm_exp))
    score_pol = scorer(s, pack_actions(t_pol, pc_pol, pm_pol))
    score_neg = scorer(s, pack_actions(t_neg, pc_neg, pm_neg))
    score_phase = scorer(s, pack_actions(t_phase, pc_phase, pm_phase))

    pol_action = torch.cat([t_pol, pc_pol, pm_pol], dim=-1).cpu().numpy().astype(np.float32)
    obs_action = _action_matrix_from_df(eval_ref, cat_cols, mode_cols_all)
    global_ref = _action_matrix_from_df(train_ref, cat_cols, mode_cols_all)
    pol_nn_global = _nearest_action_distance(pol_action, global_ref)
    obs_nn_global = _nearest_action_distance(obs_action, global_ref)

    pol_nn_same = np.full((len(eval_ref),), np.nan, dtype=np.float32)
    obs_nn_same = np.full((len(eval_ref),), np.nan, dtype=np.float32)
    for phase in sorted(eval_ref["epi_phase"].dropna().astype(int).unique().tolist()):
        mask = eval_ref["epi_phase"].astype(int).to_numpy() == int(phase)
        phase_train = train_df[train_df["epi_phase"].astype(int) == int(phase)]
        ref_df = _sample_frame(phase_train if not phase_train.empty else train_ref, max_support_rows, seed + int(phase) * 19 + 5)
        ref_mat = _action_matrix_from_df(ref_df, cat_cols, mode_cols_all)
        pol_nn_same[mask] = _nearest_action_distance(pol_action[mask], ref_mat)
        obs_nn_same[mask] = _nearest_action_distance(obs_action[mask], ref_mat)

    detail = eval_ref[["agent_id", "date", "epi_phase"]].copy()
    detail["score_exp"] = score_exp.squeeze(-1).cpu().numpy()
    detail["score_pol"] = score_pol.squeeze(-1).cpu().numpy()
    detail["score_neg"] = score_neg.squeeze(-1).cpu().numpy()
    detail["score_phase"] = score_phase.squeeze(-1).cpu().numpy()
    detail["margin_exp_pol"] = detail["score_exp"] - detail["score_pol"]
    detail["margin_exp_neg"] = detail["score_exp"] - detail["score_neg"]
    detail["margin_exp_phase"] = detail["score_exp"] - detail["score_phase"]
    detail["margin_pol_neg"] = detail["score_pol"] - detail["score_neg"]
    detail["margin_pol_phase"] = detail["score_pol"] - detail["score_phase"]
    detail["support_nn_global_pol"] = pol_nn_global
    detail["support_nn_global_obs"] = obs_nn_global
    detail["support_nn_same_phase_pol"] = pol_nn_same
    detail["support_nn_same_phase_obs"] = obs_nn_same

    eps = 1e-8
    summary = {
        "n_eval_rows": int(len(detail)),
        "n_support_rows": int(len(train_ref)),
        "acc_exp_gt_pol": float((detail["margin_exp_pol"] > 0).mean()),
        "acc_exp_gt_neg": float((detail["margin_exp_neg"] > 0).mean()),
        "acc_exp_gt_phase": float((detail["margin_exp_phase"] > 0).mean()),
        "acc_pol_gt_neg": float((detail["margin_pol_neg"] > 0).mean()),
        "acc_pol_gt_phase": float((detail["margin_pol_phase"] > 0).mean()),
        "margin_exp_pol_mean": float(detail["margin_exp_pol"].mean()),
        "margin_exp_neg_mean": float(detail["margin_exp_neg"].mean()),
        "margin_exp_phase_mean": float(detail["margin_exp_phase"].mean()),
        "support_nn_global_pol_mean": float(np.nanmean(pol_nn_global)),
        "support_nn_global_pol_p95": float(np.nanpercentile(pol_nn_global, 95)),
        "support_nn_same_phase_pol_mean": float(np.nanmean(pol_nn_same)),
        "support_nn_same_phase_pol_p95": float(np.nanpercentile(pol_nn_same, 95)),
        "support_nn_global_obs_mean": float(np.nanmean(obs_nn_global)),
        "support_nn_same_phase_obs_mean": float(np.nanmean(obs_nn_same)),
        "support_global_ratio_vs_obs": float(np.nanmean(pol_nn_global) / max(float(np.nanmean(obs_nn_global)), eps)),
        "support_same_phase_ratio_vs_obs": float(np.nanmean(pol_nn_same) / max(float(np.nanmean(obs_nn_same)), eps)),
    }
    return summary, detail


def preference_align_finetune(
    policy_model: nn.Module,
    df: pd.DataFrame,
    ctx_cols: List[str],
    cat_cols: List[str],
    mode_cols_all: List[str],
    scorer: PreferenceScoreNet,
    device: str,
    epochs: int = 2,
    batch_size: int = 2048,
    lr: float = 5e-5,
    lambda_pref: float = 0.10,
    lambda_phase_pref: float = 0.50,
    lambda_bc_cat: float = 1.0,
    lambda_bc_mode: float = 1.0,
    lambda_bc_travel: float = 1.0,
    lambda_anchor_cat: float = 0.15,
    lambda_anchor_mode: float = 0.15,
    lambda_anchor_travel: float = 0.15,
    lambda_bc_cat_h: float = 0.25,
    lambda_bc_mode_h: float = 0.40,
    lambda_zero_cat: float = 0.01,
    lambda_zero_mode: float = 0.05,
    lambda_unknown_h: float = 0.15,
    lambda_rollout_cat: float = 0.0,
    lambda_rollout_mode: float = 0.0,
    lambda_rollout_travel: float = 0.0,
    lambda_pre_policy_cat: float = 0.0,
    lambda_pre_policy_mode: float = 0.0,
    lambda_pre_policy_travel: float = 0.0,
    lambda_post_phase_response_cat: float = 0.0,
    lambda_post_phase_response_mode: float = 0.0,
    lambda_post_phase_response_travel: float = 0.0,
    post_phase_response_cat_margin: float = 0.20,
    post_phase_response_mode_margin: float = 0.15,
    post_phase_response_travel_margin: float = 0.08,
    lambda_phase_wrong_cat: float = 0.0,
    lambda_phase_wrong_mode: float = 0.0,
    lambda_phase_wrong_travel: float = 0.0,
    phase_wrong_cat_margin: float = 0.20,
    phase_wrong_mode_margin: float = 0.15,
    phase_wrong_travel_margin: float = 0.08,
    val_loader=None,
    eval_fn=None,
    eval_cat_cols: List[str] | None = None,
    mode_known_idx: List[int] | None = None,
    eval_device: str | None = None,
    phase_transition_boost: float = 3.0,
    phase_transition_scale: float = 3.0,
    selection_mode: str = "val",
    rollout_eval_fn=None,
    rollout_dataset=None,
    rollout_eval_weight: float = 0.0,
    home_cat_pos: int | None = None,
):
    # Normalize device string: '0' -> 'cuda:0', '1' -> 'cuda:1', etc.
    if device.isdigit():
        device = f'cuda:{device}'
    device_t = torch.device(device)
    opt = torch.optim.AdamW(policy_model.parameters(), lr=lr)
    ctx_mean, ctx_std = _compute_context_norm_stats(df, ctx_cols)
    phase_refs = _build_phase_reference_actions(df, ctx_cols, cat_cols, mode_cols_all, ctx_mean, ctx_std)

    # Identify lag feature indices in ctx_cols for CF distribution shift augmentation
    lag_col_indices = [i for i, c in enumerate(ctx_cols)
                       if c.startswith("prev_") or c.startswith("prev7_")]

    # Build per-phase mean lag feature values (in original scale) for augmentation
    phase_lag_means: Dict[int, np.ndarray] = {}
    for phase_id, g in df.groupby("epi_phase", dropna=False):
        if lag_col_indices:
            phase_lag_means[int(phase_id)] = g[ctx_cols].to_numpy(dtype=np.float32)[:, lag_col_indices].mean(axis=0)

    base_model = type(policy_model)(**policy_model._init_kwargs) if hasattr(policy_model, '_init_kwargs') else None
    if base_model is not None:
        base_model.load_state_dict(policy_model.state_dict(), strict=False)
        base_model.to(device_t)
        base_model.eval()

    policy_model.train()
    scorer.eval()
    hist = {
        "loss": [],
        "phase_pref_loss": [],
        "bc_cat_h_loss": [],
        "bc_mode_h_loss": [],
        "bc_unknown_h_loss": [],
        "zero_cat_loss": [],
        "zero_mode_loss": [],
        "rollout_cat_loss": [],
        "rollout_mode_loss": [],
        "rollout_travel_loss": [],
        "pre_policy_cat_loss": [],
        "pre_policy_mode_loss": [],
        "pre_policy_travel_loss": [],
        "post_phase_response_cat_loss": [],
        "post_phase_response_mode_loss": [],
        "post_phase_response_travel_loss": [],
        "phase_wrong_cat_loss": [],
        "phase_wrong_mode_loss": [],
        "phase_wrong_travel_loss": [],
        "val_selection_score": [],
        "best_epoch": None,
        "best_val_selection_score": None,
        "selection_mode": selection_mode,
    }
    best_state = copy.deepcopy(policy_model.state_dict())
    best_epoch = 0
    best_score = float("inf")
    unk_idx = int(getattr(policy_model, "unk_mode_idx", -1))
    last_state = copy.deepcopy(policy_model.state_dict())

    for ep in range(1, epochs + 1):
        batches = make_batches(df, batch_size, shuffle=True, seed=1000 + ep)
        ep_loss = 0.0
        ep_phase_pref = 0.0
        ep_cat_h = 0.0
        ep_mode_h = 0.0
        ep_unknown_h = 0.0
        ep_zero_cat = 0.0
        ep_zero_mode = 0.0
        ep_rollout_cat = 0.0
        ep_rollout_mode = 0.0
        ep_rollout_travel = 0.0
        ep_pre_policy_cat = 0.0
        ep_pre_policy_mode = 0.0
        ep_pre_policy_travel = 0.0
        ep_post_phase_response_cat = 0.0
        ep_post_phase_response_mode = 0.0
        ep_post_phase_response_travel = 0.0
        ep_phase_wrong_cat = 0.0
        ep_phase_wrong_mode = 0.0
        ep_phase_wrong_travel = 0.0
        for bi, b in enumerate(batches):
            a = torch.tensor(b["agent_idx"].values, dtype=torch.long, device=device_t)
            p = torch.tensor(b["epi_phase"].values, dtype=torch.long, device=device_t)
            x_np = b[ctx_cols].to_numpy(dtype=np.float32)

            # CF distribution shift augmentation: for phase-transition samples,
            # replace lag features with prev-phase means at 50% probability,
            # teaching the model to respond to phase even when lag features haven't adapted.
            if lag_col_indices and phase_lag_means:
                has_prev = b.get("has_prev_phase_global", pd.Series(np.zeros(len(b)))).to_numpy(dtype=np.float32)
                prev_phases = b.get("prev_phase_id_global", b["epi_phase"]).to_numpy(dtype=np.int64)
                mask = (has_prev > 0.5) & (np.random.rand(len(b)) < 0.8)
                for idx_in_batch in np.where(mask)[0]:
                    prev_ph = int(prev_phases[idx_in_batch])
                    if prev_ph in phase_lag_means:
                        x_np[idx_in_batch, lag_col_indices] = phase_lag_means[prev_ph]

            x = torch.tensor(x_np, dtype=torch.float32, device=device_t)
            x_np_norm = ((x_np - ctx_mean) / ctx_std).astype(np.float32)
            phase_vals = b["epi_phase"].astype(int).to_numpy()
            sample_weights = torch.tensor(
                _phase_transition_weights(
                    b.get("days_since_phase", pd.Series(np.zeros(len(b), dtype=np.float32))).to_numpy(dtype=np.float32),
                    phase_transition_boost,
                    phase_transition_scale,
                ),
                dtype=torch.float32,
                device=device_t,
            )
            has_prev_phase = torch.tensor(
                b.get("has_prev_phase_global", pd.Series(np.zeros(len(b), dtype=np.float32))).to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=device_t,
            )

            t_exp = torch.tensor(b["travel_frac"].values, dtype=torch.float32, device=device_t).unsqueeze(-1)
            pc_exp = torch.tensor(b[cat_cols].values, dtype=torch.float32, device=device_t)
            pm_exp = torch.tensor(b[mode_cols_all].values, dtype=torch.float32, device=device_t)
            t_phase_other, pc_phase_other, pm_phase_other, _ = build_phase_negative_actions(
                phase_vals,
                x_np_norm,
                phase_refs,
                len(cat_cols),
                len(mode_cols_all),
                device_t,
            )
            prev_phase_ids = b.get("prev_phase_id_global", pd.Series(phase_vals)).to_numpy(dtype=np.int64)
            t_prev, pc_prev, pm_prev, _ = build_reference_actions_from_source_phases(
                prev_phase_ids,
                x_np_norm,
                phase_refs,
                len(cat_cols),
                len(mode_cols_all),
                device_t,
            )
            has_prev_phase_mask = has_prev_phase > 0.5
            t_phase = torch.where(has_prev_phase_mask.unsqueeze(-1), t_prev, t_phase_other)
            pc_phase = torch.where(has_prev_phase_mask.unsqueeze(-1), pc_prev, pc_phase_other)
            pm_phase = torch.where(has_prev_phase_mask.unsqueeze(-1), pm_prev, pm_phase_other)

            with torch.no_grad():
                if base_model is not None:
                    t_base, pc_base, pm_base = _model_call(base_model, a, p, x)
                else:
                    t_base, pc_base, pm_base = _model_call(policy_model, a, p, x)
                score_exp = scorer(x, pack_actions(t_exp, pc_exp, pm_exp))
                score_base = scorer(x, pack_actions(t_base, pc_base, pm_base))
                score_phase = scorer(x, pack_actions(t_phase, pc_phase, pm_phase))
                margin = torch.sigmoid(score_exp - score_base).detach()
                phase_margin = torch.sigmoid(score_exp - score_phase).detach()

            t_pol, pc_pol, pm_pol = _model_call(policy_model, a, p, x)
            score_pol = scorer(x, pack_actions(t_pol, pc_pol, pm_pol))

            loss_pref = _weighted_mean(-(margin * score_pol), sample_weights)
            loss_phase_pref = _weighted_mean(F.softplus(-(score_pol - score_phase.detach())) * phase_margin, sample_weights)
            day_hours = torch.tensor(b.get("day_hours", pd.Series(np.full(len(b), 24.0))).to_numpy(dtype=np.float32), dtype=torch.float32, device=device_t).unsqueeze(-1)
            w_day = (day_hours / 24.0).squeeze(-1)
            w_mode = (t_exp.squeeze(-1) * w_day)
            loss_cat = _weighted_mean(kl_categorical(pc_exp, pc_pol), w_day)
            loss_mode = _weighted_mean(kl_categorical(pm_exp, pm_pol), w_mode)
            loss_travel = _weighted_mean(((t_pol - t_exp) ** 2).squeeze(-1), w_day)
            rollout_losses = _one_step_rollout_losses(
                b,
                policy_model=policy_model,
                a_cur=a,
                t_cur=t_pol,
                pc_cur=pc_pol,
                ctx_cols=ctx_cols,
                cat_cols=cat_cols,
                mode_cols_all=mode_cols_all,
                home_cat_pos=home_cat_pos,
                device_t=device_t,
            )

            stay_h_true = (1.0 - t_exp) * day_hours
            stay_h_pol = (1.0 - t_pol) * day_hours
            mode_h_true = pm_exp * (t_exp * day_hours)
            mode_h_pol = pm_pol * (t_pol * day_hours)
            cat_h_true = pc_exp * stay_h_true
            cat_h_pol = pc_pol * stay_h_pol
            loss_cat_h = _weighted_mean(((cat_h_pol - cat_h_true) ** 2).mean(dim=-1), w_day)
            loss_mode_h = _weighted_mean(((mode_h_pol - mode_h_true) ** 2).mean(dim=-1), w_day)
            zero_cat_mask = (cat_h_true <= 1e-9).float()
            zero_mode_mask = (mode_h_true <= 1e-9).float()
            loss_zero_cat = _weighted_mean((cat_h_pol * zero_cat_mask).sum(dim=-1), w_day)
            loss_zero_mode = _weighted_mean((mode_h_pol * zero_mode_mask).sum(dim=-1), w_day)
            if 0 <= unk_idx < pm_pol.size(-1):
                unk_true_h = pm_exp[..., unk_idx] * (t_exp * day_hours).squeeze(-1)
                unk_pol_h = pm_pol[..., unk_idx] * (t_pol * day_hours).squeeze(-1)
                loss_unknown_h = _weighted_mean((unk_pol_h - unk_true_h) ** 2, w_day)
            else:
                loss_unknown_h = torch.zeros((), device=device_t)

            pre_policy_cat = torch.zeros((), device=device_t)
            pre_policy_mode = torch.zeros((), device=device_t)
            pre_policy_travel = torch.zeros((), device=device_t)
            pre_policy_w_day = w_day
            pre_policy_w_mode = torch.clamp(w_mode, min=0.0)
            if lambda_pre_policy_cat > 0.0:
                pre_policy_cat = _weighted_mean_if_any(kl_categorical(pc_exp, pc_pol), pre_policy_w_day, device_t)
            if lambda_pre_policy_mode > 0.0:
                pre_policy_mode = _weighted_mean_if_any(kl_categorical(pm_exp, pm_pol), pre_policy_w_mode, device_t)
            if lambda_pre_policy_travel > 0.0:
                pre_policy_travel = _weighted_mean_if_any(((t_pol - t_exp) ** 2).squeeze(-1), pre_policy_w_day, device_t)

            post_phase_response_cat = torch.zeros((), device=device_t)
            post_phase_response_mode = torch.zeros((), device=device_t)
            post_phase_response_travel = torch.zeros((), device=device_t)
            post_phase_w_day = sample_weights * has_prev_phase * w_day
            post_phase_w_mode = sample_weights * has_prev_phase * torch.clamp(w_mode, min=0.0)
            if lambda_post_phase_response_cat > 0.0:
                post_phase_response_cat = _weighted_mean_if_any(
                    F.relu(post_phase_response_cat_margin - kl_categorical(pc_pol, pc_prev.detach())),
                    post_phase_w_day,
                    device_t,
                )
            if lambda_post_phase_response_mode > 0.0:
                post_phase_response_mode = _weighted_mean_if_any(
                    F.relu(post_phase_response_mode_margin - kl_categorical(pm_pol, pm_prev.detach())),
                    post_phase_w_mode,
                    device_t,
                )
            if lambda_post_phase_response_travel > 0.0:
                post_phase_response_travel = _weighted_mean_if_any(
                    F.relu(post_phase_response_travel_margin - (t_pol - t_prev.detach()).abs().squeeze(-1)),
                    post_phase_w_day,
                    device_t,
                )

            phase_wrong_cat = torch.zeros((), device=device_t)
            phase_wrong_mode = torch.zeros((), device=device_t)
            phase_wrong_travel = torch.zeros((), device=device_t)
            if lambda_phase_wrong_cat > 0.0:
                phase_wrong_cat = _weighted_mean_if_any(
                    F.relu(phase_wrong_cat_margin - kl_categorical(pc_pol, pc_phase.detach())) * phase_margin,
                    sample_weights * w_day,
                    device_t,
                )
            if lambda_phase_wrong_mode > 0.0:
                phase_wrong_mode = _weighted_mean_if_any(
                    F.relu(phase_wrong_mode_margin - kl_categorical(pm_pol, pm_phase.detach())) * phase_margin,
                    sample_weights * torch.clamp(w_mode, min=0.0),
                    device_t,
                )
            if lambda_phase_wrong_travel > 0.0:
                phase_wrong_travel = _weighted_mean_if_any(
                    F.relu(phase_wrong_travel_margin - (t_pol - t_phase.detach()).abs().squeeze(-1)) * phase_margin,
                    sample_weights * w_day,
                    device_t,
                )

            anchor_cat = kl_categorical(pc_base.detach(), pc_pol).mean()
            anchor_mode = kl_categorical(pm_base.detach(), pm_pol).mean()
            anchor_travel = F.mse_loss(t_pol, t_base.detach())

            loss = (
                lambda_pref * loss_pref
                + lambda_phase_pref * loss_phase_pref
                + lambda_bc_cat * loss_cat
                + lambda_bc_mode * loss_mode
                + lambda_bc_travel * loss_travel
                + lambda_bc_cat_h * loss_cat_h
                + lambda_bc_mode_h * loss_mode_h
                + lambda_zero_cat * loss_zero_cat
                + lambda_zero_mode * loss_zero_mode
                + lambda_unknown_h * loss_unknown_h
                + lambda_rollout_cat * rollout_losses["cat"]
                + lambda_rollout_mode * rollout_losses["mode"]
                + lambda_rollout_travel * rollout_losses["travel"]
                + lambda_pre_policy_cat * pre_policy_cat
                + lambda_pre_policy_mode * pre_policy_mode
                + lambda_pre_policy_travel * pre_policy_travel
                + lambda_post_phase_response_cat * post_phase_response_cat
                + lambda_post_phase_response_mode * post_phase_response_mode
                + lambda_post_phase_response_travel * post_phase_response_travel
                + lambda_phase_wrong_cat * phase_wrong_cat
                + lambda_phase_wrong_mode * phase_wrong_mode
                + lambda_phase_wrong_travel * phase_wrong_travel
                + lambda_anchor_cat * anchor_cat
                + lambda_anchor_mode * anchor_mode
                + lambda_anchor_travel * anchor_travel
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss)
            ep_phase_pref += float(loss_phase_pref)
            ep_cat_h += float(loss_cat_h)
            ep_mode_h += float(loss_mode_h)
            ep_unknown_h += float(loss_unknown_h)
            ep_zero_cat += float(loss_zero_cat)
            ep_zero_mode += float(loss_zero_mode)
            ep_rollout_cat += float(rollout_losses["cat"])
            ep_rollout_mode += float(rollout_losses["mode"])
            ep_rollout_travel += float(rollout_losses["travel"])
            ep_pre_policy_cat += float(pre_policy_cat)
            ep_pre_policy_mode += float(pre_policy_mode)
            ep_pre_policy_travel += float(pre_policy_travel)
            ep_post_phase_response_cat += float(post_phase_response_cat)
            ep_post_phase_response_mode += float(post_phase_response_mode)
            ep_post_phase_response_travel += float(post_phase_response_travel)
            ep_phase_wrong_cat += float(phase_wrong_cat)
            ep_phase_wrong_mode += float(phase_wrong_mode)
            ep_phase_wrong_travel += float(phase_wrong_travel)

            if (bi % 100) == 0:
                print(
                    f"[PrefAlign] epoch {ep} batch {bi}/{len(batches)} "
                    f"loss={float(loss):.4f} phase_pref={float(loss_phase_pref):.4f} "
                    f"rollout={float(rollout_losses['cat'] + rollout_losses['mode'] + rollout_losses['travel']):.4f}"
                )

        ep_loss /= max(1, len(batches))
        ep_phase_pref /= max(1, len(batches))
        ep_cat_h /= max(1, len(batches))
        ep_mode_h /= max(1, len(batches))
        ep_unknown_h /= max(1, len(batches))
        ep_zero_cat /= max(1, len(batches))
        ep_zero_mode /= max(1, len(batches))
        ep_rollout_cat /= max(1, len(batches))
        ep_rollout_mode /= max(1, len(batches))
        ep_rollout_travel /= max(1, len(batches))
        ep_pre_policy_cat /= max(1, len(batches))
        ep_pre_policy_mode /= max(1, len(batches))
        ep_pre_policy_travel /= max(1, len(batches))
        ep_post_phase_response_cat /= max(1, len(batches))
        ep_post_phase_response_mode /= max(1, len(batches))
        ep_post_phase_response_travel /= max(1, len(batches))
        ep_phase_wrong_cat /= max(1, len(batches))
        ep_phase_wrong_mode /= max(1, len(batches))
        ep_phase_wrong_travel /= max(1, len(batches))
        hist["loss"].append(ep_loss)
        hist["phase_pref_loss"].append(ep_phase_pref)
        hist["bc_cat_h_loss"].append(ep_cat_h)
        hist["bc_mode_h_loss"].append(ep_mode_h)
        hist["bc_unknown_h_loss"].append(ep_unknown_h)
        hist["zero_cat_loss"].append(ep_zero_cat)
        hist["zero_mode_loss"].append(ep_zero_mode)
        hist["rollout_cat_loss"].append(ep_rollout_cat)
        hist["rollout_mode_loss"].append(ep_rollout_mode)
        hist["rollout_travel_loss"].append(ep_rollout_travel)
        hist["pre_policy_cat_loss"].append(ep_pre_policy_cat)
        hist["pre_policy_mode_loss"].append(ep_pre_policy_mode)
        hist["pre_policy_travel_loss"].append(ep_pre_policy_travel)
        hist["post_phase_response_cat_loss"].append(ep_post_phase_response_cat)
        hist["post_phase_response_mode_loss"].append(ep_post_phase_response_mode)
        hist["post_phase_response_travel_loss"].append(ep_post_phase_response_travel)
        hist["phase_wrong_cat_loss"].append(ep_phase_wrong_cat)
        hist["phase_wrong_mode_loss"].append(ep_phase_wrong_mode)
        hist["phase_wrong_travel_loss"].append(ep_phase_wrong_travel)
        if selection_mode == "val" and val_loader is not None and eval_fn is not None and eval_cat_cols is not None and mode_known_idx is not None:
            policy_model.eval()
            val_metrics = eval_fn(policy_model, val_loader, eval_device or device, eval_cat_cols, mode_known_idx)
            val_score = _alignment_selection_score(val_metrics)
            if rollout_eval_fn is not None and rollout_dataset is not None and rollout_eval_weight > 0.0:
                rollout_metrics = rollout_eval_fn(policy_model, rollout_dataset, val_loader, eval_device or device)
                hist.setdefault("val_rollout_metrics", []).append(rollout_metrics)
                val_score += float(rollout_eval_weight) * float(rollout_metrics["daily_share_rollout_score"])
            hist["val_selection_score"].append(val_score)
            hist.setdefault("val_metrics", []).append(val_metrics)
            if val_score < best_score:
                best_score = val_score
                best_epoch = ep
                best_state = copy.deepcopy(policy_model.state_dict())
            policy_model.train()
            print(f"[PrefAlign] epoch {ep} mean loss={ep_loss:.4f} phase_pref={ep_phase_pref:.4f} val_score={val_score:.4f}")
        else:
            hist["val_selection_score"].append(None)
            if selection_mode in ("train_loss", "val") and ep_loss < best_score:
                best_score = ep_loss
                best_epoch = ep
                best_state = copy.deepcopy(policy_model.state_dict())
            print(f"[PrefAlign] epoch {ep} mean loss={ep_loss:.4f} phase_pref={ep_phase_pref:.4f}")
    last_state = copy.deepcopy(policy_model.state_dict())
    if selection_mode == "last":
        best_state = copy.deepcopy(last_state)
        best_epoch = epochs
        best_score = None
    policy_model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch if best_epoch > 0 else epochs
    hist["best_val_selection_score"] = best_score if (best_epoch > 0 and best_score is not None) else None
    return hist, last_state, best_state


def main():
    ap = argparse.ArgumentParser(description="Preference Alignment fine-tuning for Daily-Share policy")
    ap.add_argument("--daily_shares", type=str, required=True)
    ap.add_argument("--columns_json", type=str, required=True)
    ap.add_argument("--agent_index_csv", type=str, default="")
    ap.add_argument("--model_py", type=str, required=True)
    ap.add_argument("--model_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score_epochs", type=int, default=10)
    ap.add_argument("--finetune_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr_score", type=float, default=1e-3)
    ap.add_argument("--lr_policy", type=float, default=5e-5)
    ap.add_argument("--score_hidden", type=int, default=128)
    ap.add_argument("--score_dropout", type=float, default=0.1)
    ap.add_argument("--perturb_scale", type=float, default=0.10)
    ap.add_argument("--alignment_profile", choices=["standard", "factual", "counterfactual"], default="standard")
    ap.add_argument("--selection_mode", choices=["auto", "val", "train_loss", "last"], default="auto")
    ap.add_argument("--phase_rank_weight", type=float, default=1.00)
    ap.add_argument("--pol_phase_weight", type=float, default=0.75)
    ap.add_argument("--pol_neg_weight", type=float, default=0.25)
    ap.add_argument("--phase_neg_weight", type=float, default=0.0)
    ap.add_argument("--enable_phase_neg_ranking", action="store_true",
                    help="Explicitly enable the scorer constraint a_wrongphase > a_disturbnegative. Default off.")
    ap.add_argument("--phase_transition_boost", type=float, default=3.0)
    ap.add_argument("--phase_transition_scale", type=float, default=3.0)
    ap.add_argument("--lambda_pref", type=float, default=0.10)
    ap.add_argument("--lambda_phase_pref", type=float, default=0.50)
    ap.add_argument("--lambda_bc_cat", type=float, default=1.0)
    ap.add_argument("--lambda_bc_mode", type=float, default=1.0)
    ap.add_argument("--lambda_bc_travel", type=float, default=1.0)
    ap.add_argument("--lambda_bc_cat_h", type=float, default=0.25)
    ap.add_argument("--lambda_bc_mode_h", type=float, default=0.40)
    ap.add_argument("--lambda_zero_cat", type=float, default=0.01)
    ap.add_argument("--lambda_zero_mode", type=float, default=0.05)
    ap.add_argument("--lambda_unknown_h", type=float, default=0.15)
    ap.add_argument("--lambda_anchor_cat", type=float, default=0.15)
    ap.add_argument("--lambda_anchor_mode", type=float, default=0.15)
    ap.add_argument("--lambda_anchor_travel", type=float, default=0.15)
    ap.add_argument("--lambda_rollout_cat", type=float, default=0.0)
    ap.add_argument("--lambda_rollout_mode", type=float, default=0.0)
    ap.add_argument("--lambda_rollout_travel", type=float, default=0.0)
    ap.add_argument("--rollout_eval_weight", type=float, default=0.0)
    ap.add_argument("--lambda_pre_policy_cat", type=float, default=0.0)
    ap.add_argument("--lambda_pre_policy_mode", type=float, default=0.0)
    ap.add_argument("--lambda_pre_policy_travel", type=float, default=0.0)
    ap.add_argument("--lambda_post_phase_response_cat", type=float, default=0.0)
    ap.add_argument("--lambda_post_phase_response_mode", type=float, default=0.0)
    ap.add_argument("--lambda_post_phase_response_travel", type=float, default=0.0)
    ap.add_argument("--post_phase_response_cat_margin", type=float, default=0.20)
    ap.add_argument("--post_phase_response_mode_margin", type=float, default=0.15)
    ap.add_argument("--post_phase_response_travel_margin", type=float, default=0.08)
    ap.add_argument("--lambda_phase_wrong_cat", type=float, default=0.0)
    ap.add_argument("--lambda_phase_wrong_mode", type=float, default=0.0)
    ap.add_argument("--lambda_phase_wrong_travel", type=float, default=0.0)
    ap.add_argument("--phase_wrong_cat_margin", type=float, default=0.20)
    ap.add_argument("--phase_wrong_mode_margin", type=float, default=0.15)
    ap.add_argument("--phase_wrong_travel_margin", type=float, default=0.08)
    ap.add_argument("--diagnostic_max_rows", type=int, default=4096)
    ap.add_argument("--support_reference_rows", type=int, default=4096)
    args = ap.parse_args()
    args = _apply_profile_defaults(args)
    if not args.enable_phase_neg_ranking and abs(float(args.phase_neg_weight)) > 1e-12:
        print(
            f"[info] phase_neg_weight={args.phase_neg_weight} was provided but "
            "a_wrongphase > a_disturbnegative is disabled by default; forcing phase_neg_weight=0.0. "
            "Pass --enable_phase_neg_ranking to opt in."
        )
        args.phase_neg_weight = 0.0
    args.selection_mode = _choose_selection_mode(args.alignment_profile, args.selection_mode)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = json.loads(Path(args.columns_json).read_text())
    ctx_cols = cols["ctx_cols"]
    cat_cols = cols["cat_cols"]
    mode_cols_all = cols["mode_cols_all"]
    mode_cols_known = cols.get("mode_cols_known", [c for c in mode_cols_all if c != mode_cols_all[-1]])
    known_idx, unk_idx = _build_known_unknown_idx(mode_cols_all, mode_cols_known)

    agent_index_csv = Path(args.agent_index_csv) if args.agent_index_csv else Path(args.columns_json).with_name("agent_index.csv")
    if not agent_index_csv.exists():
        raise FileNotFoundError(f"agent_index.csv not found: {agent_index_csv}. Pass --agent_index_csv explicitly.")
    aid_df = pd.read_csv(agent_index_csv)
    aid_map = dict(zip(aid_df["agent_id"].astype(str), aid_df["agent_idx"].astype(int)))

    df = pd.read_csv(args.daily_shares)
    df["agent_id"] = df["agent_id"].astype(str)
    df[cat_cols] = (df[cat_cols].clip(lower=0) + EPS)
    df[cat_cols] = df[cat_cols].div(df[cat_cols].sum(axis=1), axis=0)
    df[mode_cols_all] = (df[mode_cols_all].clip(lower=0) + EPS)
    df[mode_cols_all] = df[mode_cols_all].div(df[mode_cols_all].sum(axis=1), axis=0)
    df["travel_frac"] = df["travel_frac"].clip(lower=0, upper=1)
    if not set(df["agent_id"].unique()).issubset(set(aid_map.keys())):
        missing = sorted(set(df["agent_id"].unique()) - set(aid_map.keys()))
        raise ValueError(f"{len(missing)} agent_id not found in agent_index.csv (first few: {missing[:5]})")
    df["agent_idx"] = df["agent_id"].map(aid_map).astype(int)
    df["epi_phase"] = pd.to_numeric(df["epi_phase"], errors="coerce").fillna(0).astype(int)
    if "days_since_phase" not in df.columns:
        df["days_since_phase"] = 0.0
    df["days_since_phase"] = pd.to_numeric(df["days_since_phase"], errors="coerce").fillna(0.0)
    df = _attach_phase_boundary_context(df)

    model_mod = _import_from_path(Path(args.model_py), "dsl_model_mod")
    PolicyClass = getattr(model_mod, "PreferenceNet")

    ckpt = torch.load(args.model_ckpt, map_location="cpu", weights_only=False)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
        cfg = ckpt.get("config", {})
        n_agents = int(ckpt.get("n_agents", len(aid_map)))
        n_phases = int(ckpt.get("n_phases", int(df["epi_phase"].max()) + 1))
        ctx_dim = int(ckpt.get("ctx_dim", len(ctx_cols)))
        n_cat = int(ckpt.get("n_cat", len(cat_cols)))
        n_mode = int(ckpt.get("n_mode", len(mode_cols_all)))
    else:
        state = ckpt
        cfg = {}
        n_agents = len(aid_map)
        n_phases = int(df["epi_phase"].max()) + 1
        ctx_dim = len(ctx_cols)
        n_cat = len(cat_cols)
        n_mode = len(mode_cols_all)

    base_kwargs = {
        "n_agents": n_agents,
        "n_phases": n_phases,
        "ctx_dim": ctx_dim,
        "n_cat": n_cat,
        "n_mode": n_mode,
        "mode_known_idx": known_idx,
        "unk_mode_idx": unk_idx,
    }
    init_kwargs = _resolve_policy_init_kwargs(PolicyClass, cfg, base_kwargs)
    # Handle both "cuda"/"cpu" and GPU ID strings like "0", "1"
    device_str = args.device
    if device_str.isdigit():
        device_str = f"cuda:{device_str}" if torch.cuda.is_available() else "cpu"
    args.device = device_str  # normalize for all downstream uses
    device = torch.device(device_str)

    policy_model = PolicyClass(**init_kwargs)
    policy_model._init_kwargs = init_kwargs
    missing, unexpected = policy_model.load_state_dict(state, strict=False)
    if missing:
        print(f"[info] load_state_dict: {len(missing)} missing key(s) (e.g. new FiLM layers) — initialised randomly: {missing[:4]}")
    if unexpected:
        print(f"[info] load_state_dict: {len(unexpected)} unexpected key(s) ignored: {unexpected[:4]}")
    # Re-enable phase-gated lag suppression if the checkpoint has it
    if hasattr(policy_model, "set_lag_feature_indices"):
        lag_indices = [i for i, c in enumerate(ctx_cols) if c.startswith("prev_") or c.startswith("prev7_")]
        if lag_indices:
            policy_model.set_lag_feature_indices(lag_indices)
            # Load gate weights from checkpoint if present
            if "phase_lag_gate.weight" in state:
                policy_model.phase_lag_gate.weight.data = state["phase_lag_gate.weight"]
                policy_model.phase_lag_gate.bias.data = state["phase_lag_gate.bias"]
    policy_model.to(device)
    policy_model.eval()

    train_loader = None
    val_loader = None
    test_loader = None
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    full_df = df.copy()
    if (
        hasattr(model_mod, "TrainConfig")
        and hasattr(model_mod, "make_loaders")
        and hasattr(model_mod, "evaluate")
    ):
        feature_meta_path = Path(args.daily_shares).with_name("feature_meta.json")
        ds_eval = DailyShareDataset(Path(args.daily_shares), feature_meta_path if feature_meta_path.exists() else None)
        train_cfg_fields = set(model_mod.TrainConfig.__dataclass_fields__.keys())
        cfg_for_split = {k: v for k, v in cfg.items() if k in train_cfg_fields}
        cfg_for_split.setdefault("batch_size", args.batch_size)
        cfg_for_split.setdefault("num_workers", 0)
        cfg_for_split.setdefault("device", args.device)
        split_cfg = model_mod.TrainConfig(**cfg_for_split)
        train_loader, val_loader, test_loader = model_mod.make_loaders(ds_eval, split_cfg, out_dir)
        train_df = _subset_dataframe_from_loader(df, train_loader)
        val_df = _subset_dataframe_from_loader(df, val_loader)
        test_df = _subset_dataframe_from_loader(df, test_loader)
    else:
        raise RuntimeError("Preference alignment now requires TrainConfig/make_loaders/evaluate from the model module.")

    if train_df.empty:
        raise RuntimeError(
            f"Preference alignment requires non-empty train split, got train={len(train_df)}."
        )

    has_val_loader = _loader_has_data(val_loader)

    if args.alignment_profile == "counterfactual":
        score_df = _attach_one_step_rollout_targets(full_df, ctx_cols, cat_cols, mode_cols_all)
        finetune_df = score_df.copy()
        rollout_dataset = None
        effective_val_loader = None
    else:
        score_df = _attach_one_step_rollout_targets(train_df, ctx_cols, cat_cols, mode_cols_all)
        finetune_df = score_df.copy()
        rollout_dataset = ds_eval if (args.alignment_profile == "factual" and has_val_loader) else None
        effective_val_loader = val_loader if has_val_loader else None

    effective_selection_mode = args.selection_mode
    if args.alignment_profile == "factual" and effective_val_loader is None and effective_selection_mode == "val":
        effective_selection_mode = "train_loss"
        print("[info] Factual alignment detected no validation split; falling back to train_loss checkpoint selection.")

    feature_meta_path = Path(args.daily_shares).with_name("feature_meta.json")
    home_cat_pos = _resolve_home_cat_position(feature_meta_path, cat_cols)

    scorer, score_hist = train_preference_scorer(
        policy_model=policy_model,
        df=score_df,
        ctx_cols=ctx_cols,
        cat_cols=cat_cols,
        mode_cols_all=mode_cols_all,
        device=args.device,
        epochs=args.score_epochs,
        batch_size=args.batch_size,
        lr=args.lr_score,
        h_dim=args.score_hidden,
        dropout=args.score_dropout,
        perturb_scale=args.perturb_scale,
        phase_rank_weight=args.phase_rank_weight,
        pol_phase_weight=args.pol_phase_weight,
        pol_neg_weight=args.pol_neg_weight,
        phase_neg_weight=args.phase_neg_weight,
        phase_transition_boost=args.phase_transition_boost,
        phase_transition_scale=args.phase_transition_scale,
    )
    torch.save({"model_state": scorer.state_dict(), "config": scorer.cfg.__dict__}, out_dir / "preference_model.pt")

    align_hist = {
        "method": "phase_aware_preference_alignment",
        "profile": args.alignment_profile,
        "score": score_hist,
        "config": {
            "selection_mode": effective_selection_mode,
            "phase_rank_weight": args.phase_rank_weight,
            "pol_phase_weight": args.pol_phase_weight,
            "pol_neg_weight": args.pol_neg_weight,
            "phase_neg_weight": args.phase_neg_weight,
            "phase_neg_ranking_enabled": bool(args.enable_phase_neg_ranking),
            "phase_transition_boost": args.phase_transition_boost,
            "phase_transition_scale": args.phase_transition_scale,
            "lambda_pref": args.lambda_pref,
            "lambda_phase_pref": args.lambda_phase_pref,
            "lambda_bc_cat": args.lambda_bc_cat,
            "lambda_bc_mode": args.lambda_bc_mode,
            "lambda_bc_travel": args.lambda_bc_travel,
            "lambda_bc_cat_h": args.lambda_bc_cat_h,
            "lambda_bc_mode_h": args.lambda_bc_mode_h,
            "lambda_unknown_h": args.lambda_unknown_h,
            "lambda_zero_cat": args.lambda_zero_cat,
            "lambda_zero_mode": args.lambda_zero_mode,
            "lambda_anchor_cat": args.lambda_anchor_cat,
            "lambda_anchor_mode": args.lambda_anchor_mode,
            "lambda_anchor_travel": args.lambda_anchor_travel,
            "lambda_rollout_cat": args.lambda_rollout_cat,
            "lambda_rollout_mode": args.lambda_rollout_mode,
            "lambda_rollout_travel": args.lambda_rollout_travel,
            "rollout_eval_weight": args.rollout_eval_weight,
            "lambda_pre_policy_cat": args.lambda_pre_policy_cat,
            "lambda_pre_policy_mode": args.lambda_pre_policy_mode,
            "lambda_pre_policy_travel": args.lambda_pre_policy_travel,
            "lambda_post_phase_response_cat": args.lambda_post_phase_response_cat,
            "lambda_post_phase_response_mode": args.lambda_post_phase_response_mode,
            "lambda_post_phase_response_travel": args.lambda_post_phase_response_travel,
            "post_phase_response_cat_margin": args.post_phase_response_cat_margin,
            "post_phase_response_mode_margin": args.post_phase_response_mode_margin,
            "post_phase_response_travel_margin": args.post_phase_response_travel_margin,
            "lambda_phase_wrong_cat": args.lambda_phase_wrong_cat,
            "lambda_phase_wrong_mode": args.lambda_phase_wrong_mode,
            "lambda_phase_wrong_travel": args.lambda_phase_wrong_travel,
            "phase_wrong_cat_margin": args.phase_wrong_cat_margin,
            "phase_wrong_mode_margin": args.phase_wrong_mode_margin,
            "phase_wrong_travel_margin": args.phase_wrong_travel_margin,
            "train_rows": int(len(score_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "diagnostic_max_rows": int(args.diagnostic_max_rows),
            "support_reference_rows": int(args.support_reference_rows),
        },
    }
    best_state_dict = copy.deepcopy(policy_model.state_dict())
    last_state_dict = copy.deepcopy(policy_model.state_dict())
    if args.finetune_epochs > 0:
        ft_hist, last_state_dict, best_state_dict = preference_align_finetune(
            policy_model=policy_model,
            df=finetune_df,
            ctx_cols=ctx_cols,
            cat_cols=cat_cols,
            mode_cols_all=mode_cols_all,
            scorer=scorer,
            device=args.device,
            epochs=args.finetune_epochs,
            batch_size=min(2048, args.batch_size),
            lr=args.lr_policy,
            lambda_pref=args.lambda_pref,
            lambda_phase_pref=args.lambda_phase_pref,
            lambda_bc_cat=args.lambda_bc_cat,
            lambda_bc_mode=args.lambda_bc_mode,
            lambda_bc_travel=args.lambda_bc_travel,
            lambda_bc_cat_h=args.lambda_bc_cat_h,
            lambda_bc_mode_h=args.lambda_bc_mode_h,
            lambda_zero_cat=args.lambda_zero_cat,
            lambda_zero_mode=args.lambda_zero_mode,
            lambda_unknown_h=args.lambda_unknown_h,
            lambda_rollout_cat=args.lambda_rollout_cat,
            lambda_rollout_mode=args.lambda_rollout_mode,
            lambda_rollout_travel=args.lambda_rollout_travel,
            lambda_pre_policy_cat=args.lambda_pre_policy_cat,
            lambda_pre_policy_mode=args.lambda_pre_policy_mode,
            lambda_pre_policy_travel=args.lambda_pre_policy_travel,
            lambda_post_phase_response_cat=args.lambda_post_phase_response_cat,
            lambda_post_phase_response_mode=args.lambda_post_phase_response_mode,
            lambda_post_phase_response_travel=args.lambda_post_phase_response_travel,
            post_phase_response_cat_margin=args.post_phase_response_cat_margin,
            post_phase_response_mode_margin=args.post_phase_response_mode_margin,
            post_phase_response_travel_margin=args.post_phase_response_travel_margin,
            lambda_phase_wrong_cat=args.lambda_phase_wrong_cat,
            lambda_phase_wrong_mode=args.lambda_phase_wrong_mode,
            lambda_phase_wrong_travel=args.lambda_phase_wrong_travel,
            phase_wrong_cat_margin=args.phase_wrong_cat_margin,
            phase_wrong_mode_margin=args.phase_wrong_mode_margin,
            phase_wrong_travel_margin=args.phase_wrong_travel_margin,
            lambda_anchor_cat=args.lambda_anchor_cat,
            lambda_anchor_mode=args.lambda_anchor_mode,
            lambda_anchor_travel=args.lambda_anchor_travel,
            val_loader=effective_val_loader,
            eval_fn=getattr(model_mod, "evaluate", None),
            eval_cat_cols=cat_cols,
            mode_known_idx=known_idx,
            eval_device=args.device,
            phase_transition_boost=args.phase_transition_boost,
            phase_transition_scale=args.phase_transition_scale,
            selection_mode=effective_selection_mode,
            rollout_eval_fn=getattr(model_mod, "evaluate_rollout_fidelity", None),
            rollout_dataset=rollout_dataset,
            rollout_eval_weight=args.rollout_eval_weight,
            home_cat_pos=home_cat_pos,
        )
        align_hist["finetune"] = ft_hist
    else:
        best_state_dict = copy.deepcopy(policy_model.state_dict())
        last_state_dict = copy.deepcopy(policy_model.state_dict())

    policy_model.load_state_dict(best_state_dict)
    torch.save({"model_state": best_state_dict, "config": cfg}, out_dir / "best_alignment_model.pt")

    bundle = {
        "model_state": best_state_dict,
        "n_agents": n_agents,
        "n_phases": n_phases,
        "ctx_dim": ctx_dim,
        "n_cat": n_cat,
        "n_mode": n_mode,
        "config": cfg,
    }
    last_bundle = dict(bundle)
    last_bundle["model_state"] = last_state_dict
    torch.save(last_bundle, out_dir / "last_model.pt")
    torch.save(bundle, out_dir / "model.pt")

    mode_temperature = 1.0
    cat_temperature = 1.0

    if (
        feature_meta_path.exists()
        and test_loader is not None
        and hasattr(model_mod, "evaluate")
        and hasattr(model_mod, "find_best_mode_temperature")
        and hasattr(model_mod, "find_best_cat_temperature")
    ):
        policy_model.eval()
        if effective_val_loader is not None and _loader_has_data(effective_val_loader):
            mode_temperature = float(model_mod.find_best_mode_temperature(policy_model, effective_val_loader, args.device, known_idx))
            cat_temperature = float(model_mod.find_best_cat_temperature(policy_model, effective_val_loader, args.device, known_idx))
        else:
            mode_temperature = 1.0
            cat_temperature = 1.0
        test_metrics = model_mod.evaluate(
            policy_model,
            test_loader,
            args.device,
            cat_cols,
            known_idx,
            mode_temperature=mode_temperature,
            cat_temperature=cat_temperature,
        )
        eval_payload = {
            "checkpoint": "alignment",
            "path": str(out_dir / "model.pt"),
            "mode_temperature": mode_temperature,
            "cat_temperature": cat_temperature,
            "test_metrics": test_metrics,
        }
        (out_dir / "test_metrics_alignment_eval.json").write_text(json.dumps(eval_payload, indent=2))
        (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
        (out_dir / "calibration.json").write_text(json.dumps({
            "mode_temperature": mode_temperature,
            "cat_temperature": cat_temperature,
            "checkpoint": "alignment",
        }, indent=2))
        print(f"[TEST:alignment] (temp-calibrated) {test_metrics}")

    if feature_meta_path.exists() and hasattr(model_mod, "emit_predictions"):
        ds = DailyShareDataset(Path(args.daily_shares), feature_meta_path)
        policy_model.eval()
        model_mod.emit_predictions(
            policy_model,
            ds,
            args.device,
            out_dir,
            mode_temperature=mode_temperature,
            cat_temperature=cat_temperature,
        )

    diag_eval_df = test_df if not test_df.empty else val_df
    if not train_df.empty and not diag_eval_df.empty:
        diag_summary, diag_detail = evaluate_preference_diagnostics(
            policy_model=policy_model,
            scorer=scorer,
            train_df=train_df,
            eval_df=diag_eval_df,
            ctx_cols=ctx_cols,
            cat_cols=cat_cols,
            mode_cols_all=mode_cols_all,
            device=args.device,
            max_eval_rows=args.diagnostic_max_rows,
            max_support_rows=args.support_reference_rows,
            seed=int(cfg.get("seed", 1234)) if isinstance(cfg, dict) else 1234,
        )
        align_hist["diagnostics"] = diag_summary
        (out_dir / "preference_diagnostics.json").write_text(json.dumps(diag_summary, indent=2))
        if not diag_detail.empty:
            diag_detail.to_csv(out_dir / "preference_diagnostics_detail.csv", index=False, float_format="%.6f")

    (out_dir / "alignment_history.json").write_text(json.dumps(align_hist, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
