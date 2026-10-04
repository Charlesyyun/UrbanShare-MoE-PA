from __future__ import annotations
import argparse, os, sys, json, math, time, queue
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager, Process
from tqdm.auto import tqdm
import json
import numpy as np
import pandas as pd
from daily_share_model import PreferenceNet as PreferenceNetMoE
from daily_share_model_baseline import PreferenceNet as PreferenceNetBase
import torch


DEFAULT_PHASE_POI_RULES: Dict[str, Dict[str, List[float]]] = {
    "0": {
        "allowed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    },
    "1": {
        "allowed": [0, 3, 5, 7, 8, 11],
        "weight": [0.8, 0.4, 0.45, 0.5, 0.9, 1.4],
    },
    "2": {
        "allowed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "weight": [1.0, 0.9, 0.85, 0.95, 0.8, 0.95, 0.85, 1.0, 1.0, 0.55, 0.95, 1.1],
    },
    "3": {
        "allowed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    },
}


def _load_phase_poi_rules(path: Optional[Path]) -> Dict[int, Dict[str, List[float]]]:
    raw = DEFAULT_PHASE_POI_RULES
    if path is not None and Path(path).exists():
        raw = json.loads(Path(path).read_text())

    rules: Dict[int, Dict[str, List[float]]] = {}
    for phase_key, spec in raw.items():
        phase_id = int(phase_key)
        allowed = [int(x) for x in spec.get("allowed", [])]
        weights = [float(x) for x in spec.get("weight", [])]
        if weights and len(weights) != len(allowed):
            raise ValueError(
                f"phase_poi_rules for phase {phase_id} has mismatched allowed/weight lengths: "
                f"{len(allowed)} vs {len(weights)}"
            )
        if not weights:
            weights = [1.0] * len(allowed)
        rules[phase_id] = {"allowed": allowed, "weight": weights}
    return rules


def _build_phase_poi_masks(
    cat_id_order: List[int],
    home_cat_idx: int,
    constraint_mode: str,
    rules_path: Optional[Path],
) -> Tuple[Dict[int, np.ndarray], Optional[int]]:
    if constraint_mode == "off":
        return {}, cat_id_order.index(home_cat_idx) if home_cat_idx in cat_id_order else None

    raw_rules = _load_phase_poi_rules(rules_path)
    masks: Dict[int, np.ndarray] = {}
    for ph, rule in raw_rules.items():
        mask = np.ones(len(cat_id_order), dtype=float) if constraint_mode == "soft" else np.zeros(len(cat_id_order), dtype=float)
        for cid, w in zip(rule.get("allowed", []), rule.get("weight", [])):
            if cid not in cat_id_order:
                continue
            idx = cat_id_order.index(cid)
            mask[idx] = 1.0 if constraint_mode == "hard" else float(w)
        masks[ph] = mask

    home_pos = cat_id_order.index(home_cat_idx) if home_cat_idx in cat_id_order else None
    return masks, home_pos


def _apply_phase_poi_constraint(
    pc_row: np.ndarray,
    phase_id: int,
    phase_poi_mask: Dict[int, np.ndarray],
    home_pos: Optional[int],
) -> np.ndarray:
    if phase_id not in phase_poi_mask:
        return pc_row

    adj = pc_row * phase_poi_mask[phase_id]
    ssum = adj.sum()
    if ssum > 1e-9:
        return adj / ssum

    if home_pos is not None and 0 <= home_pos < len(adj):
        adj[:] = 0.0
        adj[home_pos] = 1.0
    return adj


def _safe_np_dist(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s <= 1e-12:
        if arr.size == 0:
            return arr
        return np.full(arr.shape, 1.0 / arr.size, dtype=float)
    return arr / s


def _blend_same_phase_with_observed(
    *,
    pc_row: np.ndarray,
    pm_row: np.ndarray,
    t_val: float,
    obs_row: pd.Series,
    cf_phase: int,
    cat_cols: List[str],
    mode_cols_all: List[str],
    blend: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    alpha = float(np.clip(blend, 0.0, 1.0))
    if alpha <= 0.0:
        return pc_row, pm_row, float(t_val)
    try:
        obs_phase = int(obs_row.get("obs_epi_phase", obs_row.get("epi_phase", cf_phase)))
    except Exception:
        obs_phase = int(cf_phase)
    if int(cf_phase) != obs_phase:
        return pc_row, pm_row, float(t_val)

    obs_t = float(np.clip(float(obs_row.get("travel_frac", t_val)), 0.0, 1.0))
    obs_pc = _safe_np_dist(np.array([float(obs_row.get(c, 0.0)) for c in cat_cols], dtype=float))
    obs_pm = _safe_np_dist(np.array([float(obs_row.get(c, 0.0)) for c in mode_cols_all], dtype=float))
    pc_mix = _safe_np_dist(alpha * obs_pc + (1.0 - alpha) * np.asarray(pc_row, dtype=float))
    pm_mix = _safe_np_dist(alpha * obs_pm + (1.0 - alpha) * np.asarray(pm_row, dtype=float))
    t_mix = float(alpha * obs_t + (1.0 - alpha) * float(t_val))
    return pc_mix, pm_mix, t_mix


def _observed_action_from_row(
    row: pd.Series,
    cat_cols: List[str],
    mode_cols_all: List[str],
) -> Tuple[np.ndarray, np.ndarray, float]:
    t_obs = float(np.clip(float(row.get("travel_frac", 0.0)), 0.0, 1.0))
    pc_obs = _safe_np_dist(np.array([float(row.get(c, 0.0)) for c in cat_cols], dtype=float))
    pm_obs = _safe_np_dist(np.array([float(row.get(c, 0.0)) for c in mode_cols_all], dtype=float))
    return pc_obs, pm_obs, t_obs

def _write_cf_daily_shares(
    out_dir: Path,
    feature_meta_path: Path,
    npi_calendar_path: Optional[Path] = None,
    observed_daily_shares_path: Optional[Path] = None,
    outfile_name: str = "daily_shares_cf.csv",
):
    meta = json.loads(Path(feature_meta_path).read_text())
    cat_ids = [int(x) for x in meta["cat_ids"]]
    mode_ids_all = [int(x) for x in meta["mode_ids"]]
    unknown_mode_idx = int(meta["unknown_mode_idx"])

    poi_fp  = out_dir / "per_agent_daily_poi_hours_cf.csv"
    mode_fp = out_dir / "per_agent_daily_mode_hours_cf.csv"
    if not poi_fp.exists() or not mode_fp.exists():
        raise FileNotFoundError("Expected CF per-day hours missing. "
                                "Make sure per_agent_daily_*_cf.csv are written before this step.")

    poi = pd.read_csv(poi_fp)
    mode = pd.read_csv(mode_fp)

    for df in (poi, mode):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        else:
            raise ValueError("CF per-day files must include 'date' column")

    key = ["agent_id", "date", "epi_phase"]
    pick_cols = lambda df, cols: [c for c in cols if c in df.columns]
    base_cols = list(dict.fromkeys(key + pick_cols(poi, ["day_hours"])))  # stable order

    for k in cat_ids:
        col = f"cat_{k}h"
        if col not in poi.columns:
            poi[col] = 0.0
    for m in mode_ids_all:
        col = f"mode_{m}h"
        if col not in mode.columns:
            mode[col] = 0.0

    poi_agg = poi.groupby(key, dropna=False, as_index=False).agg({**{f"cat_{k}h": "sum" for k in cat_ids},
                                                                  **{c: "first" for c in pick_cols(poi, ["day_hours"])}})
    mode_agg = mode.groupby(key, dropna=False, as_index=False).agg({**{f"mode_{m}h": "sum" for m in mode_ids_all},
                                                                    **{c: "first" for c in pick_cols(mode, ["day_hours"])}})

    df = pd.merge(poi_agg, mode_agg, on=key, how="outer")
    df["day_hours"] = df["day_hours_x"].fillna(df["day_hours_y"]).fillna(24.0)
    df = df.drop(columns=[c for c in ["day_hours_x","day_hours_y"] if c in df.columns])

    df["total_stay_h"]   = df[[f"cat_{k}h"  for k in cat_ids]].sum(axis=1)
    df["total_travel_h"] = df[[f"mode_{m}h" for m in mode_ids_all]].sum(axis=1)

    dh_missing = df["day_hours"].isna() | (df["day_hours"] <= 0)
    df.loc[dh_missing, "day_hours"] = (df["total_stay_h"] + df["total_travel_h"]).where(
        (df["total_stay_h"] + df["total_travel_h"]) > 0, other=24.0
    )

    eps = 1e-12
    df["travel_frac"] = df["total_travel_h"] / (df["day_hours"] + eps)

    for k in cat_ids:
        df[f"cat_{k}"] = df[f"cat_{k}h"] / (df["day_hours"] + eps)

    for m in mode_ids_all:
        df[f"mode_{m}"] = np.where(df["total_travel_h"] > 0,
                                   df[f"mode_{m}h"] / (df["total_travel_h"] + eps),
                                   0.0)

    if npi_calendar_path is not None and Path(npi_calendar_path).exists():
        cal = pd.read_csv(npi_calendar_path)
        cal["date"] = pd.to_datetime(cal["date"]).dt.date
        keep_cal = [c for c in ["date","phase_id","days_since_phase","days_since_npi",
                                "days_since_phase1","days_since_phase2","npi_active"] if c in cal.columns]
        df = df.merge(cal[keep_cal], on="date", how="left")
        if "phase_id" in df.columns:
            df["epi_phase"] = df["phase_id"]

    dts = pd.to_datetime(df["date"])
    wkd = dts.dt.weekday  # Mon=0
    for w in range(7):
        df[f"wk_{w}"] = (wkd == w).astype(int)

    if observed_daily_shares_path is not None and Path(observed_daily_shares_path).exists():
        obs = pd.read_csv(observed_daily_shares_path, usecols=lambda c: c in (
            ["agent_id","gender","age_0","age_1","age_2","age_3","age_4"]
        ), dtype={"agent_id": str})
        obs = obs.drop_duplicates(subset=["agent_id"])
        df["agent_id"] = df["agent_id"].astype(str)
        df = df.merge(obs, on="agent_id", how="left")

    core_cols = (["agent_id","date","epi_phase","day_hours","travel_frac"]
                 + [f"cat_{k}" for k in cat_ids]
                 + [f"mode_{m}" for m in mode_ids_all])
    extras = [c for c in ["days_since_phase","days_since_npi","days_since_phase1","days_since_phase2","npi_active"]
              if c in df.columns]
    wk_cols = [f"wk_{w}" for w in range(7)]
    demo_cols = [c for c in ["age_0","age_1","age_2","age_3","age_4","gender"] if c in df.columns]

    out_cols = core_cols + extras + wk_cols + demo_cols

    out = df[out_cols].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(out_dir / outfile_name, index=False)

    cat_sum = out[[f"cat_{k}" for k in cat_ids]].sum(axis=1).values
    ok_rows = np.mean(np.isfinite(cat_sum + out["travel_frac"].values))
    print(f"[daily_shares_cf] wrote {outfile_name} with {len(out):,} rows "
          f"(finite share rows: {ok_rows*100:.1f}%).")


def _compute_pop_phase_24h(out_dir: Path, feature_meta_path: Path, daily_shares_path: Optional[Path] = None):
    import json
    meta = json.loads(Path(feature_meta_path).read_text())
    cat_ids = [int(x) for x in meta["cat_ids"]]
    unknown_mode_idx = int(meta["unknown_mode_idx"])
    mode_ids_all = [int(x) for x in meta["mode_ids"]]
    mode_ids = [m for m in mode_ids_all if m != unknown_mode_idx]

    poi_df  = pd.read_csv(out_dir / "per_agent_daily_poi_hours_cf.csv")
    mode_df = pd.read_csv(out_dir / "per_agent_daily_mode_hours_cf.csv")

    def scale_block(df, cols):
        scale = (24.0 / df["day_hours"].replace(0, np.nan)).fillna(0.0)
        out = df[["epi_phase"]].copy()
        for c in cols:
            out[c.replace("h", "h_24")] = df[c] * scale
        return out

    poi_cols  = [f"cat_{k}h" for k in cat_ids]
    mode_cols = [f"mode_{m}h" for m in mode_ids]

    poi_scaled  = scale_block(poi_df,  poi_cols)
    mode_scaled = scale_block(mode_df, mode_cols)

    pop_poi  = poi_scaled.groupby("epi_phase", dropna=False)[[c for c in poi_scaled.columns if c.endswith("h_24")]].mean().reset_index()
    pop_mode = mode_scaled.groupby("epi_phase", dropna=False)[[c for c in mode_scaled.columns if c.endswith("h_24")]].mean().reset_index()

    pop = pd.merge(pop_poi, pop_mode, on="epi_phase", how="outer").fillna(0.0)
    pop["total_stay_h_24"]   = pop[[f"cat_{k}h_24"  for k in cat_ids]].sum(axis=1)
    pop["total_travel_h_24"] = pop[[f"mode_{m}h_24" for m in mode_ids]].sum(axis=1)
    pop.to_csv(out_dir / "population_phase_summary_24h_cf.csv", index=False)

    if daily_shares_path is not None:
        ds = pd.read_csv(daily_shares_path)
        for k in cat_ids:
            ds[f"cat_{k}h_24"] = 24.0 * ds[f"cat_{k}"]
        for m in mode_ids:
            ds[f"mode_{m}h_24"] = 24.0 * (ds["travel_frac"] * ds[f"mode_{m}"])

        obs_cols = [f"cat_{k}h_24" for k in cat_ids] + [f"mode_{m}h_24" for m in mode_ids]
        obs = ds.groupby("epi_phase", dropna=False)[obs_cols].mean().reset_index()
        obs["total_stay_h_24"]   = obs[[f"cat_{k}h_24"  for k in cat_ids]].sum(axis=1)
        obs["total_travel_h_24"] = obs[[f"mode_{m}h_24" for m in mode_ids]].sum(axis=1)
        obs.to_csv(out_dir / "population_phase_summary_24h_obs.csv", index=False)

        delta = pd.merge(pop, obs, on="epi_phase", how="outer", suffixes=("_cf", "_obs")).fillna(0.0)
        for c in [f"cat_{k}h_24" for k in cat_ids] + [f"mode_{m}h_24" for m in mode_ids] + ["total_stay_h_24","total_travel_h_24"]:
            delta[f"{c}_delta"] = delta[f"{c}_cf"] - delta[f"{c}_obs"]
        delta.to_csv(out_dir / "population_phase_summary_24h_delta.csv", index=False)

def _read_one_for_merge(d: Path, fname: str):
    p = d / fname
    if not p.exists():
        return None
    return pd.read_csv(p)

def _exists(p: str|Path) -> Path:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p

def load_calendar(npi_csv: Path) -> pd.DataFrame:
    cal = pd.read_csv(_exists(npi_csv))
    if "date" not in cal: raise ValueError("Counterfactual NPI CSV must contain 'date'")
    if "epi_phase" not in cal and "phase_id" in cal:
        cal = cal.rename(columns={"phase_id": "epi_phase"})
    cal["date"] = pd.to_datetime(cal["date"]).dt.date
    cal = cal.sort_values("date").reset_index(drop=True)

    if "days_since_npi" not in cal:
        start = pd.to_datetime(cal["date"].iloc[0])
        cal["days_since_npi"] = (pd.to_datetime(cal["date"]) - start).dt.days.astype(int)
    if "days_since_phase" not in cal:
        cal["days_since_phase"] = 0
        for _, g in cal.groupby("epi_phase"):
            d0 = pd.to_datetime(g["date"].min())
            cal.loc[g.index, "days_since_phase"] = (pd.to_datetime(g["date"]) - d0).dt.days.astype(int)
    return cal[["date","epi_phase","days_since_phase","days_since_npi"]]

def load_feature_meta(path: Path) -> Dict:
    meta = json.loads(Path(path).read_text())
    for k in ["cat_ids","mode_ids","home_cat_idx","unknown_mode_idx"]:
        if k not in meta:
            raise ValueError(f"feature_meta.json missing '{k}'")
    return meta

def load_model_bundle(model_dir: Path, device: str):
    bundle = torch.load(_exists(model_dir/"model.pt"), map_location=device, weights_only=False)

    cols = json.loads((model_dir/"columns.json").read_text())
    ctx_cols        = cols["ctx_cols"]
    cat_cols        = cols["cat_cols"]              # <-- ADD
    mode_cols_all   = cols["mode_cols_all"]
    mode_cols_known = cols["mode_cols_known"]

    known_set = set(mode_cols_known)
    unk_candidates = [c for c in mode_cols_all if c not in known_set]
    assert len(unk_candidates) == 1, "Expected exactly one unknown mode column"
    unk_col = unk_candidates[0]
    unk_mode_idx = mode_cols_all.index(unk_col)
    mode_known_idx = [mode_cols_all.index(c) for c in mode_cols_known]

    state_keys = set(bundle["model_state"].keys())
    is_moe = any(k.startswith("head_cat.gate.") or k.startswith("head_mode.gate.") for k in state_keys)
    model_cls = PreferenceNetMoE if is_moe else PreferenceNetBase
    model_kwargs = dict(
        n_agents=int(bundle["n_agents"]),
        n_phases=int(bundle["n_phases"]),
        ctx_dim=int(bundle["ctx_dim"]),
        n_cat=int(bundle["n_cat"]),
        n_mode=int(bundle["n_mode"]),
        emb_dim=bundle["config"]["emb_dim"],
        h_dim=bundle["config"]["h_dim"],
        dropout=bundle["config"]["dropout"],
        mode_known_idx=mode_known_idx,
        unk_mode_idx=unk_mode_idx,
    )
    if is_moe:
        cfg = bundle.get("config", {})
        model_kwargs.update(
            moe_num_experts=int(cfg.get("moe_num_experts", 4)),
            moe_top_k=int(cfg.get("moe_top_k", 2)),
            moe_gate_temperature=float(cfg.get("moe_gate_temperature", 1.0)),
            moe_gate_noise_std=float(cfg.get("moe_gate_noise_std", 0.0)),
            moe_gate_dropout=float(cfg.get("moe_gate_dropout", 0.0)),
            moe_cat_top_k=cfg.get("moe_cat_top_k"),
            moe_mode_top_k=cfg.get("moe_mode_top_k"),
            moe_cat_gate_temperature=cfg.get("moe_cat_gate_temperature"),
            moe_mode_gate_temperature=cfg.get("moe_mode_gate_temperature"),
            moe_cat_gate_noise_std=cfg.get("moe_cat_gate_noise_std"),
            moe_mode_gate_noise_std=cfg.get("moe_mode_gate_noise_std"),
            moe_cat_gate_dropout=cfg.get("moe_cat_gate_dropout"),
            moe_mode_gate_dropout=cfg.get("moe_mode_gate_dropout"),
        )

    model = model_cls(**model_kwargs).to(device)
    missing, unexpected = model.load_state_dict(bundle["model_state"], strict=False)
    if missing:
        print(f"[info] load_state_dict: {len(missing)} missing key(s) (e.g. new FiLM layers) — initialised randomly: {missing[:4]}")
    if unexpected:
        print(f"[info] load_state_dict: {len(unexpected)} unexpected key(s) ignored: {unexpected[:4]}")
    model.eval()

    calib = json.loads((model_dir/"calibration.json").read_text())
    tau_mode = float(calib.get("mode_temperature", 1.0))
    tau_cat  = float(calib.get("cat_temperature",  1.0))

    aidx_df = pd.read_csv(model_dir/"agent_index.csv")
    agent2idx = dict(zip(aidx_df["agent_id"].astype(str), aidx_df["agent_idx"].astype(int)))

    return model, ctx_cols, cat_cols, mode_cols_all, mode_cols_known, unk_mode_idx, mode_known_idx, tau_mode, tau_cat, agent2idx

@torch.no_grad()
def predict_batch(model, batch_df: pd.DataFrame, ctx_cols: List[str], agent2idx: Dict[str,int],
                  tau_mode: float, tau_cat: float, device: str):
    a = torch.tensor([agent2idx.get(str(x), 0) for x in batch_df["agent_id"].astype(str)],
                     dtype=torch.long, device=device)
    p = torch.tensor(batch_df["epi_phase"].astype(int).to_numpy(),
                     dtype=torch.long, device=device)
    x = torch.tensor(batch_df[ctx_cols].to_numpy(np.float32),
                     dtype=torch.float32, device=device)
    pc, pm, t = model(a, p, x, mode_temperature=float(tau_mode), cat_temperature=float(tau_cat))
    return pc.cpu().numpy(), pm.cpu().numpy(), t.cpu().numpy()

class LagRoller:
    def __init__(self, ctx_cols: List[str], home_cat_idx: int,
                 phase_lag_means: Dict[int, Dict[str, float]] | None = None):
        self.ctx_cols = set(ctx_cols)
        self.home = []
        self.travel = []
        self.cat_rings: Dict[int, List[float]] = {}
        self.home_idx = int(home_cat_idx)
        self.phase_lag_means = phase_lag_means or {}
        self.current_phase = -1

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
                except: pass

    def _push7(self, ring: List[float], v: float):
        ring.append(float(v))
        if len(ring) > 7: ring.pop(0)

    def update_next_row(self, next_ctx: pd.Series, pc_row: np.ndarray, t_val: float,
                        next_phase: int = -1):
        # On phase transition, reset lag rings to target phase first-week means
        if next_phase >= 0 and next_phase != self.current_phase and next_phase in self.phase_lag_means:
            means = self.phase_lag_means[next_phase]
            t_reset = means.get("travel_frac", t_val)
            h_reset = means.get(f"cat_{self.home_idx}", float(pc_row[self.home_idx]) if self.home_idx < len(pc_row) else 0.0)
            if "prev_travel_frac" in self.ctx_cols:
                self.travel = [t_reset] * 7
                next_ctx["prev_travel_frac"] = t_reset
                if "prev7_travel_frac_mean" in self.ctx_cols:
                    next_ctx["prev7_travel_frac_mean"] = t_reset
            if "prev_home_share" in self.ctx_cols:
                self.home = [h_reset] * 7
                next_ctx["prev_home_share"] = h_reset
                if "prev7_home_share_mean" in self.ctx_cols:
                    next_ctx["prev7_home_share_mean"] = h_reset
            for k in list(self.cat_rings.keys()):
                v_reset = means.get(f"cat_{k}", 0.0)
                self.cat_rings[k] = [v_reset] * 7
                c = f"prev_cat_{k}"
                if c in next_ctx.index:
                    next_ctx[c] = v_reset
                c7 = f"prev7_cat_{k}_mean"
                if c7 in self.ctx_cols and c7 in next_ctx.index:
                    next_ctx[c7] = v_reset
            self.current_phase = next_phase
            return  # lag features fully reset, skip normal rolling update
        if "prev_travel_frac" in self.ctx_cols:
            next_ctx["prev_travel_frac"] = float(t_val)
            self._push7(self.travel, t_val)
            if "prev7_travel_frac_mean" in self.ctx_cols:
                next_ctx["prev7_travel_frac_mean"] = float(np.mean(self.travel))
        if "prev_home_share" in self.ctx_cols:
            h = float(pc_row[self.home_idx]) if self.home_idx < len(pc_row) else 0.0
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
                except:
                    pass

def _shard_agents(agent_ids: List[str], ngpu: int) -> List[List[str]]:
    shards = [[] for _ in range(ngpu)]
    for i, a in enumerate(sorted(agent_ids)):
        shards[i % ngpu].append(a)
    return shards

def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")

def _worker_dir_name(worker_id: Optional[int]) -> str:
    return "cpu0" if worker_id is None else f"gpu{worker_id}"


def gpu_worker(gpu_id: Optional[int],
               shard_agents: List[str],
               args,
               progress_q,
               done_q):
    if gpu_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)

    model, ctx_cols, cat_cols, mode_cols_all, mode_cols_known, unk_mode_idx, mode_known_idx, tau_mode, tau_cat, agent2idx = \
        load_model_bundle(Path(args.model_dir), device=device)

    meta = load_feature_meta(Path(args.feature_meta))
    home_cat_idx = int(meta["home_cat_idx"])
    cat_hour_cols  = [f"{c}h" for c in cat_cols]
    mode_hour_cols = [f"{c}h" for c in mode_cols_all]

    # ===== Phase-based POI mechanistic constraints =====
    cat_id_order = [int(c.split("_")[1]) for c in cat_cols]
    phase_poi_mask, home_pos = _build_phase_poi_masks(
        cat_id_order,
        home_cat_idx,
        args.poi_constraint_mode,
        Path(args.phase_poi_rules_json) if args.phase_poi_rules_json else None,
    )
    # ===== end mechanistic constraints =====

    base = pd.read_csv(_exists(args.daily_shares))
    base["date"] = pd.to_datetime(base["date"]).dt.date
    base = base[base["agent_id"].astype(str).isin(set(shard_agents))].copy()
    if "epi_phase" in base.columns:
        base["obs_epi_phase"] = pd.to_numeric(base["epi_phase"], errors="coerce").fillna(0).astype(int)
    else:
        base["obs_epi_phase"] = 0

    cf_cal = load_calendar(Path(args.cf_npi_csv))
    cf = base.drop(columns=["epi_phase","days_since_phase","days_since_npi"], errors="ignore") \
             .merge(cf_cal, on="date", how="left").copy()
    if cf["epi_phase"].isna().any():
        cf = cf.sort_values(["agent_id","date"])
        cf["epi_phase"] = cf["epi_phase"].ffill().bfill().astype(int)
        cf["days_since_phase"] = cf["days_since_phase"].ffill().bfill().astype(int)
        cf["days_since_npi"]   = cf["days_since_npi"].ffill().bfill().astype(int)

    for c in ctx_cols:
        if c not in cf.columns:
            cf[c] = 0.0
        cf[c] = pd.to_numeric(cf[c], errors="coerce").fillna(0.0).astype(np.float32)

    out_dir = Path(args.out_dir) / _worker_dir_name(gpu_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "dynamic":
        total_units = cf["agent_id"].nunique()
    else:
        total_units = len(cf)
    progress_q.put(("init", total_units))

    rows = []
    if args.mode == "static":
        cf = cf.sort_values(["agent_id","date"]).reset_index(drop=True)
        N = len(cf)
        bs = int(args.batch_size)
        for s in range(0, N, bs):
            sl = cf.iloc[s:s+bs]
            pc, pm, t = predict_batch(model, sl, ctx_cols, agent2idx, tau_mode, tau_cat, device)

            phases = sl["epi_phase"].astype(int).to_numpy()
            obs_phases = pd.to_numeric(sl.get("obs_epi_phase", sl["epi_phase"]), errors="coerce").fillna(0).astype(int).to_numpy()
            for i in range(len(sl)):
                if int(args.pre_intervention_factual) == 1 and int(phases[i]) == int(obs_phases[i]):
                    pc_obs, pm_obs, t_obs = _observed_action_from_row(sl.iloc[i], cat_cols, mode_cols_all)
                    pc[i] = pc_obs
                    pm[i] = pm_obs
                    t[i, 0] = t_obs
                pc[i], pm[i], t_i = _blend_same_phase_with_observed(
                    pc_row=pc[i],
                    pm_row=pm[i],
                    t_val=float(t[i, 0]),
                    obs_row=sl.iloc[i],
                    cf_phase=int(phases[i]),
                    cat_cols=cat_cols,
                    mode_cols_all=mode_cols_all,
                    blend=float(args.same_phase_observed_blend),
                )
                t[i, 0] = t_i
                pc[i] = _apply_phase_poi_constraint(pc[i].copy(), int(phases[i]), phase_poi_mask, home_pos)
            dh = sl["day_hours"].to_numpy(float)
            stay_h = (1.0 - t.squeeze(-1)) * dh
            trav_h = (t.squeeze(-1)) * dh

            for i in range(len(sl)):
                meta_row = {
                    "agent_id": sl["agent_id"].iloc[i],
                    "date": str(sl["date"].iloc[i]),
                    "epi_phase": int(sl["epi_phase"].iloc[i]),
                    "day_hours": float(dh[i]),
                    "total_stay_h": float(stay_h[i]),
                    "total_travel_h": float(trav_h[i]),
                }
                poi_hours  = {cat_hour_cols[j]:  float(pc[i, j] * stay_h[i])  for j in range(pc.shape[1])}
                mode_hours = {mode_hour_cols[j]: float(pm[i, j] * trav_h[i]) for j in range(pm.shape[1])}

                rows.append({**meta_row, **poi_hours, **mode_hours})
            progress_q.put(("tick", len(sl)))

    else:
        cf = cf.sort_values(["agent_id","date"]).reset_index(drop=True)

        # Compute per-phase FIRST-WEEK means from observed data for lag reset on phase transition
        # Using first 7 days of each phase captures the immediate behavioral response
        cat_cols_base = [c for c in base.columns if c.startswith("cat_") and not c.endswith("h")]
        phase_lag_means: Dict[int, Dict[str, float]] = {}
        for ph, g_ph in base.groupby("epi_phase"):
            g_ph_sorted = g_ph.sort_values("date")
            first_dates = g_ph_sorted["date"].unique()[:7]
            g_first = g_ph_sorted[g_ph_sorted["date"].isin(first_dates)]
            means: Dict[str, float] = {"travel_frac": float(g_first["travel_frac"].mean())}
            for c in cat_cols_base:
                try:
                    k = int(c.split("_")[1])
                    means[f"cat_{k}"] = float(g_first[c].mean())
                except: pass
            phase_lag_means[int(ph)] = means

        for agent_id, g in cf.groupby("agent_id", sort=False):
            g = g.copy()
            roller = LagRoller(ctx_cols, home_cat_idx, phase_lag_means)
            roller.seed_from_row(g.iloc[0])
            has_diverged = False

            for i in range(len(g)):
                sl = g.iloc[i:i+1]
                ph = int(sl["epi_phase"].iloc[0])
                obs_ph = int(pd.to_numeric(sl.get("obs_epi_phase", sl["epi_phase"]), errors="coerce").fillna(ph).iloc[0])
                if (not has_diverged) and ph == obs_ph and int(args.pre_intervention_factual) == 1:
                    pc_obs, pm_obs, t_obs = _observed_action_from_row(sl.iloc[0], cat_cols, mode_cols_all)
                    pc = np.asarray(pc_obs, dtype=float)[None, :]
                    pm = np.asarray(pm_obs, dtype=float)[None, :]
                    t = np.asarray([[t_obs]], dtype=float)
                else:
                    has_diverged = True
                    pc, pm, t = predict_batch(model, sl, ctx_cols, agent2idx, tau_mode, tau_cat, device)
                    pc[0], pm[0], t_i = _blend_same_phase_with_observed(
                        pc_row=pc[0],
                        pm_row=pm[0],
                        t_val=float(t[0, 0]),
                        obs_row=sl.iloc[0],
                        cf_phase=ph,
                        cat_cols=cat_cols,
                        mode_cols_all=mode_cols_all,
                        blend=float(args.same_phase_observed_blend),
                    )
                    t[0, 0] = t_i
                pc[0] = _apply_phase_poi_constraint(pc[0].copy(), ph, phase_poi_mask, home_pos)

                dh = float(sl["day_hours"].iloc[0])
                stay_h = float((1.0 - t[0,0]) * dh)
                trav_h = float(t[0,0] * dh)
                meta_row = {
                    "agent_id": agent_id,
                    "date": str(sl["date"].iloc[0]),
                    "epi_phase": int(sl["epi_phase"].iloc[0]),
                    "day_hours": dh,
                    "total_stay_h": stay_h,
                    "total_travel_h": trav_h,
                }
                poi_hours  = {cat_hour_cols[j]:  float(pc[0, j] * stay_h)  for j in range(pc.shape[1])}
                mode_hours = {mode_hour_cols[j]: float(pm[0, j] * trav_h) for j in range(pm.shape[1])}

                rows.append({**meta_row, **poi_hours, **mode_hours})

                if i+1 < len(g):
                    nxt_idx = g.index[i+1]
                    nxt_ctx = g.loc[nxt_idx, ctx_cols].copy()
                    nxt_ph = int(g.iloc[i+1]["epi_phase"])
                    roller.update_next_row(nxt_ctx, pc_row=pc[0], t_val=float(t[0,0]),
                                          next_phase=nxt_ph)
                    for k, v in nxt_ctx.items():
                        cf.loc[nxt_idx, k] = float(v)
                        g.loc[nxt_idx, k] = float(v)  # also update g so next iteration reads updated lag features

            progress_q.put(("tick", 1))

    pred_df = pd.DataFrame(rows).sort_values(["agent_id","date"]).reset_index(drop=True)
    poi_cols  = ["agent_id","date","epi_phase","day_hours","total_stay_h"]  + cat_hour_cols
    mode_cols = ["agent_id","date","epi_phase","day_hours","total_travel_h"] + mode_hour_cols

    poi_df  = pred_df[poi_cols].copy()
    mode_df = pred_df[mode_cols].copy()
    _write_csv(poi_df,  out_dir/"per_agent_daily_poi_hours_cf.csv")
    _write_csv(mode_df, out_dir/"per_agent_daily_mode_hours_cf.csv")

    agg_poi  = poi_df.groupby(["agent_id","epi_phase"], dropna=False)[[c for c in poi_cols if c.endswith("h")]].mean().reset_index()
    agg_mode = mode_df.groupby(["agent_id","epi_phase"], dropna=False)[[c for c in mode_cols if c.endswith("h")]].mean().reset_index()
    agent_phase = pd.merge(agg_poi, agg_mode, on=["agent_id","epi_phase"], how="outer").fillna(0.0)
    _write_csv(agent_phase, out_dir/"per_agent_phase_summary_cf.csv")

    obs = base.copy()
    stay_h_obs = (1.0 - obs["travel_frac"]) * obs["day_hours"]
    trav_h_obs = obs["travel_frac"] * obs["day_hours"]

    cat_ids = [int(x.split("_")[1]) for x in obs.columns if x.startswith("cat_") and "_" in x]
    mode_ids = [int(x.split("_")[1]) for x in obs.columns if x.startswith("mode_") and "_" in x]

    poi_obs = pd.DataFrame({"agent_id": obs["agent_id"], "date": obs["date"], "epi_phase": obs["epi_phase"],
                            **{f"cat_{k}h": obs[f"cat_{k}"] * stay_h_obs for k in cat_ids}})
    mode_obs = pd.DataFrame({"agent_id": obs["agent_id"], "date": obs["date"], "epi_phase": obs["epi_phase"],
                             **{f"mode_{k}h": obs[f"mode_{k}"] * trav_h_obs for k in mode_ids}})

    poi_delta  = poi_df.merge(poi_obs,  on=["agent_id","date","epi_phase"], suffixes=("_cf","_obs"))
    mode_delta = mode_df.merge(mode_obs, on=["agent_id","date","epi_phase"], suffixes=("_cf","_obs"))

    for c in [col for col in poi_df.columns if col.startswith("cat_") and col.endswith("h")]:
        poi_delta[c.replace("h","_deltah")] = poi_delta[f"{c}_cf"] - poi_delta[f"{c}_obs"]
    for c in [col for col in mode_df.columns if col.startswith("mode_") and col.endswith("h")]:
        mode_delta[c.replace("h","_deltah")] = mode_delta[f"{c}_cf"] - mode_delta[f"{c}_obs"]

    keep_poi  = ["agent_id","date","epi_phase"] + [c for c in poi_delta.columns  if c.endswith("_deltah")]
    keep_mode = ["agent_id","date","epi_phase"] + [c for c in mode_delta.columns if c.endswith("_deltah")]
    _write_csv(poi_delta[keep_poi],  out_dir/"delta_vs_observed_poi_hours.csv")
    _write_csv(mode_delta[keep_mode], out_dir/"delta_vs_observed_mode_hours.csv")

    chk = pd.DataFrame({
        "agent_id": pred_df["agent_id"], "date": pred_df["date"], "day_hours": pred_df["day_hours"],
        "stay_sum": pred_df[[c for c in pred_df.columns if c.startswith("cat_") and c.endswith("h")]].sum(axis=1),
        "trav_sum": pred_df[[c for c in pred_df.columns if c.startswith("mode_") and c.endswith("h")]].sum(axis=1),
    })
    chk["residual"] = chk["day_hours"] - (chk["stay_sum"] + chk["trav_sum"])
    _write_csv(chk, out_dir/"sums_check_cf.csv")

    done_q.put(("done", _worker_dir_name(gpu_id)))

def main():
    ap = argparse.ArgumentParser(description="Counterfactual NPI on 2 GPUs + 32 CPUs with global progress.")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--daily_shares", required=True)
    ap.add_argument("--feature_meta", required=True)
    ap.add_argument("--cf_npi_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--poi_constraint_mode", choices=["off", "soft", "hard"], default="off",
                    help="How phase-specific POI availability rules are applied during CF generation")
    ap.add_argument("--phase_poi_rules_json", default="",
                    help="Optional JSON file with phase-specific POI availability/weight rules")
    ap.add_argument("--devices", default="0,1", help="Comma-separated GPU ids")
    ap.add_argument("--mode", choices=["static","dynamic"], default="dynamic")
    ap.add_argument("--batch_size", type=int, default=16384, help="Static mode GPU batch size")
    ap.add_argument("--same_phase_observed_blend", type=float, default=0.0,
                    help="Optional blend toward observed behavior when CF phase equals observed phase on the same date (0 disables, 1 copies observed).")
    ap.add_argument("--pre_intervention_factual", type=int, default=0,
                    help="If 1, copy observed behavior before the first date when CF phase diverges from observed phase for an agent. "
                         "Default 0: always let the model predict so lag features evolve freely.")
    ap.add_argument("--merge_workers", type=int, default=32, help="CPU processes for final merges")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cf_policy_constraints.json").write_text(json.dumps({
        "poi_constraint_mode": args.poi_constraint_mode,
        "phase_poi_rules_json": str(Path(args.phase_poi_rules_json).resolve()) if args.phase_poi_rules_json else "",
        "same_phase_observed_blend": float(args.same_phase_observed_blend),
        "pre_intervention_factual": int(args.pre_intervention_factual),
    }, indent=2))

    # Load once to split agents
    df_head = pd.read_csv(_exists(args.daily_shares), usecols=["agent_id","date"])
    agent_ids = sorted(df_head["agent_id"].astype(str).unique().tolist())
    gpu_ids = [int(x) for x in args.devices.split(",") if x.strip()!=""]
    worker_specs: List[Tuple[Optional[int], List[str]]]
    if gpu_ids:
        shards = _shard_agents(agent_ids, len(gpu_ids))
        worker_specs = list(zip(gpu_ids, shards))
    else:
        worker_specs = [(None, agent_ids)]

    manager = Manager()
    progress_q = manager.Queue()
    done_q = manager.Queue()

    if args.mode == "dynamic":
        total_units = len(agent_ids)
    else:
        full_rows = pd.read_csv(_exists(args.daily_shares), usecols=["agent_id"]).shape[0]
        total_units = full_rows

    # Launch GPU workers
    procs: List[Process] = []
    for gpu, shard in worker_specs:
        p = Process(target=gpu_worker, args=(gpu, shard, args, progress_q, done_q), daemon=True)
        p.start()
        procs.append(p)

    pbar = tqdm(total=total_units, desc="Counterfactual simulation", dynamic_ncols=True)
    inited = 0
    done_workers = 0
    while done_workers < len(procs):
        try:
            msg, val = progress_q.get(timeout=0.2)
            if msg == "init":
                inited += 1
            elif msg == "tick":
                pbar.update(int(val))
        except queue.Empty:
            pass
        try:
            dmsg, gid = done_q.get_nowait()
            if dmsg == "done":
                done_workers += 1
        except queue.Empty:
            pass
        time.sleep(0.02)
    pbar.close()

    for p in procs: p.join()

    shard_dirs = [out_dir / _worker_dir_name(g) for g, _ in worker_specs]
    file_names = [
        "per_agent_daily_poi_hours_cf.csv",
        "per_agent_daily_mode_hours_cf.csv",
        "per_agent_phase_summary_cf.csv",
        "delta_vs_observed_poi_hours.csv",
        "delta_vs_observed_mode_hours.csv",
        "sums_check_cf.csv",
    ]

    merge_procs = max(1, int(args.merge_workers))
    chunksz = max(1, math.ceil(len(shard_dirs) / (merge_procs * 2)))  # reasonable default

    with Pool(processes=merge_procs) as pool:
        for fname in file_names:
            parts = pool.starmap(_read_one_for_merge, [(d, fname) for d in shard_dirs], chunksize=chunksz)
            parts = [x for x in parts if x is not None and len(x)]
            if parts:
                merged = pd.concat(parts, ignore_index=True)
                _write_csv(merged, out_dir / fname)

    print(f"[OK] All outputs merged in: {out_dir}")

    try:
        _write_cf_daily_shares(
            out_dir=out_dir,
            feature_meta_path=Path(args.feature_meta),
            npi_calendar_path=Path(args.cf_npi_csv),
            observed_daily_shares_path=Path(getattr(args, "daily_shares", "")) if hasattr(args, "daily_shares") else None,
            outfile_name="daily_shares_cf.csv",
        )
        print("[OK] Wrote daily_shares_cf.csv (core+calendar+weekday+demographics where available)")
    except Exception as e:
        print(f"[WARN] daily_shares_cf.csv not produced: {e}")

    try:
        _compute_pop_phase_24h(out_dir, Path(args.feature_meta), Path(args.daily_shares))
        print("[OK] Wrote population_phase_summary_24h_cf.csv (+ obs & delta if available)")
    except Exception as e:
        print(f"[WARN] population_phase_summary_24h not computed: {e}")

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "32")
    os.environ.setdefault("MKL_NUM_THREADS", "32")
    main()
