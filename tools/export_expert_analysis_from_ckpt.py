from __future__ import annotations

import argparse
import inspect
import json
import sys
import uuid
from pathlib import Path
from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preference_dataset import DailyShareDataset


def _import_from_path(py_path: Path, base_name: str):
    unique_name = f"{base_name}_{uuid.uuid4().hex}"
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(unique_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for: {py_path}")
    mod = _ilu.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_policy_init_kwargs(policy_cls, cfg: dict, base_kwargs: dict) -> dict:
    sig = inspect.signature(policy_cls.__init__)
    kwargs = dict(base_kwargs)
    for name in sig.parameters:
        if name == "self":
            continue
        if name not in kwargs and name in cfg:
            kwargs[name] = cfg[name]
    return kwargs


def _entropy_from_probs(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _phase_distribution(df: pd.DataFrame, head_name: str) -> pd.DataFrame:
    expert_col = f"{head_name}_expert_top1"
    phase_share = (
        df.groupby([expert_col, "epi_phase"]).size()
        / df.groupby(expert_col).size()
    ).reset_index(name="phase_share")
    phase_share.insert(0, "head", head_name)
    return phase_share.rename(columns={expert_col: "expert_id"})


def _summarize_by_expert(df: pd.DataFrame, schema, head_name: str, n_experts: int) -> pd.DataFrame:
    expert_col = f"{head_name}_expert_top1"
    prob_col = f"{head_name}_expert_top1_prob"
    entropy_col = f"{head_name}_expert_entropy"
    context_cols = [
        "days_since_phase", "days_since_npi", "day_hours", "n_trips",
        "total_trip_km", "mean_trip_km", "max_trip_km",
        "prev_travel_frac", "prev7_travel_frac_mean",
        "prev_home_share", "prev7_home_share_mean",
    ]
    context_cols = [c for c in context_cols if c in df.columns]

    rows = []
    for expert_id in range(n_experts):
        sub = df[df[expert_col] == expert_id]
        row = {
            "head": head_name,
            "expert_id": expert_id,
            "n_samples": int(len(sub)),
            "sample_share": float(len(sub) / max(len(df), 1)),
        }
        if len(sub) == 0:
            rows.append(row)
            continue
        row.update({
            "avg_top1_prob": float(sub[prob_col].mean()),
            "avg_entropy": float(sub[entropy_col].mean()),
            "avg_travel_frac_true": float(sub["travel_frac_true"].mean()),
            "avg_travel_frac_pred": float(sub["travel_frac_pred"].mean()),
            "avg_day_hours": float(sub["day_hours"].mean()),
            "avg_mode_unknown_pred_share": float(sub["pred_mode_5"].mean()) if "pred_mode_5" in sub.columns else np.nan,
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
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily_shares", type=str, required=True)
    ap.add_argument("--model_py", type=str, required=True)
    ap.add_argument("--model_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--checkpoint_label", type=str, default="alignment")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--calibration_json", type=str, default="")
    args = ap.parse_args()

    device_str = args.device
    if device_str.isdigit():
        device_str = f"cuda:{device_str}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    daily_path = Path(args.daily_shares)
    feature_meta_path = daily_path.with_name("feature_meta.json")
    out_dir = Path(args.out_dir)
    analysis_dir = out_dir / "expert_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    ds = DailyShareDataset(daily_path, feature_meta_path if feature_meta_path.exists() else None)
    schema = ds.schema()

    model_mod = _import_from_path(Path(args.model_py), "dsl_model_mod_export")
    PolicyClass = getattr(model_mod, "PreferenceNet")

    ckpt = torch.load(args.model_ckpt, map_location="cpu", weights_only=False)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
        cfg = ckpt.get("config", {})
        if not isinstance(cfg, dict):
            if is_dataclass(cfg):
                cfg = asdict(cfg)
            elif hasattr(cfg, "__dict__"):
                cfg = dict(cfg.__dict__)
            else:
                raise TypeError(f"Unsupported config type: {type(cfg)!r}")
        n_agents = int(ckpt.get("n_agents", len(ds.agents)))
        n_phases = int(ckpt.get("n_phases", int(ds.daily["epi_phase"].max()) + 1))
        ctx_dim = int(ckpt.get("ctx_dim", len(schema.ctx_cols)))
        n_cat = int(ckpt.get("n_cat", len(schema.cat_cols)))
        n_mode = int(ckpt.get("n_mode", len(schema.mode_cols_all)))
    else:
        raise RuntimeError("Expected checkpoint with 'model_state'.")

    init_kwargs = _resolve_policy_init_kwargs(
        PolicyClass,
        cfg,
        {
            "n_agents": n_agents,
            "n_phases": n_phases,
            "ctx_dim": ctx_dim,
            "n_cat": n_cat,
            "n_mode": n_mode,
        },
    )
    model = PolicyClass(**init_kwargs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[info] load_state_dict: {len(missing)} missing key(s), first few: {missing[:4]}")
    if unexpected:
        print(f"[info] load_state_dict: {len(unexpected)} unexpected key(s), first few: {unexpected[:4]}")
    if hasattr(model, "set_lag_feature_indices"):
        lag_indices = [i for i, c in enumerate(schema.ctx_cols) if c.startswith("prev_") or c.startswith("prev7_")]
        if lag_indices:
            model.set_lag_feature_indices(lag_indices)
            if "phase_lag_gate.weight" in state:
                model.phase_lag_gate.weight.data = state["phase_lag_gate.weight"]
                model.phase_lag_gate.bias.data = state["phase_lag_gate.bias"]
    model.to(device)
    model.eval()

    mode_temperature = 1.0
    cat_temperature = 1.0
    if args.calibration_json:
        cal_path = Path(args.calibration_json)
        if cal_path.exists():
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
            mode_temperature = float(cal.get("mode_temperature", 1.0))
            cat_temperature = float(cal.get("cat_temperature", 1.0))

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    offset = 0
    context_cols = [
        "days_since_phase", "days_since_npi", "day_hours", "n_trips",
        "total_trip_km", "mean_trip_km", "max_trip_km",
        "prev_travel_frac", "prev7_travel_frac_mean",
        "prev_home_share", "prev7_home_share_mean",
    ]
    context_cols = [c for c in context_cols if c in ds.daily.columns]

    with torch.no_grad():
        for batch in loader:
            a = batch["agent_idx"].to(device)
            p = batch["phase_id"].to(device)
            x = batch["x_ctx"].to(device)
            pc, pm, t, routing = model(
                a,
                p,
                x,
                mode_temperature=mode_temperature,
                cat_temperature=cat_temperature,
                return_routing=True,
            )

            mode_probs = routing["mode"]["gate_probs"].cpu().numpy()
            cat_probs = routing["cat"]["gate_probs"].cpu().numpy()
            pc_np = pc.cpu().numpy()
            pm_np = pm.cpu().numpy()
            t_np = t.squeeze(-1).cpu().numpy()
            y_cat = batch["y_cat"].cpu().numpy()
            y_mode = batch["y_mode"].cpu().numpy()
            travel_true = batch["travel_frac"].squeeze(-1).cpu().numpy()
            day_hours = batch["day_hours"].squeeze(-1).cpu().numpy()

            batch_n = len(a)
            idx_slice = ds.index[offset: offset + batch_n]
            row_block = ds.daily.loc[idx_slice].reset_index(drop=True)
            offset += batch_n

            for i in range(batch_n):
                record = {
                    "checkpoint": args.checkpoint_label,
                    "agent_id": str(row_block.loc[i, "agent_id"]),
                    "date": str(row_block.loc[i, "date"]),
                    "epi_phase": int(row_block.loc[i, "epi_phase"]),
                    "mode_expert_top1": int(np.argmax(mode_probs[i])),
                    "mode_expert_top1_prob": float(np.max(mode_probs[i])),
                    "mode_expert_entropy": float(_entropy_from_probs(mode_probs[i:i+1])[0]),
                    "cat_expert_top1": int(np.argmax(cat_probs[i])),
                    "cat_expert_top1_prob": float(np.max(cat_probs[i])),
                    "cat_expert_entropy": float(_entropy_from_probs(cat_probs[i:i+1])[0]),
                    "travel_frac_true": float(travel_true[i]),
                    "travel_frac_pred": float(t_np[i]),
                    "day_hours": float(day_hours[i]),
                }
                for col in context_cols:
                    record[col] = float(row_block.loc[i, col])
                for j, col in enumerate(schema.mode_cols_all):
                    record[f"true_{col}"] = float(y_mode[i, j])
                    record[f"pred_{col}"] = float(pm_np[i, j])
                for j, col in enumerate(schema.cat_cols):
                    record[f"true_{col}"] = float(y_cat[i, j])
                    record[f"pred_{col}"] = float(pc_np[i, j])
                rows.append(record)

    routing_df = pd.DataFrame(rows)
    routing_path = analysis_dir / f"per_sample_routing_{args.checkpoint_label}.csv"
    routing_df.to_csv(routing_path, index=False, float_format="%.6f")

    mode_summary = _summarize_by_expert(routing_df, schema, "mode", model.head_mode.num_experts)
    cat_summary = _summarize_by_expert(routing_df, schema, "cat", model.head_cat.num_experts)
    mode_summary.to_csv(analysis_dir / f"expert_profile_mode_{args.checkpoint_label}.csv", index=False, float_format="%.6f")
    cat_summary.to_csv(analysis_dir / f"expert_profile_cat_{args.checkpoint_label}.csv", index=False, float_format="%.6f")

    phase_df = pd.concat([
        _phase_distribution(routing_df, "mode"),
        _phase_distribution(routing_df, "cat"),
    ], ignore_index=True)
    phase_df.to_csv(analysis_dir / f"expert_phase_distribution_{args.checkpoint_label}.csv", index=False, float_format="%.6f")

    print(f"written -> {routing_path}")
    print(f"written -> {analysis_dir / f'expert_profile_mode_{args.checkpoint_label}.csv'}")
    print(f"written -> {analysis_dir / f'expert_profile_cat_{args.checkpoint_label}.csv'}")
    print(f"written -> {analysis_dir / f'expert_phase_distribution_{args.checkpoint_label}.csv'}")


if __name__ == "__main__":
    main()
