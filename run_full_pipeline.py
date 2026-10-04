"""
Full behavior-to-epidemic pipeline:
  1) Build enriched time-share
  2) Train supervised daily-share model
  3) Preference alignment fine-tune (writes alignment bundle)
  4) Counterfactual simulation (reconstruct daily_shares_cf.csv)
  5) SEIR from observed daily shares
  6) SEIR from counterfactual daily shares

This script:
  - Creates/validates paths at every stage
  - Copies columns.json & agent_index.csv next to alignment bundle (CF loader needs them)
  - Writes default calibration.json if missing (tau=1.0)
  - Can run a single PA branch or dual PA branches (`factual` + `counterfactual`)

Usage:
 if you want to train for 100 agents
  python run_full_pipeline.py --num 100 --devices 0,1
 if you want to train for 911 agents:
  python run_full_pipeline.py --num 911 --devices 0,1
 if you want both PA branches in one command:
  python run_full_pipeline.py --num 100 --devices 0,1 --align_profile dual
"""

from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

# ----------------------------- utils -----------------------------
def sh(cmd, cwd=None):
    print("\n$ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise SystemExit(f"Command failed (exit {res.returncode}): {' '.join(cmd)}")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def must_exist(path: Path, why: str = "") -> Path:
    if not path.exists():
        msg = f"Missing required file: {path}"
        if why: msg += f"\n  ↳ {why}"
        raise SystemExit(msg)
    return path

def copy_if_missing(src: Path, dst: Path):
    if not dst.exists():
        shutil.copy2(src, dst)

def write_default_calibration_json(model_dir: Path):
    calib = model_dir / "calibration.json"
    if not calib.exists():
        calib.write_text(json.dumps({"mode_temperature": 1.0, "cat_temperature": 1.0}, indent=2))
        print(f"[info] wrote default calibration.json → {calib}")


def build_effective_seir_config(
    *,
    daily_shares: Path,
    hazard_params: Path,
    feature_meta: Path,
    latent_days: int,
    infectious_days: int,
    rng_seed: int,
    workers: int,
    mixing_scale: str,
    home_beta_alpha: float,
    importation_prob: float | None,
    seed_pct: float,
    beta_scale: float,
):
    with open(hazard_params, "r", encoding="utf-8") as f:
        hp = json.load(f)
    hp["beta_base"] = hp.get("beta_base", 0.06) * beta_scale
    if importation_prob is not None:
        hp["importation_prob"] = importation_prob
    phase_beta_mult = {str(k): float(v) for k, v in (hp.get("phase_beta_mult", {}) or {}).items()}
    return {
        "daily_shares": str(daily_shares.resolve()),
        "hazard_params_source": str(hazard_params.resolve()),
        "feature_meta": str(feature_meta.resolve()),
        "latent_days": int(latent_days),
        "infectious_days": int(infectious_days),
        "seed_pct": float(seed_pct),
        "rng_seed": int(rng_seed),
        "workers": int(max(1, workers)),
        "beta_scale": float(beta_scale),
        "mixing_scale": mixing_scale,
        "home_beta_alpha": float(home_beta_alpha),
        "phase_beta_mult": phase_beta_mult,
        "hazard_params_effective": hp,
    }


def _json_matches(summary_path: Path, desired: dict, keys: list[str]) -> bool:
    if not summary_path.exists():
        return False
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except Exception:
        return False
    return all(existing.get(k) == desired.get(k) for k in keys)


def build_effective_template_config(
    *,
    daily_geo: Path,
    daily_shares: Path,
    feature_meta: Path,
    out_path: Path,
    bucket_mode: str,
    topk: int,
    min_template_hours: float,
    min_template_days: int,
    include_home: bool,
):
    return {
        "daily_geo": str(daily_geo),
        "daily_shares": str(daily_shares),
        "out_path": str(out_path),
        "home_cat_idx": int(json.loads(feature_meta.read_text()).get("home_cat_idx", 11)),
        "include_home": bool(include_home),
        "bucket_mode": str(bucket_mode),
        "topk": int(topk),
        "min_template_hours": float(min_template_hours),
        "min_template_days": int(min_template_days),
    }


def run_poi_template_build(
    *,
    daily_geo: Path,
    daily_shares: Path,
    feature_meta: Path,
    out_path: Path,
    bucket_mode: str,
    topk: int,
    min_template_hours: float,
    min_template_days: int,
    include_home: bool,
):
    ensure_dir(out_path.parent)
    summary_json = Path(str(out_path) + ".summary.json")
    desired_cfg = build_effective_template_config(
        daily_geo=daily_geo,
        daily_shares=daily_shares,
        feature_meta=feature_meta,
        out_path=out_path,
        bucket_mode=bucket_mode,
        topk=topk,
        min_template_hours=min_template_hours,
        min_template_days=min_template_days,
        include_home=include_home,
    )
    should_run = not (
        out_path.exists()
        and _json_matches(
            summary_json,
            desired_cfg,
            [
                "daily_geo",
                "daily_shares",
                "out_path",
                "home_cat_idx",
                "include_home",
                "bucket_mode",
                "topk",
                "min_template_hours",
                "min_template_days",
            ],
        )
    )
    if should_run:
        cmd = [
            sys.executable, "build_poi_allocation_templates.py",
            "--daily_geo", str(daily_geo),
            "--daily_shares", str(daily_shares),
            "--feature_meta", str(feature_meta),
            "--out_path", str(out_path),
            "--bucket_mode", bucket_mode,
            "--topk", str(int(topk)),
            "--min_template_hours", str(float(min_template_hours)),
            "--min_template_days", str(int(min_template_days)),
        ]
        if include_home:
            cmd.append("--include_home")
        sh(cmd)
    must_exist(out_path, "POI allocation template table missing.")
    must_exist(summary_json, "POI allocation template summary JSON missing.")
    return out_path


def build_effective_poi_alloc_config(
    *,
    daily_shares: Path,
    templates: Path,
    feature_meta: Path,
    out_path: Path,
    bucket_mode: str,
):
    return {
        "daily_shares": str(daily_shares),
        "templates": str(templates),
        "out_path": str(out_path),
        "bucket_mode": str(bucket_mode),
        "home_cat_idx": int(json.loads(feature_meta.read_text()).get("home_cat_idx", 11)),
    }


def run_poi_allocation(
    *,
    daily_shares: Path,
    templates: Path,
    feature_meta: Path,
    out_path: Path,
    bucket_mode: str,
):
    ensure_dir(out_path.parent)
    summary_json = Path(str(out_path) + ".summary.json")
    desired_cfg = build_effective_poi_alloc_config(
        daily_shares=daily_shares,
        templates=templates,
        feature_meta=feature_meta,
        out_path=out_path,
        bucket_mode=bucket_mode,
    )
    should_run = not (
        out_path.exists()
        and _json_matches(
            summary_json,
            desired_cfg,
            ["daily_shares", "templates", "out_path", "bucket_mode", "home_cat_idx"],
        )
    )
    if should_run:
        sh([
            sys.executable, "allocate_poi_hours_from_daily_shares.py",
            "--daily_shares", str(daily_shares),
            "--templates", str(templates),
            "--feature_meta", str(feature_meta),
            "--out_path", str(out_path),
            "--bucket_mode", bucket_mode,
        ])
    must_exist(out_path, "Allocated POI-hours table missing.")
    must_exist(summary_json, "Allocated POI-hours summary JSON missing.")
    return out_path


def build_effective_poi_seir_config(
    *,
    daily_shares: Path,
    allocated_poi_hours: Path,
    hazard_params: Path,
    feature_meta: Path,
    latent_days: int,
    infectious_days: int,
    rng_seed: int,
    workers: int,
    importation_prob: float | None,
    seed_pct: float,
    beta_scale: float,
):
    with open(hazard_params, "r", encoding="utf-8") as f:
        hp = json.load(f)
    hp["beta_base"] = hp.get("beta_base", 0.06) * beta_scale
    if importation_prob is not None:
        hp["importation_prob"] = importation_prob
    phase_beta_mult = {str(k): float(v) for k, v in (hp.get("phase_beta_mult", {}) or {}).items()}
    return {
        "daily_shares": str(daily_shares.resolve()),
        "allocated_poi_hours": str(allocated_poi_hours.resolve()),
        "hazard_params_source": str(hazard_params.resolve()),
        "feature_meta": str(feature_meta.resolve()),
        "latent_days": int(latent_days),
        "infectious_days": int(infectious_days),
        "seed_pct": float(seed_pct),
        "rng_seed": int(rng_seed),
        "workers": int(max(1, workers)),
        "beta_scale": float(beta_scale),
        "phase_beta_mult": phase_beta_mult,
        "hazard_params_effective": hp,
    }


def run_poi_seir_from_daily(
    *,
    daily_shares: Path,
    allocated_poi_hours: Path,
    hazard_params: Path,
    out_dir: Path,
    feature_meta: Path,
    latent_days: int,
    infectious_days: int,
    rng_seed: int,
    workers: int,
    importation_prob: float | None,
    seed_pct: float,
    beta_scale: float,
    label: str,
):
    ensure_dir(out_dir)
    out_csv = out_dir / "seir_timeseries.csv"
    effective_cfg_path = out_dir / "effective_hazard_params.json"
    desired_cfg = build_effective_poi_seir_config(
        daily_shares=daily_shares,
        allocated_poi_hours=allocated_poi_hours,
        hazard_params=hazard_params,
        feature_meta=feature_meta,
        latent_days=latent_days,
        infectious_days=infectious_days,
        rng_seed=rng_seed,
        workers=workers,
        importation_prob=importation_prob,
        seed_pct=seed_pct,
        beta_scale=beta_scale,
    )
    should_run = True
    if out_csv.exists() and effective_cfg_path.exists():
        try:
            with open(effective_cfg_path, "r", encoding="utf-8") as f:
                existing_cfg = json.load(f)
            should_run = existing_cfg != desired_cfg
            if should_run:
                print(f"[info] Re-running {label} POI-SEIR because effective configuration changed.")
        except Exception:
            should_run = True
    if should_run:
        cmd = [
            sys.executable, "SEIR/seir_timeseries_poi.py",
            "--daily_shares", str(daily_shares),
            "--allocated_poi_hours", str(allocated_poi_hours),
            "--hazard_params", str(hazard_params),
            "--out_dir", str(out_dir),
            "--feature_meta", str(feature_meta),
            "--latent_days", str(latent_days),
            "--infectious_days", str(infectious_days),
            "--rng_seed", str(rng_seed),
            "--workers", str(max(1, workers)),
            "--seed_pct", str(seed_pct),
            "--beta_scale", str(beta_scale),
        ]
        if importation_prob is not None:
            cmd.append(f"--importation_prob={importation_prob}")
        sh(cmd)
    must_exist(out_csv, f"{label} POI-SEIR output missing.")
    return out_csv

def run_make_hours(daily_path, meta_path, out_dir, workers=28, chunksize=500_000):
    ensure_dir(out_dir)
    sh([
        sys.executable, "make_hours_from_daily_shares_mp.py",
        "--daily_shares", str(daily_path),
        "--feature_meta", str(meta_path),
        "--out_dir",      str(out_dir),
        "--workers",      str(max(1, int(workers))),
        "--chunksize",    str(int(chunksize)),
    ])


def build_alignment_runs(result_dir: Path, num: int, align_profile: str):
    profile_specs = {
        "standard": {
            "profile": "standard",
            "label": "PreferenceNet-MoE+PA",
            "align_dir": result_dir / "out" / "preference_align" / str(num),
            "cf_dir": result_dir / "out" / "OLD" / str(num) / "cf2" / "phase_shift_dynamic_mgpu",
            "cf_seir_dir": result_dir / "out" / "OLD" / str(num) / "cf2" / "seir_timeseries",
            "cf_seir_poi_dir": result_dir / "out" / "OLD" / str(num) / "cf2" / "seir_timeseries_poi",
        },
        "factual": {
            "profile": "factual",
            "label": "PreferenceNet-MoE+PA factual",
            "align_dir": result_dir / "out" / "preference_align_factual" / str(num),
            "cf_dir": result_dir / "out" / "OLD" / str(num) / "cf2_factual" / "phase_shift_dynamic_mgpu",
            "cf_seir_dir": result_dir / "out" / "OLD" / str(num) / "cf2_factual" / "seir_timeseries",
            "cf_seir_poi_dir": result_dir / "out" / "OLD" / str(num) / "cf2_factual" / "seir_timeseries_poi",
        },
        "counterfactual": {
            "profile": "counterfactual",
            "label": "PreferenceNet-MoE+PA counterfactual",
            "align_dir": result_dir / "out" / "preference_align_counterfactual" / str(num),
            "cf_dir": result_dir / "out" / "OLD" / str(num) / "cf2_counterfactual" / "phase_shift_dynamic_mgpu",
            "cf_seir_dir": result_dir / "out" / "OLD" / str(num) / "cf2_counterfactual" / "seir_timeseries",
            "cf_seir_poi_dir": result_dir / "out" / "OLD" / str(num) / "cf2_counterfactual" / "seir_timeseries_poi",
        },
    }
    if align_profile == "dual":
        ordered = ["factual", "counterfactual"]
    else:
        ordered = [align_profile]
    return [dict(name=name, **profile_specs[name]) for name in ordered]


def run_alignment_branch(
    branch: dict,
    *,
    enr_daily: Path,
    sup_columns: Path,
    sup_aindex: Path,
    sup_model_pt: Path,
    args,
):
    align_dir = branch["align_dir"]
    ensure_dir(align_dir)
    if args.force_alignment or not (align_dir / "model.pt").exists():
        cmd = [
            sys.executable, "preference_alignment_finetune.py",
            "--daily_shares", str(enr_daily),
            "--columns_json", str(sup_columns),
            "--agent_index_csv", str(sup_aindex),
            "--model_py", "daily_share_model.py",
            "--model_ckpt", str(sup_model_pt),
            "--out_dir", str(align_dir),
            "--score_epochs", str(args.align_score_epochs),
            "--finetune_epochs", str(args.align_finetune_epochs),
            "--alignment_profile", branch["profile"],
            "--selection_mode", args.align_selection_mode,
            "--phase_rank_weight", str(args.align_phase_rank_weight),
            "--pol_phase_weight", str(args.align_pol_phase_weight),
            "--pol_neg_weight", str(args.align_pol_neg_weight),
            "--phase_neg_weight", str(args.align_phase_neg_weight),
            "--phase_transition_boost", str(args.align_phase_transition_boost),
            "--phase_transition_scale", str(args.align_phase_transition_scale),
            "--lambda_phase_pref", str(args.align_lambda_phase_pref),
            "--device", args.devices.split(",")[0] if args.devices.strip() else "cpu",
        ]
        if branch["profile"] == "counterfactual":
            cmd += [
                "--lambda_post_phase_response_cat",    "0.60",
                "--lambda_post_phase_response_mode",   "0.45",
                "--lambda_post_phase_response_travel", "0.45",
                "--post_phase_response_cat_margin",    "0.40",
                "--post_phase_response_mode_margin",   "0.28",
                "--post_phase_response_travel_margin", "0.12",
                "--lambda_phase_wrong_cat",            "0.30",
                "--lambda_phase_wrong_mode",           "0.20",
                "--lambda_phase_wrong_travel",         "0.20",
                "--phase_wrong_cat_margin",            "0.30",
                "--phase_wrong_mode_margin",           "0.22",
                "--phase_wrong_travel_margin",         "0.10",
                "--finetune_epochs",                   "24",
            ]
        if args.align_enable_phase_neg_ranking:
            cmd.append("--enable_phase_neg_ranking")
        sh(cmd)
    copy_if_missing(sup_columns, align_dir / "columns.json")
    copy_if_missing(sup_aindex,  align_dir / "agent_index.csv")
    write_default_calibration_json(align_dir)
    must_exist(align_dir / "model.pt", f"{branch['label']} bundle (model.pt) was not created.")
    return align_dir


def run_counterfactual(
    *,
    model_dir: Path,
    daily_shares: Path,
    feature_meta: Path,
    cf_npi_csv: Path,
    out_dir: Path,
    poi_constraint_mode: str,
    phase_poi_rules: Path | None,
    devices: str,
    merge_workers: int,
    label: str,
):
    ensure_dir(out_dir)
    sh([
        sys.executable, "counterfactual_pipeline.py",
        "--model_dir",    str(model_dir),
        "--daily_shares", str(daily_shares),
        "--feature_meta", str(feature_meta),
        "--cf_npi_csv",   str(cf_npi_csv),
        "--out_dir",      str(out_dir),
        "--poi_constraint_mode", poi_constraint_mode,
        "--phase_poi_rules_json", str(phase_poi_rules) if phase_poi_rules is not None else "",
        "--devices",      devices,
        "--mode",         "dynamic",
        "--merge_workers", str(max(1, merge_workers)),
        "--pre_intervention_factual", "0"  # 🔧 关键修复：允许整个时间段使用模型预测而非观察数据
    ])
    cf_daily = out_dir / "daily_shares_cf.csv"
    must_exist(cf_daily, f"{label} counterfactual reconstructed daily shares.")
    return cf_daily


def run_seir_from_daily(
    *,
    daily_shares: Path,
    hazard_params: Path,
    out_dir: Path,
    feature_meta: Path,
    latent_days: int,
    infectious_days: int,
    rng_seed: int,
    workers: int,
    mixing_scale: str,
    home_beta_alpha: float,
    importation_prob: float | None,
    seed_pct: float,
    beta_scale: float,
    label: str,
):
    ensure_dir(out_dir)
    out_csv = out_dir / "seir_timeseries.csv"
    effective_cfg_path = out_dir / "effective_hazard_params.json"
    desired_cfg = build_effective_seir_config(
        daily_shares=daily_shares,
        hazard_params=hazard_params,
        feature_meta=feature_meta,
        latent_days=latent_days,
        infectious_days=infectious_days,
        rng_seed=rng_seed,
        workers=workers,
        mixing_scale=mixing_scale,
        home_beta_alpha=home_beta_alpha,
        importation_prob=importation_prob,
        seed_pct=seed_pct,
        beta_scale=beta_scale,
    )
    should_run = True
    if out_csv.exists() and effective_cfg_path.exists():
        try:
            with open(effective_cfg_path, "r", encoding="utf-8") as f:
                existing_cfg = json.load(f)
            should_run = existing_cfg != desired_cfg
            if should_run:
                print(f"[info] Re-running {label} SEIR because effective configuration changed.")
        except Exception:
            should_run = True
    if should_run:
        cmd = [
            sys.executable, "SEIR/seir_timeseries.py",
            "--daily_shares", str(daily_shares),
            "--hazard_params", str(hazard_params),
            "--out_dir", str(out_dir),
            "--feature_meta", str(feature_meta),
            "--latent_days", str(latent_days),
            "--infectious_days", str(infectious_days),
            "--rng_seed", str(rng_seed),
            "--workers", str(max(1, workers)),
            "--mixing_scale", mixing_scale,
            "--home_beta_alpha", str(home_beta_alpha),
            "--seed_pct", str(seed_pct),
            "--beta_scale", str(beta_scale),
        ]
        if importation_prob is not None:
            cmd.append(f"--importation_prob={importation_prob}")
        sh(cmd)
    must_exist(out_csv, f"{label} SEIR output missing.")
    return out_csv


# ----------------------------- pipeline -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Run pipeline through SEIR (observed + CF), no comparison.")
    ap.add_argument("--num", type=int, required=True, help="Dataset number, e.g., 100")
    ap.add_argument("--devices", default="0,1", help="Comma-separated GPU ids for CF (e.g., 0,1 or '' for CPU)")
    ap.add_argument("--jobs", type=int, default=28, help="CPU workers for various steps")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cf_calendar", default="early/calendar.csv")
    ap.add_argument("--poi_constraint_mode", choices=["off", "soft", "hard"], default="off",
                    help="How strongly external phase-specific POI rules constrain CF predictions; default off uses pure model response")
    ap.add_argument("--phase_poi_rules_json", default="dataset/MetaData/phase_poi_rules.json",
                    help="Optional JSON file defining phase-specific POI availability/weights")
    ap.add_argument("--hazard_params", default="dataset/MetaData/hazard_params.json")
    ap.add_argument("--beta_scale_obs", type=float, default=1.0)
    ap.add_argument("--beta_scale_cf",  type=float, default=1.0)
    ap.add_argument("--latent_obs", type=int, default=3)
    ap.add_argument("--infectious_obs", type=int, default=7)
    ap.add_argument("--latent_cf", type=int, default=3)
    ap.add_argument("--infectious_cf", type=int, default=7)
    ap.add_argument("--seed_pct_obs", type=float, default=0.05)
    ap.add_argument("--seed_pct_cf",  type=float, default=0.05)
    ap.add_argument("--importation_obs", type=float, default=None,
                    help="Optional observed-path importation override. Default None uses hazard_params.json.")
    ap.add_argument("--importation_cf",  type=float, default=None,
                    help="Optional counterfactual-path importation override. Default None uses hazard_params.json.")
    ap.add_argument("--mixing_scale", choices=["observed","24h"], default="24h")
    ap.add_argument("--home_beta_alpha", type=float, default=0.0)
    ap.add_argument("--split_mode", choices=["phase_block_random", "phase_temporal", "phase_last_week_test", "global_temporal", "leave_one_phase_out", "transition_backtest", "unseen_agent"], default="phase_last_week_test",
                    help="Data split mode passed to PreferenceNet and PreferenceNet-MoE training")
    ap.add_argument("--holdout_phase", type=int, default=None,
                    help="Held-out phase ID used when split_mode=leave_one_phase_out.")
    ap.add_argument("--transition_backtest_idx", type=int, default=None,
                    help="0-based global phase-transition index used when split_mode=transition_backtest.")
    ap.add_argument("--agent_val_frac", type=float, default=0.10,
                    help="Validation-agent fraction used when split_mode=unseen_agent.")
    ap.add_argument("--agent_test_frac", type=float, default=0.10,
                    help="Test-agent fraction used when split_mode=unseen_agent.")
    ap.add_argument("--val_days", type=int, default=14, help="Validation days passed to supervised training")
    ap.add_argument("--test_days", type=int, default=14, help="Test days passed to supervised training")
    ap.add_argument("--supervised_moe_aux_weight", type=float, default=0.05)
    ap.add_argument("--supervised_phase_sens_weight", type=float, default=0.20)
    ap.add_argument("--supervised_phase_sens_margin", type=float, default=0.05)
    ap.add_argument("--supervised_disable_film", action="store_true",
                    help="Disable FiLM in daily_share_model.py for supervised ablations.")
    ap.add_argument("--supervised_disable_moe_heads", action="store_true",
                    help="Disable MoE heads in daily_share_model.py for supervised ablations.")
    ap.add_argument("--supervised_disable_phase_lag_gate", action="store_true",
                    help="Disable phase-gated lag suppression in daily_share_model.py for supervised ablations.")
    ap.add_argument("--supervised_out", default="", help="Reuse an existing PreferenceNet-MoE model dir (columns.json, agent_index.csv, model.pt)")
    ap.add_argument("--run_origin_baseline", action="store_true", help="Also train/run the PreferenceNet baseline and produce factual/counterfactual outputs for dashboard comparison")
    ap.add_argument("--origin_supervised_out", default="", help="Reuse an existing PreferenceNet baseline model dir")
    ap.add_argument("--align_phase_rank_weight", type=float, default=1.00)
    ap.add_argument("--align_pol_phase_weight", type=float, default=0.75)
    ap.add_argument("--align_pol_neg_weight", type=float, default=0.25)
    ap.add_argument("--align_phase_neg_weight", type=float, default=0.0)
    ap.add_argument("--align_enable_phase_neg_ranking", action="store_true",
                    help="Explicitly enable the scorer constraint a_wrongphase > a_disturbnegative during PA. Default off.")
    ap.add_argument("--align_phase_transition_boost", type=float, default=3.0)
    ap.add_argument("--align_phase_transition_scale", type=float, default=3.0)
    ap.add_argument("--align_lambda_phase_pref", type=float, default=0.50)
    ap.add_argument("--align_score_epochs", type=int, default=50, help="Preference scorer training epochs")
    ap.add_argument("--align_finetune_epochs", type=int, default=8, help="Preference alignment fine-tuning epochs")
    ap.add_argument("--align_profile", choices=["standard", "factual", "counterfactual", "dual"], default="standard",
                    help="Alignment profile passed to preference_alignment_finetune.py")
    ap.add_argument("--align_selection_mode", choices=["auto", "val", "train_loss", "last"], default="auto",
                    help="Checkpoint selection strategy for preference alignment")
    ap.add_argument("--force_alignment", action="store_true", help="Re-run preference alignment even if ALIGN_DIR/model.pt already exists")
    ap.add_argument("--run_poi_seir", action="store_true",
                    help="Also build POI allocation templates, allocate POI hours, and run POI-level SEIR.")
    ap.add_argument("--poi_bucket_mode", choices=["weekday_weekend", "full_week"], default="weekday_weekend")
    ap.add_argument("--poi_topk", type=int, default=5)
    ap.add_argument("--poi_min_template_hours", type=float, default=0.25)
    ap.add_argument("--poi_min_template_days", type=int, default=1)
    ap.add_argument("--poi_include_home", action="store_true")
    ap.add_argument("--enrich_safe_mode", action="store_true",
                    help="Run enrichment with the robust CSV worker. Slower, but safer for malformed chain files.")
    ap.add_argument("--enrich_skip_bad_files", action="store_true",
                    help="Allow enrichment to skip malformed chain files and continue with the remaining agents.")
    ap.add_argument("--chains_dir", default="",
                    help="Optional override for dataset/<NUM>/all_agents. Useful when multiple scenario folders share one raw-chain directory.")
    args = ap.parse_args()
    if not args.align_enable_phase_neg_ranking and abs(float(args.align_phase_neg_weight)) > 1e-12:
        print(
            f"[info] align_phase_neg_weight={args.align_phase_neg_weight} was provided but "
            "a_wrongphase > a_disturbnegative is disabled by default; forcing align_phase_neg_weight=0.0. "
            "Pass --align_enable_phase_neg_ranking to opt in."
        )
        args.align_phase_neg_weight = 0.0

    NUM = args.num
    ROOT = Path(".").resolve()

    # Inputs & canonical paths
    META_DIR = ROOT / "dataset" / "MetaData"
    CHAINS   = (Path(args.chains_dir).resolve() if args.chains_dir else (ROOT / "dataset" / str(NUM) / "all_agents"))
    POIS     = META_DIR / f"pois_visited_345678_{NUM}.csv"
    HOME     = META_DIR / f"{NUM}_agents_home_centroids_with_demographics_345678.csv"
    CAL      = META_DIR / "npi_calendar.csv"
    INDEXMAP = META_DIR / "index_maps.json"
    CF_CAL   = ROOT / args.cf_calendar
    PHASE_POI_RULES = ROOT / args.phase_poi_rules_json if args.phase_poi_rules_json else None
    HAZARD   = ROOT / args.hazard_params

    RESULT_DIR = ROOT / "Results"
    ENR_DIR    = RESULT_DIR / "runs" / f"enriched_{NUM}"
    SUP_DIR    = RESULT_DIR / "out" / "OLD" / str(NUM) / "preferences"
    ORIGIN_SUP_DIR = RESULT_DIR / "out" / "OLD" / str(NUM) / "preferences_origin"
    CF_DIR_ORIGIN = RESULT_DIR / "out" / "OLD" / str(NUM) / "cf2_origin" / "phase_shift_dynamic_mgpu"
    CF_DIR_SUP = RESULT_DIR / "out" / "OLD" / str(NUM) / "cf2_supervised" / "phase_shift_dynamic_mgpu"
    OBS_SEIR   = RESULT_DIR / "out" / str(NUM) / "seir_timeseries"
    OBS_SEIR_POI = RESULT_DIR / "out" / str(NUM) / "seir_timeseries_poi"
    CF_SEIR_ORIGIN = RESULT_DIR / "out" / "OLD" / str(NUM) / "cf2_origin" / "seir_timeseries"
    CF_SEIR_SUP = RESULT_DIR / "out" / "OLD" / str(NUM) / "cf2_supervised" / "seir_timeseries"
    CF_SEIR_ORIGIN_POI = RESULT_DIR / "out" / "OLD" / str(NUM) / "cf2_origin" / "seir_timeseries_poi"
    CF_SEIR_SUP_POI = RESULT_DIR / "out" / "OLD" / str(NUM) / "cf2_supervised" / "seir_timeseries_poi"
    POI_TEMPLATE = RESULT_DIR / "out" / f"poi_allocation_templates_{NUM}.csv"
    OBS_ALLOC_POI = RESULT_DIR / "out" / str(NUM) / "allocated_poi_hours_observed.csv"
    ALIGN_RUNS = build_alignment_runs(RESULT_DIR, NUM, args.align_profile)

    for d in (ENR_DIR, SUP_DIR, ORIGIN_SUP_DIR, CF_DIR_ORIGIN, CF_DIR_SUP, OBS_SEIR, OBS_SEIR_POI, CF_SEIR_ORIGIN, CF_SEIR_SUP, CF_SEIR_ORIGIN_POI, CF_SEIR_SUP_POI):
        ensure_dir(d)
    for branch in ALIGN_RUNS:
        ensure_dir(branch["align_dir"])
        ensure_dir(branch["cf_dir"])
        ensure_dir(branch["cf_seir_dir"])
        ensure_dir(branch["cf_seir_poi_dir"])

    # Check critical inputs
    must_exist(CHAINS, "Folder with raw per-agent chain CSVs.")
    must_exist(POIS, "POIs CSV.")
    must_exist(HOME, "Home centroids/demographics CSV.")
    must_exist(CAL, "Baseline NPI calendar.")
    must_exist(INDEXMAP, "index_maps.json (POIs/Modes ID mapping).")
    must_exist(CF_CAL, "Counterfactual NPI calendar CSV.")
    if PHASE_POI_RULES is not None and str(args.phase_poi_rules_json).strip():
        must_exist(PHASE_POI_RULES, "Phase-specific POI rules JSON.")
    must_exist(HAZARD, "SEIR hazard parameters JSON.")

    # 1) Enrichment
    enr_daily = ENR_DIR / "daily_shares.csv"
    enr_meta  = ENR_DIR / "feature_meta.json"
    if not enr_daily.exists() or not enr_meta.exists():
        enrich_cmd = [
            sys.executable, "build_enriched_timeshare.py",
            "--chains_dir", str(CHAINS),
            "--pois_csv",   str(POIS),
            "--home_csv",   str(HOME),
            "--npi_csv",    str(CAL),
            "--index_json", str(INDEXMAP),
            "--out_dir",    str(ENR_DIR),
            "--jobs",       str(max(1, args.jobs)),
            "--resume_existing",
        ]
        if args.enrich_safe_mode:
            enrich_cmd.append("--safe_mode")
        if args.enrich_skip_bad_files:
            enrich_cmd.append("--skip_bad_files")
        sh(enrich_cmd)
    must_exist(enr_daily, "Output of enrichment step.")
    must_exist(enr_meta,  "feature_meta.json written by enrichment.")

    OBS_HOURS = RESULT_DIR / "out" / "OLD" / str(NUM) / "observed_hours_mp"
    run_make_hours(enr_daily, enr_meta, OBS_HOURS, workers=args.jobs, chunksize=500_000)

    # 2) Supervised training (or reuse)
    if args.supervised_out:
        SUP_DIR = Path(args.supervised_out).resolve()
        print(f"[info] Using provided PreferenceNet-MoE model dir: {SUP_DIR}")
    else:
        sup_cmd = [
            sys.executable, "daily_share_model.py",
            "--daily_shares", str(enr_daily),
            "--feature_meta", str(enr_meta),
            "--out_dir",      str(SUP_DIR),
            "--epochs", "200",
            "--warmup_epochs", "10",
            "--batch_size", "1024",
            "--lr", "6e-4",
            "--weight_decay", "1e-4",
            "--emb_dim", "64",
            "--h_dim", "256",
            "--dropout", "0.25",
            "--val_days", str(args.val_days),
            "--test_days", str(args.test_days),
            "--moe_aux_weight", str(args.supervised_moe_aux_weight),
            "--moe_gate_noise_std", "0.15",
            "--moe_gate_dropout", "0.02",
            "--default_checkpoint_metric", "balanced",
            "--split_mode", args.split_mode,
            "--seed", str(args.seed),
            "--num_workers", str(max(1, args.jobs)),
            "--agent_val_frac", str(args.agent_val_frac),
            "--agent_test_frac", str(args.agent_test_frac),
            "--phase_sens_weight", str(args.supervised_phase_sens_weight),
            "--phase_sens_margin", str(args.supervised_phase_sens_margin),
        ]
        if args.holdout_phase is not None:
            sup_cmd += ["--holdout_phase", str(args.holdout_phase)]
        if args.transition_backtest_idx is not None:
            sup_cmd += ["--transition_backtest_idx", str(args.transition_backtest_idx)]
        if args.supervised_disable_film:
            sup_cmd.append("--disable_film")
        if args.supervised_disable_moe_heads:
            sup_cmd.append("--disable_moe_heads")
        if args.supervised_disable_phase_lag_gate:
            sup_cmd.append("--disable_phase_lag_gate")
        sh(sup_cmd)
    sup_model_pt  = SUP_DIR / "model.pt"
    sup_columns   = SUP_DIR / "columns.json"
    sup_aindex    = SUP_DIR / "agent_index.csv"
    must_exist(sup_model_pt, "Supervised model checkpoint/bundle.")
    must_exist(sup_columns,  "columns.json from supervised training.")
    must_exist(sup_aindex,   "agent_index.csv from supervised training.")

    origin_model_pt = None
    origin_columns = None
    origin_aindex = None
    if args.run_origin_baseline:
        if args.origin_supervised_out:
            ORIGIN_SUP_DIR = Path(args.origin_supervised_out).resolve()
            print(f"[info] Using provided PreferenceNet baseline model dir: {ORIGIN_SUP_DIR}")
        else:
            origin_cmd = [
                sys.executable, "daily_share_model_baseline.py",
                "--daily_shares", str(enr_daily),
                "--feature_meta", str(enr_meta),
                "--out_dir",      str(ORIGIN_SUP_DIR),
                "--epochs", "200",
                "--warmup_epochs", "10",
                "--batch_size", "1024",
                "--lr", "6e-4",
                "--weight_decay", "1e-4",
                "--emb_dim", "64",
                "--h_dim", "256",
                "--dropout", "0.25",
                "--val_days", str(args.val_days),
                "--test_days", str(args.test_days),
                "--split_mode", args.split_mode,
                "--agent_val_frac", str(args.agent_val_frac),
                "--agent_test_frac", str(args.agent_test_frac),
                "--seed", str(args.seed),
                "--num_workers", str(max(1, args.jobs))
            ]
            if args.holdout_phase is not None:
                origin_cmd += ["--holdout_phase", str(args.holdout_phase)]
            if args.transition_backtest_idx is not None:
                origin_cmd += ["--transition_backtest_idx", str(args.transition_backtest_idx)]
            sh(origin_cmd)
        origin_model_pt  = ORIGIN_SUP_DIR / "model.pt"
        origin_columns   = ORIGIN_SUP_DIR / "columns.json"
        origin_aindex    = ORIGIN_SUP_DIR / "agent_index.csv"
        must_exist(origin_model_pt, "Origin baseline model checkpoint/bundle.")
        must_exist(origin_columns,  "columns.json from PreferenceNet baseline training.")
        must_exist(origin_aindex,   "agent_index.csv from PreferenceNet baseline training.")

    # 3) Preference alignment fine-tune (writes one or more alignment bundles)
    aligned_dirs = {}
    for branch in ALIGN_RUNS:
        aligned_dirs[branch["name"]] = run_alignment_branch(
            branch,
            enr_daily=enr_daily,
            sup_columns=sup_columns,
            sup_aindex=sup_aindex,
            sup_model_pt=sup_model_pt,
            args=args,
        )

    # 4) Counterfactual simulation (origin direct, optional baseline)
    cf_daily_origin = None
    if args.run_origin_baseline:
        cf_daily_origin = run_counterfactual(
            model_dir=ORIGIN_SUP_DIR,
            daily_shares=enr_daily,
            feature_meta=enr_meta,
            cf_npi_csv=CF_CAL,
            out_dir=CF_DIR_ORIGIN,
            poi_constraint_mode=args.poi_constraint_mode,
            phase_poi_rules=PHASE_POI_RULES,
            devices=args.devices,
            merge_workers=args.jobs,
            label="Origin baseline",
        )

    # 4b) Counterfactual simulation (supervised direct)
    cf_daily_sup = run_counterfactual(
        model_dir=SUP_DIR,
        daily_shares=enr_daily,
        feature_meta=enr_meta,
        cf_npi_csv=CF_CAL,
        out_dir=CF_DIR_SUP,
        poi_constraint_mode=args.poi_constraint_mode,
        phase_poi_rules=PHASE_POI_RULES,
        devices=args.devices,
        merge_workers=args.jobs,
        label="PreferenceNet-MoE",
    )

    # 4c) Counterfactual simulation (preference-aligned, one or more branches)
    aligned_cf_daily = {}
    for branch in ALIGN_RUNS:
        aligned_cf_daily[branch["name"]] = run_counterfactual(
            model_dir=aligned_dirs[branch["name"]],
            daily_shares=enr_daily,
            feature_meta=enr_meta,
            cf_npi_csv=CF_CAL,
            out_dir=branch["cf_dir"],
            poi_constraint_mode=args.poi_constraint_mode,
            phase_poi_rules=PHASE_POI_RULES,
            devices=args.devices,
            merge_workers=args.jobs,
            label=branch["label"],
        )

    poi_template_path = None
    obs_alloc_poi = None
    cf_alloc_origin = None
    cf_alloc_sup = None
    aligned_cf_alloc = {}
    if args.run_poi_seir:
        daily_geo = ENR_DIR / "daily_geo.csv"
        must_exist(daily_geo, "POI allocation templates require enriched daily_geo.csv.")
        poi_template_path = run_poi_template_build(
            daily_geo=daily_geo,
            daily_shares=enr_daily,
            feature_meta=enr_meta,
            out_path=POI_TEMPLATE,
            bucket_mode=args.poi_bucket_mode,
            topk=args.poi_topk,
            min_template_hours=args.poi_min_template_hours,
            min_template_days=args.poi_min_template_days,
            include_home=args.poi_include_home,
        )
        obs_alloc_poi = run_poi_allocation(
            daily_shares=enr_daily,
            templates=poi_template_path,
            feature_meta=enr_meta,
            out_path=OBS_ALLOC_POI,
            bucket_mode=args.poi_bucket_mode,
        )
        if args.run_origin_baseline and cf_daily_origin is not None:
            cf_alloc_origin = run_poi_allocation(
                daily_shares=cf_daily_origin,
                templates=poi_template_path,
                feature_meta=enr_meta,
                out_path=CF_DIR_ORIGIN / "allocated_poi_hours_cf.csv",
                bucket_mode=args.poi_bucket_mode,
            )
        cf_alloc_sup = run_poi_allocation(
            daily_shares=cf_daily_sup,
            templates=poi_template_path,
            feature_meta=enr_meta,
            out_path=CF_DIR_SUP / "allocated_poi_hours_cf.csv",
            bucket_mode=args.poi_bucket_mode,
        )
        for branch in ALIGN_RUNS:
            aligned_cf_alloc[branch["name"]] = run_poi_allocation(
                daily_shares=aligned_cf_daily[branch["name"]],
                templates=poi_template_path,
                feature_meta=enr_meta,
                out_path=branch["cf_dir"] / "allocated_poi_hours_cf.csv",
                bucket_mode=args.poi_bucket_mode,
            )

    # 5) SEIR (Observed)
    run_seir_from_daily(
        daily_shares=enr_daily,
        hazard_params=HAZARD,
        out_dir=OBS_SEIR,
        feature_meta=enr_meta,
        latent_days=args.latent_obs,
        infectious_days=args.infectious_obs,
        rng_seed=args.seed,
        workers=max(1, args.jobs // 2),
        mixing_scale=args.mixing_scale,
        home_beta_alpha=args.home_beta_alpha,
        importation_prob=args.importation_obs,
        seed_pct=args.seed_pct_obs,
        beta_scale=args.beta_scale_obs,
        label="Observed",
    )
    if args.run_poi_seir and obs_alloc_poi is not None:
        run_poi_seir_from_daily(
            daily_shares=enr_daily,
            allocated_poi_hours=obs_alloc_poi,
            hazard_params=HAZARD,
            out_dir=OBS_SEIR_POI,
            feature_meta=enr_meta,
            latent_days=args.latent_obs,
            infectious_days=args.infectious_obs,
            rng_seed=args.seed,
            workers=max(1, args.jobs // 2),
            importation_prob=args.importation_obs,
            seed_pct=args.seed_pct_obs,
            beta_scale=args.beta_scale_obs,
            label="Observed",
        )

    # 6) SEIR (Counterfactual, origin direct baseline)
    if args.run_origin_baseline and cf_daily_origin is not None:
        run_seir_from_daily(
            daily_shares=cf_daily_origin,
            hazard_params=HAZARD,
            out_dir=CF_SEIR_ORIGIN,
            feature_meta=enr_meta,
            latent_days=args.latent_cf,
            infectious_days=args.infectious_cf,
            rng_seed=args.seed,
            workers=args.jobs,
            mixing_scale=args.mixing_scale,
            home_beta_alpha=args.home_beta_alpha,
            importation_prob=args.importation_cf,
            seed_pct=args.seed_pct_cf,
            beta_scale=args.beta_scale_cf,
            label="Origin baseline counterfactual",
        )
        if args.run_poi_seir and cf_alloc_origin is not None:
            run_poi_seir_from_daily(
                daily_shares=cf_daily_origin,
                allocated_poi_hours=cf_alloc_origin,
                hazard_params=HAZARD,
                out_dir=CF_SEIR_ORIGIN_POI,
                feature_meta=enr_meta,
                latent_days=args.latent_cf,
                infectious_days=args.infectious_cf,
                rng_seed=args.seed,
                workers=args.jobs,
                importation_prob=args.importation_cf,
                seed_pct=args.seed_pct_cf,
                beta_scale=args.beta_scale_cf,
                label="Origin baseline counterfactual",
            )

    # 6b) SEIR (Counterfactual, supervised direct)
    run_seir_from_daily(
        daily_shares=cf_daily_sup,
        hazard_params=HAZARD,
        out_dir=CF_SEIR_SUP,
        feature_meta=enr_meta,
        latent_days=args.latent_cf,
        infectious_days=args.infectious_cf,
        rng_seed=args.seed,
        workers=args.jobs,
        mixing_scale=args.mixing_scale,
        home_beta_alpha=args.home_beta_alpha,
        importation_prob=args.importation_cf,
        seed_pct=args.seed_pct_cf,
        beta_scale=args.beta_scale_cf,
        label="PreferenceNet-MoE counterfactual",
    )
    if args.run_poi_seir and cf_alloc_sup is not None:
        run_poi_seir_from_daily(
            daily_shares=cf_daily_sup,
            allocated_poi_hours=cf_alloc_sup,
            hazard_params=HAZARD,
            out_dir=CF_SEIR_SUP_POI,
            feature_meta=enr_meta,
            latent_days=args.latent_cf,
            infectious_days=args.infectious_cf,
            rng_seed=args.seed,
            workers=args.jobs,
            importation_prob=args.importation_cf,
            seed_pct=args.seed_pct_cf,
            beta_scale=args.beta_scale_cf,
            label="PreferenceNet-MoE counterfactual",
        )

    # 6c) SEIR (Counterfactual, aligned, one or more branches)
    for branch in ALIGN_RUNS:
        run_seir_from_daily(
            daily_shares=aligned_cf_daily[branch["name"]],
            hazard_params=HAZARD,
            out_dir=branch["cf_seir_dir"],
            feature_meta=enr_meta,
            latent_days=args.latent_cf,
            infectious_days=args.infectious_cf,
            rng_seed=args.seed,
            workers=args.jobs,
            mixing_scale=args.mixing_scale,
            home_beta_alpha=args.home_beta_alpha,
            importation_prob=args.importation_cf,
            seed_pct=args.seed_pct_cf,
            beta_scale=args.beta_scale_cf,
            label=f"{branch['label']} counterfactual",
        )
        if args.run_poi_seir:
            run_poi_seir_from_daily(
                daily_shares=aligned_cf_daily[branch["name"]],
                allocated_poi_hours=aligned_cf_alloc[branch["name"]],
                hazard_params=HAZARD,
                out_dir=branch["cf_seir_poi_dir"],
                feature_meta=enr_meta,
                latent_days=args.latent_cf,
                infectious_days=args.infectious_cf,
                rng_seed=args.seed,
                workers=args.jobs,
                importation_prob=args.importation_cf,
                seed_pct=args.seed_pct_cf,
                beta_scale=args.beta_scale_cf,
                label=f"{branch['label']} counterfactual",
            )

    print("\nPipeline completed (no comparison).")
    print(f" Enriched dir:               {ENR_DIR}")
    print(f" PreferenceNet-MoE dir:      {SUP_DIR}")
    if args.run_origin_baseline:
        print(f" PreferenceNet dir:          {ORIGIN_SUP_DIR}")
    if args.run_origin_baseline:
        print(f" CF dir (PreferenceNet):     {CF_DIR_ORIGIN}")
    print(f" CF dir (PreferenceNet-MoE): {CF_DIR_SUP}")
    print(f" Observed SEIR dir:          {OBS_SEIR}")
    if args.run_poi_seir:
        print(f" Observed POI-SEIR dir:      {OBS_SEIR_POI}")
        print(f" POI template table:         {POI_TEMPLATE}")
        print(f" Observed POI alloc:         {OBS_ALLOC_POI}")
    if args.run_origin_baseline:
        print(f" CF SEIR dir (PreferenceNet): {CF_SEIR_ORIGIN}")
        if args.run_poi_seir:
            print(f" CF POI-SEIR dir (PreferenceNet): {CF_SEIR_ORIGIN_POI}")
    print(f" CF SEIR dir (PreferenceNet-MoE): {CF_SEIR_SUP}")
    if args.run_poi_seir:
        print(f" CF POI-SEIR dir (PreferenceNet-MoE): {CF_SEIR_SUP_POI}")
    for branch in ALIGN_RUNS:
        print(f" {branch['label']} dir: {branch['align_dir']}")
        print(f" CF dir ({branch['label']}): {branch['cf_dir']}")
        print(f" CF SEIR dir ({branch['label']}): {branch['cf_seir_dir']}")
        if args.run_poi_seir:
            print(f" CF POI-SEIR dir ({branch['label']}): {branch['cf_seir_poi_dir']}")

if __name__ == "__main__":
    main()
