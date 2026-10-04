"""Run the packaged behavior trajectories through the SEIR simulator.

Outputs are written below ``SEIR/results/scenarios`` by default and are ignored
by Git. The script uses only files included in this release.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SEIR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SEIR_ROOT.parent
if str(SEIR_ROOT) not in sys.path:
    sys.path.insert(0, str(SEIR_ROOT))

from seir_timeseries import SEIRTimeseries

POLICIES = ("early", "late", "short", "long")
MODELS = ("urbanshare", "urbanshare_moe", "urbanshare_moe_pa")


def _csv_list(value: str, allowed: tuple[str, ...]) -> list[str]:
    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(selected).difference(allowed))
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported value(s): {', '.join(invalid)}")
    return selected


def _load_behavior(policy: str, model: str) -> pd.DataFrame:
    behavior_path = REPO_ROOT / policy / "behavior" / f"{model}.csv.gz"
    calendar_path = REPO_ROOT / policy / "calendar.csv"
    behavior = pd.read_csv(behavior_path)
    calendar = pd.read_csv(calendar_path, usecols=["date", "phase_id"]).rename(
        columns={"phase_id": "epi_phase"}
    )
    behavior["date"] = pd.to_datetime(behavior["date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    behavior = behavior.drop(columns=["epi_phase"], errors="ignore").merge(
        calendar, on="date", how="left", validate="many_to_one"
    )
    if behavior["epi_phase"].isna().any():
        raise ValueError(f"calendar does not cover every date for policy={policy}")
    behavior["epi_phase"] = behavior["epi_phase"].astype(int)
    behavior["date"] = behavior["date"].dt.date
    return behavior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--hazard-params", type=Path, default=SEIR_ROOT / "results" / "hazard_params_recalibrated_2026.json")
    parser.add_argument("--feature-meta", type=Path, default=REPO_ROOT / "data" / "behavior" / "feature_meta.json")
    parser.add_argument("--output-dir", type=Path, default=SEIR_ROOT / "results" / "scenarios")
    parser.add_argument("--latent-days", type=int, default=3)
    parser.add_argument("--infectious-days", type=int, default=7)
    parser.add_argument("--seed-pct", type=float, default=0.002)
    parser.add_argument("--rng-seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    policies = _csv_list(args.policies, POLICIES)
    models = _csv_list(args.models, MODELS)
    feature_meta = json.loads(args.feature_meta.read_text(encoding="utf-8"))
    hazard = json.loads(args.hazard_params.read_text(encoding="utf-8"))
    candidate = json.loads((SEIR_ROOT / "results" / "best_candidate.json").read_text(encoding="utf-8"))

    summaries: list[dict[str, object]] = []
    for policy in policies:
        for model in models:
            out_dir = args.output_dir / policy / model
            result_path = out_dir / "seir_timeseries.csv"
            if args.skip_existing and result_path.is_file():
                result = pd.read_csv(result_path)
            else:
                simulator = SEIRTimeseries(
                    daily_df=_load_behavior(policy, model),
                    feature_meta=feature_meta,
                    hazard_params=hazard,
                    latent_days=args.latent_days,
                    infectious_days=args.infectious_days,
                    seed_pct=args.seed_pct,
                    seed_agent_ids=None,
                    rng_seed=args.rng_seed,
                    n_workers=max(1, args.workers),
                )
                simulator.mixing_scale = "24h"
                simulator.home_beta_alpha = float(candidate.get("home_beta_alpha", 0.0))
                simulator.phase_beta_mult = {int(key): float(value) for key, value in candidate.get("phase_beta_mult", {}).items()}
                out_dir.mkdir(parents=True, exist_ok=True)
                simulator.run(out_dir=out_dir, rng_seed=args.rng_seed)
                result = pd.read_csv(result_path)

            summaries.append({
                "policy": policy,
                "model": model,
                "peak_I": float(result["I"].max()),
                "final_R": float(result["R"].iloc[-1]),
                "area_I": float(result["I"].sum()),
                "area_new_E": float(result["new_E"].sum()),
            })
            print(f"[ok] {policy}/{model}: {result_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
