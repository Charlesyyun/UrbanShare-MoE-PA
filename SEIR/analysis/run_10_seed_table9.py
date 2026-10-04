from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEIR = ROOT / "seir_timeseries.py"
POLICIES = {
    "Original policy": "original_policy",
    "Early lockdown": "early_lockdown",
    "Late lockdown": "late_lockdown",
    "Short lockdown": "short_lockdown",
    "Long lockdown": "long_lockdown",
}
SEEDS = list(range(1234, 1244))
METRICS = ["peak_I", "peak_new_E_7dma", "final_R", "area_I", "area_new_E", "duration_I_gt_1"]
OUTPUT_NAMES = {
    "peak_I": "Peak I",
    "peak_new_E_7dma": "Peak new E (7d centered mean)",
    "final_R": "Final R",
    "area_I": "Area(I)",
    "area_new_E": "Area(new E)",
    "duration_I_gt_1": "Epidemic duration (I>1 days)",
}


def metric_row(csv: Path, policy: str, seed: int) -> dict:
    df = pd.read_csv(csv)
    smooth = df["new_E"].rolling(7, center=True, min_periods=1).mean()
    return {
        "policy": policy,
        "seed": seed,
        "peak_I": float(df["I"].max()),
        "peak_new_E_7dma": float(smooth.max()),
        "final_R": float(df["R"].iloc[-1]),
        "area_I": float(df["I"].sum()),
        "area_new_E": float(df["new_E"].sum()),
        "duration_I_gt_1": int((df["I"] > 1).sum()),
    }


def write_markdown(table: pd.DataFrame, path: Path) -> None:
    columns = list(table.columns)
    rows = [[str(value) if isinstance(value, str) else f"{value:.2f}" for value in row]
            for row in table.itertuples(index=False, name=None)]
    widths = [max(len(columns[i]), *(len(row[i]) for row in rows)) for i in range(len(columns))]
    header = "| " + " | ".join(columns[i].ljust(widths[i]) for i in range(len(columns))) + " |"
    rule = "|:" + "|:".join("-" * (width + 1) for width in widths) + "|"
    body = ["| " + " | ".join(row[i].rjust(widths[i]) for i in range(len(columns))) + " |" for row in rows]
    path.write_text("\n".join([header, rule, *body]) + "\n", encoding="utf-8")


def run(policy: str, seed: int, manifest: dict | None, rerun: bool, out: Path, hazard: Path, meta: Path, seed_list: Path, table9_components: Path) -> dict:
    run_dir = out / "runs" / POLICIES[policy] / f"seed_{seed}"
    csv = run_dir / "seir_timeseries.csv"
    if rerun or not csv.exists():
        if manifest is None:
            raise FileNotFoundError(
                f"Missing cached run {csv}. To simulate, pass --rerun and a "
                "--daily-shares-manifest with the private daily-share input paths."
            )
        daily = manifest["daily_shares"][policy]
        reference = manifest["crowding_reference_daily_shares"]
        cmd = [
            sys.executable, str(SEIR), "--daily_shares", str(Path(daily).expanduser()),
            "--hazard_params", str(hazard), "--feature_meta", str(meta),
            "--out_dir", str(run_dir), "--latent_days", "3", "--infectious_days", "7",
            "--seed_pct", "0.002", "--seed_list", str(seed_list), "--rng_seed", str(seed),
            "--workers", "1", "--mixing_scale", "24h", "--beta_scale", "1.8055325359443612",
            "--home_beta_alpha", "0.211730151512386", "--home_protection_scope", "population",
            "--crowding_eta", "1.0", "--crowding_reference_daily_shares", str(Path(reference).expanduser()),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
    return metric_row(csv, policy, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the ten-seed SEIR metrics and Table 9.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hazard-params", type=Path, required=True)
    parser.add_argument("--feature-meta", type=Path, required=True)
    parser.add_argument("--seed-list", type=Path, required=True)
    parser.add_argument("--table9-components", type=Path, required=True)
    parser.add_argument(
        "--rerun", action="store_true",
        help="rerun all simulations; requires --daily-shares-manifest",
    )
    parser.add_argument(
        "--daily-shares-manifest", type=Path,
        help="JSON mapping each calendar and the crowding reference to local input CSV paths",
    )
    args = parser.parse_args()
    if args.rerun and not args.daily_shares_manifest:
        parser.error("--rerun requires --daily-shares-manifest")
    manifest = None
    if args.daily_shares_manifest:
        with args.daily_shares_manifest.open(encoding="utf-8") as f:
            manifest = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run(policy, seed, manifest, args.rerun, args.output_dir, args.hazard_params, args.feature_meta, args.seed_list, args.table9_components) for seed in SEEDS for policy in POLICIES]
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_dir / "raw_metrics.csv", index=False)
    if raw.duplicated(["policy", "seed"]).any() or len(raw) != len(POLICIES) * len(SEEDS):
        raise ValueError("Expected exactly one result for every policy and seed pair.")

    summary = raw.groupby("policy")[METRICS].agg(["mean", "std", "min", "max"])
    summary.to_csv(args.output_dir / "epidemic_summary_by_calendar.csv")

    economic = pd.read_csv(args.table9_components)
    economic = economic.set_index("policy_key")["mobility_augmented_output_all_travel_w031"]
    policy_keys = {
        "Original policy": "factual", "Early lockdown": "early", "Late lockdown": "late",
        "Short lockdown": "short", "Long lockdown": "long",
    }
    table = pd.DataFrame({
        "Scenario": list(POLICIES),
        "O": [float(economic[policy_keys[p]]) for p in POLICIES],
    })
    original_o = table.loc[table["Scenario"] == "Original policy", "O"].iloc[0]
    table["Change O vs original (%)"] = 100 * (table["O"] - original_o) / original_o
    for metric, label in OUTPUT_NAMES.items():
        table[f"{label} mean"] = [summary.loc[p, (metric, "mean")] for p in POLICIES]
        table[f"{label} sd"] = [summary.loc[p, (metric, "std")] for p in POLICIES]
    for label in OUTPUT_NAMES.values():
        col = f"{label} mean"
        original = table.loc[table["Scenario"] == "Original policy", col].iloc[0]
        table[f"Change {col} vs original (%)"] = 100 * (table[col] - original) / original

    keep = ["Scenario", "O", "Change O vs original (%)"]
    for label in OUTPUT_NAMES.values():
        keep.extend([f"{label} mean", f"{label} sd"])
    table[keep].to_csv(args.output_dir / "table9_10_seed_summary.csv", index=False)
    write_markdown(table[keep], args.output_dir / "table9_10_seed_summary.md")
    print(table[keep].to_string(index=False))


if __name__ == "__main__":
    main()
