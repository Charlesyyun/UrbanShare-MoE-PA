"""Rebuild the policy-calendar figure in the original 2x4 Figure 12 format."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {
    "early": ("Early lockdown", "2020-03-31", "2020-05-25"),
    "late": ("Late lockdown", "2020-04-14", "2020-06-08"),
    "short": ("Short lockdown", "2020-04-07", "2020-05-18"),
    "long": ("Long lockdown", "2020-04-07", "2020-06-15"),
}
MODELS = {
    "urbanshare": ("UrbanShare", "#56B4E9", "--", 1.75),
    "urbanshare_moe": ("UrbanShare-MoE", "#009E73", "-.", 1.75),
}
PA_STYLE = ("UrbanShare-MoE-PA (10-seed mean)", "#C0392B", "-", 1.9)


def read_seir(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["new_E_7dma"] = df["new_E"].rolling(7, center=True, min_periods=1).mean()
    return df


def read_pa_seeds(results: Path, policy: str) -> pd.DataFrame:
    rows = []
    for path in sorted((results / "runs" / f"{policy}_lockdown").glob("seed_*/seir_timeseries.csv")):
        df = read_seir(path)
        rows.append(df[["date", "I", "new_E_7dma"]])
    if len(rows) != 10:
        raise ValueError(f"{policy}: expected 10 PA seed runs, found {len(rows)}")
    long = pd.concat(rows, ignore_index=True)
    return long.groupby("date")[["I", "new_E_7dma"]].agg(["mean", "std"]).reset_index()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta-dir", type=Path, default=ROOT / "results" / "policy_calendar_eta1_reproduction" / "inputs" / "seir")
    parser.add_argument("--random-results", type=Path, default=ROOT / "results" / "policy_calendar_eta1_reproduction" / "random_seed_table9")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args(argv)

    observed = read_seir(args.eta_dir / "observed_reference" / "seir_timeseries.csv")
    model_data = {}
    pa_data = {}
    for policy in POLICIES:
        model_data[policy] = {key: read_seir(args.eta_dir / policy / key / "seir_timeseries.csv") for key in MODELS}
        pa_data[policy] = read_pa_seeds(args.random_results, policy)

    fig, axes = plt.subplots(2, 4, figsize=(12.6, 5.9), sharex=True, gridspec_kw={"hspace": 0.24, "wspace": 0.18})
    row_specs = [("I", "Active infectious"), ("new_E_7dma", "New exposure\n(7-day mean)")]
    for row, (metric, ylabel) in enumerate(row_specs):
        max_y = float(observed[metric].max())
        for policy in POLICIES:
            for df in model_data[policy].values():
                max_y = max(max_y, float(df[metric].max()))
            stats = pa_data[policy]
            max_y = max(max_y, float((stats[(metric, "mean")] + stats[(metric, "std")].fillna(0)).max()))
        for col, (policy, (title, start, end)) in enumerate(POLICIES.items()):
            ax = axes[row, col]
            ax.plot(observed["date"], observed[metric], color="#1F2937", lw=2.1)
            for key, (label, color, ls, lw) in MODELS.items():
                df = model_data[policy][key]
                ax.plot(df["date"], df[metric], color=color, ls=ls, lw=lw)
            stats = pa_data[policy]
            mean = stats[(metric, "mean")]
            sd = stats[(metric, "std")].fillna(0.0)
            dates = stats["date"]
            ax.fill_between(dates, (mean - sd).clip(lower=0), mean + sd, color=PA_STYLE[1], alpha=0.15, linewidth=0)
            ax.plot(dates, mean, color=PA_STYLE[1], ls=PA_STYLE[2], lw=PA_STYLE[3])
            for day in (start, end):
                ax.axvline(pd.Timestamp(day), color="#C0392B", lw=1.0, alpha=0.70)
            for day in ("2020-04-07", "2020-06-01"):
                ax.axvline(pd.Timestamp(day), color="#4B5563", lw=1.0, ls=(0, (4, 3)), alpha=0.75)
            ax.set_ylim(0, max_y * 1.10 if max_y > 0 else 1)
            ax.grid(axis="y", color="#E5E7EB", lw=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.tick_params(axis="both", labelsize=8)
            if row == 0:
                ax.set_title(title, fontsize=10.5, pad=8)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)

    handles = [
        Line2D([], [], color="#1F2937", lw=2.1, label="Observed factual"),
        Line2D([], [], color=MODELS["urbanshare"][1], lw=1.75, ls="--", label="UrbanShare"),
        Line2D([], [], color=MODELS["urbanshare_moe"][1], lw=1.75, ls="-.", label="UrbanShare-MoE"),
        Line2D([], [], color=PA_STYLE[1], lw=PA_STYLE[3], label=PA_STYLE[0]),
        Line2D([], [], color="#4B5563", lw=1.0, ls=(0, (4, 3)), label="Observed lockdown window"),
        Line2D([], [], color="#C0392B", lw=1.0, label="Counterfactual lockdown window"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=6, frameon=False, fontsize=8.4)
    fig.supxlabel("Date (2020)", y=0.04, fontsize=10)
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.07, right=0.99)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / f"fig15_seir_policy_compare_four_calendars_10seed.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
