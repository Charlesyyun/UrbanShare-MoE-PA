from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DIR = os.getenv("DSO_DIR", "100")
OUTPUT_DIR = ROOT / "Results" / "out" / "dashboard_eval" / DIR

WINDOW_DAYS = 7
LAG_WINDOW_DAYS = 14
EPS = 1e-8


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    column: str
    label: str
    family: str


SERIES_SPECS: tuple[SeriesSpec, ...] = (
    SeriesSpec("travel_frac", "travel_frac", "travel fraction", "mobility"),
    SeriesSpec("cat_11", "cat_11", "home share", "poi"),
    SeriesSpec("cat_7", "cat_7", "retail share", "poi"),
    SeriesSpec("cat_4", "cat_4", "dining share", "poi"),
    SeriesSpec("mode_2", "mode_2", "car share", "mode"),
    SeriesSpec("mode_3", "mode_3", "mrt share", "mode"),
    SeriesSpec("mode_4", "mode_4", "bus share", "mode"),
)


def build_model_paths() -> dict[str, Path]:
    model_paths: dict[str, Path] = {
        "PreferenceNet direct CF": ROOT / "Results" / "out" / "OLD" / DIR / "cf2_origin" / "phase_shift_dynamic_mgpu" / "daily_shares_cf.csv",
        "PreferenceNet-MoE direct CF": ROOT / "Results" / "out" / "OLD" / DIR / "cf2_supervised" / "phase_shift_dynamic_mgpu" / "daily_shares_cf.csv",
    }
    pa_factual = ROOT / "Results" / "out" / "OLD" / DIR / "cf2_factual" / "phase_shift_dynamic_mgpu" / "daily_shares_cf.csv"
    pa_counterfactual = ROOT / "Results" / "out" / "OLD" / DIR / "cf2_counterfactual" / "phase_shift_dynamic_mgpu" / "daily_shares_cf.csv"
    pa_legacy = ROOT / "Results" / "out" / "OLD" / DIR / "cf2" / "phase_shift_dynamic_mgpu" / "daily_shares_cf.csv"
    if pa_factual.exists():
        model_paths["PreferenceNet-MoE+PA factual CF"] = pa_factual
    if pa_counterfactual.exists():
        model_paths["PreferenceNet-MoE+PA counterfactual CF"] = pa_counterfactual
    elif pa_legacy.exists():
        model_paths["PreferenceNet-MoE+PA CF"] = pa_legacy
    return model_paths

OBSERVED_PATH = ROOT / "Results" / "runs" / f"enriched_{DIR}" / "daily_shares.csv"
ACTUAL_CALENDAR_PATH = ROOT / "dataset" / "MetaData" / "npi_calendar.csv"
CF_CALENDAR_PATH = ROOT / "dataset" / "MetaData" / "npi_calendar_early_lockdown_7d.csv"


def load_population_daily(path: Path, specs: Iterable[SeriesSpec]) -> pd.DataFrame:
    usecols = ["date"] + [spec.column for spec in specs]
    df = pd.read_csv(path, usecols=usecols)
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date", as_index=False)[[spec.column for spec in specs]].mean()
    return daily.sort_values("date").reset_index(drop=True)


def load_transitions(path: Path) -> pd.DataFrame:
    cal = pd.read_csv(path)
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal.sort_values("date").reset_index(drop=True)
    cal["prev_phase_id"] = cal["phase_id"].shift()
    cal["prev_phase_name"] = cal["phase_name"].shift()
    changes = cal.loc[cal["phase_id"].ne(cal["prev_phase_id"])].copy()
    changes = changes.loc[changes["prev_phase_id"].notna()].copy()
    changes["transition"] = (
        changes["prev_phase_name"].astype(str)
        + "->"
        + changes["phase_name"].astype(str)
    )
    return changes[
        ["date", "prev_phase_id", "prev_phase_name", "phase_id", "phase_name", "transition"]
    ].reset_index(drop=True)


def window_mean(series: pd.Series) -> float:
    return float(series.mean()) if not series.empty else np.nan


def window_std(series: pd.Series) -> float:
    return float(series.std(ddof=0)) if not series.empty else np.nan


def sign_or_zero(x: float, tol: float = 1e-5) -> int:
    if pd.isna(x) or abs(x) <= tol:
        return 0
    return 1 if x > 0 else -1


def compute_transition_metrics(
    daily: pd.DataFrame,
    transition_date: pd.Timestamp,
    column: str,
    window_days: int = WINDOW_DAYS,
    lag_window_days: int = LAG_WINDOW_DAYS,
) -> dict[str, float]:
    date_to_value = daily.set_index("date")[column].sort_index()

    pre_prev_start = transition_date - pd.Timedelta(days=2 * window_days)
    pre_prev_end = transition_date - pd.Timedelta(days=window_days + 1)
    pre_start = transition_date - pd.Timedelta(days=window_days)
    pre_end = transition_date - pd.Timedelta(days=1)
    post_end = transition_date + pd.Timedelta(days=window_days - 1)
    lag_end = transition_date + pd.Timedelta(days=lag_window_days - 1)

    pre_prev = date_to_value.loc[pre_prev_start:pre_prev_end]
    pre = date_to_value.loc[pre_start:pre_end]
    post = date_to_value.loc[transition_date:post_end]
    post_long = date_to_value.loc[transition_date:lag_end]

    pre_mean = window_mean(pre)
    post_mean = window_mean(post)
    delta = post_mean - pre_mean if pd.notna(pre_mean) and pd.notna(post_mean) else np.nan
    leakage = (
        pre_mean - window_mean(pre_prev)
        if pd.notna(pre_mean) and not pre_prev.empty
        else np.nan
    )
    sharpness = (
        abs(delta) / (window_std(pre) + EPS)
        if pd.notna(delta) and not pre.empty
        else np.nan
    )

    lag_50 = np.nan
    if pd.notna(pre_mean) and not post_long.empty:
        deviations = (post_long - pre_mean).abs().to_numpy()
        max_dev = float(np.nanmax(deviations)) if len(deviations) else np.nan
        if pd.notna(max_dev) and max_dev > EPS:
            threshold = 0.5 * max_dev
            idx = np.where(deviations >= threshold)[0]
            if len(idx) > 0:
                lag_50 = int(idx[0])

    return {
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "delta": delta,
        "abs_delta": abs(delta) if pd.notna(delta) else np.nan,
        "pre_policy_leakage": leakage,
        "abs_pre_policy_leakage": abs(leakage) if pd.notna(leakage) else np.nan,
        "sharpness": sharpness,
        "lag50_days": lag_50,
    }


def build_reference_responses(
    observed_daily: pd.DataFrame,
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, tr in transitions.iterrows():
        for spec in SERIES_SPECS:
            metrics = compute_transition_metrics(observed_daily, tr["date"], spec.column)
            rows.append(
                {
                    "transition": tr["transition"],
                    "transition_date": tr["date"].date().isoformat(),
                    "series": spec.key,
                    "label": spec.label,
                    "family": spec.family,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_paths = build_model_paths()

    observed_daily = load_population_daily(OBSERVED_PATH, SERIES_SPECS)
    actual_transitions = load_transitions(ACTUAL_CALENDAR_PATH)
    cf_transitions = load_transitions(CF_CALENDAR_PATH)

    reference = build_reference_responses(observed_daily, actual_transitions)
    reference.to_csv(OUTPUT_DIR / "policy_response_reference_observed.csv", index=False)

    ref_map = reference.set_index(["transition", "series"])

    detail_rows: list[dict[str, object]] = []
    for model_name, path in model_paths.items():
        if not path.exists():
            print(f"[skip] missing model daily shares: {path}")
            continue
        model_daily = load_population_daily(path, SERIES_SPECS)
        for _, tr in cf_transitions.iterrows():
            for spec in SERIES_SPECS:
                metrics = compute_transition_metrics(model_daily, tr["date"], spec.column)
                ref = ref_map.loc[(tr["transition"], spec.key)]
                ref_delta = float(ref["delta"])
                ratio = np.nan
                if pd.notna(metrics["delta"]) and abs(ref_delta) > EPS:
                    ratio = float(metrics["delta"]) / ref_delta

                detail_rows.append(
                    {
                        "model": model_name,
                        "transition": tr["transition"],
                        "transition_date_cf": tr["date"].date().isoformat(),
                        "reference_date_observed": ref["transition_date"],
                        "series": spec.key,
                        "label": spec.label,
                        "family": spec.family,
                        **metrics,
                        "observed_delta_same_transition": ref_delta,
                        "observed_abs_delta_same_transition": float(ref["abs_delta"]),
                        "direction_match_vs_observed": int(
                            sign_or_zero(float(metrics["delta"])) == sign_or_zero(ref_delta)
                        ),
                        "magnitude_ratio_vs_observed": ratio,
                        "magnitude_error_vs_observed": (
                            abs(abs(float(metrics["delta"])) - float(ref["abs_delta"]))
                            if pd.notna(metrics["delta"])
                            else np.nan
                        ),
                    }
                )

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(OUTPUT_DIR / "policy_response_metrics_detail.csv", index=False)

    summary = (
        detail.groupby("model", as_index=False)
        .agg(
            n_series=("series", "count"),
            mean_abs_response=("abs_delta", "mean"),
            mean_direction_match=("direction_match_vs_observed", "mean"),
            mean_abs_pre_policy_leakage=("abs_pre_policy_leakage", "mean"),
            mean_lag50_days=("lag50_days", "mean"),
            mean_sharpness=("sharpness", "mean"),
            mean_abs_magnitude_error_vs_observed=("magnitude_error_vs_observed", "mean"),
        )
    )
    summary["mean_direction_match"] = summary["mean_direction_match"] * 100.0
    summary.to_csv(OUTPUT_DIR / "policy_response_metrics_summary.csv", index=False)

    by_transition = (
        detail.groupby(["model", "transition"], as_index=False)
        .agg(
            mean_abs_response=("abs_delta", "mean"),
            mean_direction_match=("direction_match_vs_observed", "mean"),
            mean_abs_pre_policy_leakage=("abs_pre_policy_leakage", "mean"),
            mean_lag50_days=("lag50_days", "mean"),
            mean_sharpness=("sharpness", "mean"),
            mean_abs_magnitude_error_vs_observed=("magnitude_error_vs_observed", "mean"),
        )
    )
    by_transition["mean_direction_match"] = by_transition["mean_direction_match"] * 100.0
    by_transition.to_csv(OUTPUT_DIR / "policy_response_metrics_by_transition.csv", index=False)

    print("Saved:")
    print(OUTPUT_DIR / "policy_response_reference_observed.csv")
    print(OUTPUT_DIR / "policy_response_metrics_detail.csv")
    print(OUTPUT_DIR / "policy_response_metrics_summary.csv")
    print(OUTPUT_DIR / "policy_response_metrics_by_transition.csv")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
