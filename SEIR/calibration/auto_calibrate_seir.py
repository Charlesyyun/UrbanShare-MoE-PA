from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEIR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SEIR_ROOT.parent
if str(SEIR_ROOT) not in sys.path:
    sys.path.insert(0, str(SEIR_ROOT))

from seir_timeseries import SEIRTimeseries, _load_feature_meta

ROOT = REPO_ROOT
# All defaults refer to user-supplied files in the current working directory.
DEFAULT_DAILY_SHARES = Path("daily_shares.csv")
DEFAULT_FEATURE_META = Path("feature_meta.json")
# The official curve is a user-supplied external input, not a repository file.
OFFICIAL_CURVE_HTML = Path("official_curve.html")
DEFAULT_BASE_HAZARD = Path("hazard_params.json")
DEFAULT_OUT_DIR = Path("seir_calibration_output")
DEFAULT_SEED_LIST = DEFAULT_OUT_DIR / "seed_agents.csv"
DEFAULT_SCRATCH_DIR = DEFAULT_OUT_DIR / "_scratch"

STUDY_START = pd.Timestamp("2020-03-01")
STUDY_END = pd.Timestamp("2020-08-31")
INCIDENCE_LEAD_DAYS = 21
POST_PEAK_OFFSET_DAYS = 14
TAIL_START = pd.Timestamp("2020-07-01")
PHASE_WINDOWS = {
    "pre-NPI": ("2020-03-01", "2020-04-06"),
    "lockdown": ("2020-04-07", "2020-06-01"),
    "phase 1": ("2020-06-02", "2020-06-21"),
    "phase 2": ("2020-06-22", "2020-08-31"),
}
PHASE_NAME_TO_ID = {"pre-NPI": 0, "lockdown": 1, "phase 1": 2, "phase 2": 3}

SEARCH_BOUNDS_PROFILES: dict[str, dict[str, tuple[float, float, str]]] = {
    "default_v1": {
        "beta_scale": (0.90, 1.80, "linear"),
        "home_beta_alpha": (0.00, 0.45, "linear"),
        "importation_prob": (2e-5, 5e-4, "log"),
        "phase_beta_0": (0.90, 1.25, "linear"),
        "phase_beta_1": (0.20, 0.65, "linear"),
        "phase_beta_2": (0.40, 0.90, "linear"),
        "phase_beta_3": (0.50, 1.00, "linear"),
        "phase_import_1": (1.00, 8.00, "linear"),
        "phase_import_2": (0.40, 2.00, "linear"),
        "phase_import_3": (0.50, 4.00, "linear"),
    },
    "expanded_v2": {
        "beta_scale": (0.70, 1.80, "linear"),
        "home_beta_alpha": (0.00, 0.45, "linear"),
        "importation_prob": (2e-5, 1.5e-3, "log"),
        "phase_beta_0": (0.90, 1.25, "linear"),
        "phase_beta_1": (0.20, 0.75, "linear"),
        "phase_beta_2": (0.35, 0.95, "linear"),
        "phase_beta_3": (0.50, 1.05, "linear"),
        "phase_import_1": (1.00, 12.00, "linear"),
        "phase_import_2": (0.40, 3.00, "linear"),
        "phase_import_3": (0.50, 6.00, "linear"),
    },
    "expanded_v3": {
        "beta_scale": (0.60, 2.10, "linear"),
        "home_beta_alpha": (0.00, 0.55, "linear"),
        "importation_prob": (1e-5, 2.5e-3, "log"),
        "phase_beta_0": (0.85, 1.45, "linear"),
        "phase_beta_1": (0.15, 0.95, "linear"),
        "phase_beta_2": (0.30, 1.05, "linear"),
        "phase_beta_3": (0.35, 1.15, "linear"),
        "phase_import_1": (1.00, 14.00, "linear"),
        "phase_import_2": (0.30, 4.00, "linear"),
        "phase_import_3": (0.40, 7.00, "linear"),
    },
}

OFFICIAL_IMPORT_SIGNALS = {
    "official_active_cases",
    "official_active_cases_7dma",
    "reported_new_cases",
    "reported_cases_7dma",
    "official_total_cases",
}


def _is_seed_mode(importation_mode: str) -> bool:
    return importation_mode in {"official_curve_seed"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_html_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols: list[str] = []
    for col in df.columns:
        if isinstance(col, tuple):
            cols.append(" | ".join(str(x) for x in col if str(x) != "nan"))
        else:
            cols.append(str(col))
    out = df.copy()
    out.columns = cols
    return out


def _as_float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().replace(",", "")
    if text in {"-", "—", "nan", "None", ""}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _as_official_count(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().replace(",", "")
    if text in {"-", "—", "nan", "None", ""}:
        return float("nan")
    if text.count(".") == 1:
        left, right = text.split(".", 1)
        if left.isdigit() and right.isdigit() and len(right) == 3:
            text = left + right
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _smooth(series: pd.Series, window: int = 7) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _build_seed_list(seed_list_path: Path, daily_shares_path: Path, seed_pct: float, rng_seed: int) -> list[str]:
    if seed_list_path.exists():
        tmp = pd.read_csv(seed_list_path)
        if "agent_id" not in tmp.columns:
            raise ValueError(f"{seed_list_path} must contain an 'agent_id' column.")
        return tmp["agent_id"].astype(str).tolist()

    df = pd.read_csv(daily_shares_path, usecols=["agent_id"])
    agents = sorted(df["agent_id"].astype(str).unique().tolist())
    rng = np.random.default_rng(rng_seed)
    k = max(1, int(round(seed_pct * len(agents))))
    picked = sorted(rng.choice(agents, size=k, replace=False).tolist())
    seed_list_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"agent_id": picked}).to_csv(seed_list_path, index=False)
    return picked


def _load_official_curve(html_path: Path) -> pd.DataFrame:
    if not html_path.exists():
        raise FileNotFoundError(f"Missing archived Singapore curve source: {html_path}")

    month_ids = [f"2020-{month:02d}" for month in range(1, 13)]
    tables = pd.read_html(str(html_path))
    rows: list[dict[str, Any]] = []

    for month_str, table in zip(month_ids, tables[:12]):
        df = _flatten_html_columns(table)
        day_cols = [c for c in df.columns if c.startswith("Day")]
        new_cols = [c for c in df.columns if "New cases" in c and c.endswith("Total")]
        active_cols = [c for c in df.columns if "Active cases" in c]
        total_cols = [c for c in df.columns if "Total cases" in c]
        if not day_cols or not new_cols or not active_cols or not total_cols:
            continue

        day_col = day_cols[0]
        new_col = new_cols[-1]
        active_col = active_cols[-1]
        total_col = total_cols[-1]

        for _, row in df.iterrows():
            day = _as_float(row[day_col])
            if np.isnan(day):
                continue
            rows.append(
                {
                    "date": pd.Timestamp(f"{month_str}-{int(day):02d}"),
                    "reported_new_cases": _as_official_count(row[new_col]),
                    "official_active_cases": _as_official_count(row[active_col]),
                    "official_total_cases": _as_official_count(row[total_col]),
                }
            )

    official = pd.DataFrame(rows).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    official["reported_new_cases"] = pd.to_numeric(official["reported_new_cases"], errors="coerce").fillna(0.0)
    official["official_active_cases"] = (
        pd.to_numeric(official["official_active_cases"], errors="coerce").interpolate().bfill().ffill()
    )
    official["official_total_cases"] = (
        pd.to_numeric(official["official_total_cases"], errors="coerce").interpolate().bfill().ffill()
    )
    official["official_active_cases_7dma"] = _smooth(official["official_active_cases"], 7)
    official["reported_cases_7dma"] = _smooth(official["reported_new_cases"], 7)
    return official


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    value = float(a.corr(b))
    if math.isnan(value):
        return -1.0
    return max(-1.0, min(1.0, value))


def _peak_mass_ratio(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    total = max(float(arr.sum()), 1e-8)
    return float(arr.max()) / total


def _share_after(df: pd.DataFrame, value_col: str, start_date: pd.Timestamp) -> float:
    values = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    total = max(float(values.sum()), 1e-8)
    return float(values[df["date"] >= start_date].sum()) / total


def _ratio_at_offset(df: pd.DataFrame, value_col: str, peak_date: pd.Timestamp, offset_days: int) -> float:
    target_date = peak_date + pd.Timedelta(days=offset_days)
    peak_mask = df["date"] == peak_date
    if not peak_mask.any():
        peak_value = max(float(pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).max()), 1e-8)
    else:
        peak_value = max(float(pd.to_numeric(df.loc[peak_mask, value_col], errors="coerce").fillna(0.0).iloc[0]), 1e-8)
    after = df[df["date"] >= target_date]
    if after.empty:
        current_value = float(pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).iloc[-1])
    else:
        current_value = float(pd.to_numeric(after.iloc[0][value_col], errors="coerce"))
    return current_value / peak_value


def _candidate_from_manual(
    *,
    name: str,
    beta_scale: float,
    home_beta_alpha: float,
    importation_prob: float,
    phase_beta_mult: dict[str, float] | dict[int, float],
    phase_importation_mult: dict[str, float] | dict[int, float] | None = None,
) -> dict[str, Any]:
    phase_import = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    if phase_importation_mult:
        phase_import.update({int(k): float(v) for k, v in phase_importation_mult.items()})
    return {
        "name": name,
        "beta_scale": float(beta_scale),
        "home_beta_alpha": float(home_beta_alpha),
        "importation_prob": float(importation_prob),
        "phase_beta_mult": {int(k): float(v) for k, v in phase_beta_mult.items()},
        "phase_importation_mult": phase_import,
    }


def _clamp_with_bounds(
    key: str,
    value: float,
    search_bounds: dict[str, tuple[float, float, str]],
) -> float:
    low, high, _ = search_bounds[key]
    return float(min(max(value, low), high))


def _project_monotonic_phase_beta(
    phase_beta_mult: dict[int, float] | dict[str, float],
    search_bounds: dict[str, tuple[float, float, str]],
    *,
    tie_phase2_to_pre: bool = False,
) -> dict[int, float]:
    vals = {int(k): float(v) for k, v in phase_beta_mult.items()}
    b1 = _clamp_with_bounds("phase_beta_1", vals.get(1, 0.4), search_bounds)
    b2 = _clamp_with_bounds("phase_beta_2", max(vals.get(2, 0.65), b1), search_bounds)
    if tie_phase2_to_pre:
        b0 = _clamp_with_bounds("phase_beta_0", max(vals.get(0, 1.10), b2), search_bounds)
        b3 = b0
    else:
        b3 = _clamp_with_bounds("phase_beta_3", max(vals.get(3, 0.78), b2), search_bounds)
        b0 = _clamp_with_bounds("phase_beta_0", max(vals.get(0, 1.10), b3), search_bounds)

    # Enforce the intuitive policy ordering:
    # lockdown <= phase 1 <= phase 2 <= pre-NPI.
    b2 = max(b2, b1)
    if tie_phase2_to_pre:
        b0 = max(b0, b2)
        b3 = b0
    else:
        b3 = max(b3, b2)
        b0 = max(b0, b3)
    return {0: float(b0), 1: float(b1), 2: float(b2), 3: float(b3)}


def _normalize_candidate(
    candidate: dict[str, Any],
    *,
    search_bounds: dict[str, tuple[float, float, str]],
    importation_mode: str,
    enforce_monotonic_phase_beta: bool,
    tie_phase2_to_pre: bool,
) -> dict[str, Any]:
    out = copy.deepcopy(candidate)
    out["phase_beta_mult"] = {int(k): float(v) for k, v in out["phase_beta_mult"].items()}
    out["phase_importation_mult"] = {
        int(k): float(v) for k, v in out.get("phase_importation_mult", {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}).items()
    }

    if enforce_monotonic_phase_beta:
        out["phase_beta_mult"] = _project_monotonic_phase_beta(
            out["phase_beta_mult"],
            search_bounds,
            tie_phase2_to_pre=tie_phase2_to_pre,
        )
    elif tie_phase2_to_pre:
        out["phase_beta_mult"][3] = float(out["phase_beta_mult"][0])

    if importation_mode in {"constant", "official_curve", "official_curve_seed"}:
        out["phase_importation_mult"] = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

    return out


def _build_observed_date_importation_schedule(
    phase_importation_mult: dict[int, float] | dict[str, float],
) -> pd.DataFrame:
    phase_import = {int(k): float(v) for k, v in phase_importation_mult.items()}
    rows: list[dict[str, Any]] = []
    for phase_name, (start, end) in PHASE_WINDOWS.items():
        phase_id = PHASE_NAME_TO_ID[phase_name]
        mult = float(phase_import.get(phase_id, 1.0))
        for day in pd.date_range(start, end, freq="D"):
            rows.append(
                {
                    "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                    "observed_phase": phase_name,
                    "observed_phase_id": phase_id,
                    "importation_mult": mult,
                }
            )
    return pd.DataFrame(rows)


def _build_official_curve_importation_schedule(
    official: pd.DataFrame,
    *,
    signal_col: str,
    floor: float = 0.0,
) -> pd.DataFrame:
    if signal_col not in official.columns:
        raise ValueError(f"Unknown official importation signal: {signal_col}")

    sched = official[["date", signal_col]].copy()
    sched = sched[(sched["date"] >= STUDY_START) & (sched["date"] <= STUDY_END)].copy()
    sched["signal"] = pd.to_numeric(sched[signal_col], errors="coerce").fillna(0.0)
    max_signal = max(float(sched["signal"].max()), 1e-8)
    sched["importation_mult"] = sched["signal"] / max_signal
    if floor > 0.0:
        sched["importation_mult"] = floor + (1.0 - floor) * sched["importation_mult"]
    sched["importation_mult"] = sched["importation_mult"].clip(lower=0.0)
    sched["date"] = pd.to_datetime(sched["date"]).dt.strftime("%Y-%m-%d")
    sched["signal_col"] = signal_col
    return sched[["date", "signal_col", "signal", "importation_mult"]]


def _random_value(rng: np.random.Generator, low: float, high: float, kind: str) -> float:
    if kind == "log":
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return float(rng.uniform(low, high))


def _sample_random_candidate(
    rng: np.random.Generator,
    idx: int,
    search_bounds: dict[str, tuple[float, float, str]],
) -> dict[str, Any]:
    draw = {key: _random_value(rng, low, high, kind) for key, (low, high, kind) in search_bounds.items()}
    phase_import = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    if "phase_import_1" in draw:
        phase_import[1] = draw["phase_import_1"]
    if "phase_import_2" in draw:
        phase_import[2] = draw["phase_import_2"]
    if "phase_import_3" in draw:
        phase_import[3] = draw["phase_import_3"]
    return {
        "name": f"random_{idx:04d}",
        "beta_scale": draw["beta_scale"],
        "home_beta_alpha": draw["home_beta_alpha"],
        "importation_prob": draw["importation_prob"],
        "phase_beta_mult": {
            0: draw["phase_beta_0"],
            1: draw["phase_beta_1"],
            2: draw["phase_beta_2"],
            3: draw.get("phase_beta_3", draw["phase_beta_0"]),
        },
        "phase_importation_mult": phase_import,
    }


def _sample_local_candidate(
    center: dict[str, Any],
    rng: np.random.Generator,
    idx: int,
    sigma: float,
    search_bounds: dict[str, tuple[float, float, str]],
) -> dict[str, Any]:
    def clamp(key: str, value: float) -> float:
        return _clamp_with_bounds(key, value, search_bounds)

    def jitter_linear(key: str, value: float) -> float:
        low, high, _ = search_bounds[key]
        return clamp(key, value + rng.normal(0.0, sigma * (high - low)))

    def jitter_log(key: str, value: float) -> float:
        low, high, _ = search_bounds[key]
        log_value = np.log(max(value, low))
        new_value = float(np.exp(log_value + rng.normal(0.0, sigma * (np.log(high) - np.log(low)))))
        return clamp(key, new_value)

    phase_import = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    if "phase_import_1" in search_bounds:
        phase_import[1] = jitter_linear("phase_import_1", center["phase_importation_mult"][1])
    if "phase_import_2" in search_bounds:
        phase_import[2] = jitter_linear("phase_import_2", center["phase_importation_mult"][2])
    if "phase_import_3" in search_bounds:
        phase_import[3] = jitter_linear("phase_import_3", center["phase_importation_mult"][3])

    return {
        "name": f"local_{idx:04d}",
        "beta_scale": jitter_linear("beta_scale", center["beta_scale"]),
        "home_beta_alpha": jitter_linear("home_beta_alpha", center["home_beta_alpha"]),
        "importation_prob": jitter_log("importation_prob", center["importation_prob"]),
        "phase_beta_mult": {
            0: jitter_linear("phase_beta_0", center["phase_beta_mult"][0]),
            1: jitter_linear("phase_beta_1", center["phase_beta_mult"][1]),
            2: jitter_linear("phase_beta_2", center["phase_beta_mult"][2]),
            3: (
                jitter_linear("phase_beta_3", center["phase_beta_mult"][3])
                if "phase_beta_3" in search_bounds
                else center["phase_beta_mult"][0]
            ),
        },
        "phase_importation_mult": phase_import,
    }


def _evaluate_candidate(
    *,
    candidate: dict[str, Any],
    base_hazard: dict[str, Any],
    daily_df: pd.DataFrame,
    feature_meta: dict[str, Any],
    official: pd.DataFrame,
    seed_ids: list[str],
    scratch_dir: Path,
    latent_days: int,
    infectious_days: int,
    seed_pct: float,
    rng_seed: int,
    workers: int,
    mixing_scale: str,
    home_protection_scope: str,
    objective_weights: dict[str, float],
    search_bounds: dict[str, tuple[float, float, str]],
    importation_mode: str,
    enforce_monotonic_phase_beta: bool,
    tie_phase2_to_pre: bool,
    official_import_signal: str,
    official_import_floor: float,
    max_external_seed_share: float,
    w_external_seed_share_excess: float,
    shape_scale: float,
    verbose_trials: bool,
) -> dict[str, Any]:
    candidate = _normalize_candidate(
        candidate,
        search_bounds=search_bounds,
        importation_mode=importation_mode,
        enforce_monotonic_phase_beta=enforce_monotonic_phase_beta,
        tie_phase2_to_pre=tie_phase2_to_pre,
    )
    hp = copy.deepcopy(base_hazard)
    hp["beta_base"] = float(base_hazard.get("beta_base", 0.06)) * float(candidate["beta_scale"])
    if _is_seed_mode(importation_mode):
        hp["importation_prob"] = 0.0
        hp["external_seed_prob"] = float(candidate["importation_prob"])
    else:
        hp["importation_prob"] = float(candidate["importation_prob"])
        hp["external_seed_prob"] = 0.0

    sim = SEIRTimeseries(
        daily_df=daily_df,
        feature_meta=feature_meta,
        hazard_params=hp,
        latent_days=latent_days,
        infectious_days=infectious_days,
        seed_pct=seed_pct,
        seed_agent_ids=seed_ids,
        rng_seed=rng_seed,
        n_workers=workers,
    )
    sim.mixing_scale = mixing_scale
    sim.home_beta_alpha = float(candidate["home_beta_alpha"])
    sim.home_protection_scope = home_protection_scope
    sim.phase_beta_mult = {int(k): float(v) for k, v in candidate["phase_beta_mult"].items()}
    sim.phase_importation_mult = {int(k): float(v) for k, v in candidate["phase_importation_mult"].items()}
    if importation_mode == "observed_date_phase":
        sched_df = _build_observed_date_importation_schedule(candidate["phase_importation_mult"])
        sim.date_importation_mult = {
            pd.to_datetime(row["date"]).date(): float(row["importation_mult"])
            for _, row in sched_df.iterrows()
        }
    elif importation_mode == "official_curve":
        sched_df = _build_official_curve_importation_schedule(
            official,
            signal_col=official_import_signal,
            floor=official_import_floor,
        )
        sim.date_importation_mult = {
            pd.to_datetime(row["date"]).date(): float(row["importation_mult"])
            for _, row in sched_df.iterrows()
        }
    elif importation_mode == "official_curve_seed":
        sched_df = _build_official_curve_importation_schedule(
            official,
            signal_col=official_import_signal,
            floor=official_import_floor,
        )
        sim.date_external_seed_mult = {
            pd.to_datetime(row["date"]).date(): float(row["importation_mult"])
            for _, row in sched_df.iterrows()
        }

    scratch_dir.mkdir(parents=True, exist_ok=True)
    if verbose_trials:
        sim.run(out_dir=scratch_dir, rng_seed=rng_seed)
    else:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sim.run(out_dir=scratch_dir, rng_seed=rng_seed)
    sim_df = pd.read_csv(scratch_dir / "seir_timeseries.csv")
    sim_df["date"] = pd.to_datetime(sim_df["date"])
    sim_df = sim_df[(sim_df["date"] >= STUDY_START) & (sim_df["date"] <= STUDY_END)].copy()

    pop_n = float(sim_df.loc[0, ["S", "E", "I", "R"]].sum())
    sim_df["cum_inf"] = pop_n - sim_df["S"].astype(float)
    sim_df["new_E_7dma"] = _smooth(sim_df["new_E"].astype(float), 7)
    if "new_E_external" in sim_df.columns:
        external_seed_share = float(sim_df["new_E_external"].sum()) / max(float(sim_df["new_E"].sum()), 1e-8)
    else:
        external_seed_share = 0.0
    external_seed_share_excess = max(0.0, external_seed_share - max_external_seed_share)

    official_cum = official[(official["date"] >= STUDY_START) & (official["date"] <= STUDY_END)].copy()
    official_inc = official_cum.copy()

    cum = sim_df[["date", "cum_inf"]].merge(
        official_cum[["date", "official_total_cases"]],
        on="date",
        how="inner",
    )
    cum["sim_norm"] = cum["cum_inf"] / max(float(cum["cum_inf"].max()), 1e-8)
    cum["official_norm"] = cum["official_total_cases"] / max(float(cum["official_total_cases"].max()), 1e-8)

    inc = sim_df[["date", "new_E_7dma"]].copy()
    inc["date"] = inc["date"] + pd.Timedelta(days=INCIDENCE_LEAD_DAYS)
    inc = inc.merge(official_inc[["date", "reported_cases_7dma"]], on="date", how="inner")
    inc["sim_norm"] = inc["new_E_7dma"] / max(float(inc["new_E_7dma"].max()), 1e-8)
    inc["official_norm"] = inc["reported_cases_7dma"] / max(float(inc["reported_cases_7dma"].max()), 1e-8)

    act = sim_df[["date", "I"]].copy()
    act["I_7dma"] = _smooth(act["I"].astype(float), 7)
    act = act.merge(official_inc[["date", "official_active_cases_7dma"]], on="date", how="inner")
    act["sim_norm"] = act["I_7dma"] / max(float(act["I_7dma"].max()), 1e-8)
    act["official_norm"] = act["official_active_cases_7dma"] / max(float(act["official_active_cases_7dma"].max()), 1e-8)

    peak_sim = inc.loc[inc["sim_norm"].idxmax(), "date"]
    peak_off = inc.loc[inc["official_norm"].idxmax(), "date"]
    peak_sim_active = act.loc[act["sim_norm"].idxmax(), "date"]
    peak_off_active = act.loc[act["official_norm"].idxmax(), "date"]

    phase_rows: list[dict[str, Any]] = []
    sim_total = max(float(inc["new_E_7dma"].sum()), 1e-8)
    off_total = max(float(inc["reported_cases_7dma"].sum()), 1e-8)
    for phase_name, (start, end) in PHASE_WINDOWS.items():
        mask = (inc["date"] >= pd.Timestamp(start)) & (inc["date"] <= pd.Timestamp(end))
        sim_share = float(inc.loc[mask, "new_E_7dma"].sum()) / sim_total
        off_share = float(inc.loc[mask, "reported_cases_7dma"].sum()) / off_total
        phase_rows.append(
            {
                "phase": phase_name,
                "sim_share": sim_share,
                "official_share": off_share,
                "abs_diff": abs(sim_share - off_share),
            }
        )

    incidence_corr = _safe_corr(inc["sim_norm"], inc["official_norm"])
    cumulative_corr = _safe_corr(cum["sim_norm"], cum["official_norm"])
    incidence_nrmse = float(np.sqrt(np.mean((inc["sim_norm"] - inc["official_norm"]) ** 2)))
    cumulative_nrmse = float(np.sqrt(np.mean((cum["sim_norm"] - cum["official_norm"]) ** 2)))
    peak_date_error_days = int(abs((peak_sim - peak_off).days))
    phase_share_mae = float(np.mean([row["abs_diff"] for row in phase_rows]))
    active_corr = _safe_corr(act["sim_norm"], act["official_norm"])
    active_nrmse = float(np.sqrt(np.mean((act["sim_norm"] - act["official_norm"]) ** 2)))
    active_peak_date_error_days = int(abs((peak_sim_active - peak_off_active).days))

    inc_peak_mass_mae = abs(_peak_mass_ratio(inc["new_E_7dma"]) - _peak_mass_ratio(inc["reported_cases_7dma"]))
    inc_post_peak14_mae = abs(
        _ratio_at_offset(inc, "new_E_7dma", peak_sim, POST_PEAK_OFFSET_DAYS)
        - _ratio_at_offset(inc, "reported_cases_7dma", peak_off, POST_PEAK_OFFSET_DAYS)
    )
    active_tail_share_mae = abs(
        _share_after(act, "I_7dma", TAIL_START)
        - _share_after(act, "official_active_cases_7dma", TAIL_START)
    )

    loss = (
        objective_weights["incidence_corr"] * (1.0 - incidence_corr)
        + objective_weights["cumulative_corr"] * (1.0 - cumulative_corr)
        + objective_weights["incidence_nrmse"] * incidence_nrmse
        + objective_weights["cumulative_nrmse"] * cumulative_nrmse
        + objective_weights["peak_date_error_days"] * (peak_date_error_days / 7.0)
        + objective_weights["phase_share_mae"] * (phase_share_mae / 0.25)
        + objective_weights["active_corr"] * (1.0 - active_corr)
        + objective_weights["active_nrmse"] * active_nrmse
        + objective_weights["active_peak_date_error_days"] * (active_peak_date_error_days / 7.0)
        + objective_weights["inc_peak_mass_mae"] * (inc_peak_mass_mae / max(shape_scale, 1e-8))
        + objective_weights["inc_post_peak14_mae"] * (inc_post_peak14_mae / max(shape_scale, 1e-8))
        + objective_weights["active_tail_share_mae"] * (active_tail_share_mae / max(shape_scale, 1e-8))
        + float(w_external_seed_share_excess) * (external_seed_share_excess / max(max_external_seed_share, 1e-8))
    )

    return {
        "candidate": candidate,
        "effective_hazard_params": hp,
        "loss": float(loss),
        "metrics": {
            "incidence_corr": incidence_corr,
            "incidence_nrmse": incidence_nrmse,
            "peak_date_error_days": peak_date_error_days,
            "phase_share_mae": phase_share_mae,
            "cumulative_corr": cumulative_corr,
            "cumulative_nrmse": cumulative_nrmse,
            "active_corr": active_corr,
            "active_nrmse": active_nrmse,
            "active_peak_date_error_days": active_peak_date_error_days,
            "inc_peak_mass_mae": inc_peak_mass_mae,
            "inc_post_peak14_mae": inc_post_peak14_mae,
            "active_tail_share_mae": active_tail_share_mae,
            "external_seed_share": external_seed_share,
            "external_seed_share_excess": external_seed_share_excess,
        },
        "phase_share": pd.DataFrame(phase_rows),
        "sim_df": sim_df,
    }


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    cand = result["candidate"]
    met = result["metrics"]
    return {
        "name": cand["name"],
        "loss": result["loss"],
        "beta_scale": cand["beta_scale"],
        "home_beta_alpha": cand["home_beta_alpha"],
        "importation_prob": cand["importation_prob"],
        "phase_beta_0": cand["phase_beta_mult"][0],
        "phase_beta_1": cand["phase_beta_mult"][1],
        "phase_beta_2": cand["phase_beta_mult"][2],
        "phase_beta_3": cand["phase_beta_mult"][3],
        "phase_import_0": cand["phase_importation_mult"][0],
        "phase_import_1": cand["phase_importation_mult"][1],
        "phase_import_2": cand["phase_importation_mult"][2],
        "phase_import_3": cand["phase_importation_mult"][3],
        "external_seed_prob": cand["importation_prob"],
        "incidence_corr": met["incidence_corr"],
        "incidence_nrmse": met["incidence_nrmse"],
        "peak_date_error_days": met["peak_date_error_days"],
        "phase_share_mae": met["phase_share_mae"],
        "cumulative_corr": met["cumulative_corr"],
        "cumulative_nrmse": met["cumulative_nrmse"],
        "active_corr": met["active_corr"],
        "active_nrmse": met["active_nrmse"],
        "active_peak_date_error_days": met["active_peak_date_error_days"],
        "inc_peak_mass_mae": met["inc_peak_mass_mae"],
        "inc_post_peak14_mae": met["inc_post_peak14_mae"],
        "active_tail_share_mae": met["active_tail_share_mae"],
        "external_seed_share": met.get("external_seed_share", 0.0),
        "external_seed_share_excess": met.get("external_seed_share_excess", 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproducible automatic trend calibration for the SEIR simulator.")
    ap.add_argument("--daily_shares", default=str(DEFAULT_DAILY_SHARES))
    ap.add_argument("--feature_meta", default=str(DEFAULT_FEATURE_META))
    ap.add_argument("--base_hazard", default=str(DEFAULT_BASE_HAZARD))
    ap.add_argument("--official_curve_html", default=str(OFFICIAL_CURVE_HTML))
    ap.add_argument("--seed_list", default=str(DEFAULT_SEED_LIST))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--scratch_dir", default=str(DEFAULT_SCRATCH_DIR))
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--refine_trials", type=int, default=24)
    ap.add_argument("--refine_sigma", type=float, default=0.12)
    ap.add_argument("--bounds_profile", choices=sorted(SEARCH_BOUNDS_PROFILES), default="default_v1")
    ap.add_argument("--search_seed", type=int, default=20260710)
    ap.add_argument("--sim_seed", type=int, default=1234)
    ap.add_argument("--seed_pct", type=float, default=0.002)
    ap.add_argument("--latent_days", type=int, default=3)
    ap.add_argument("--infectious_days", type=int, default=7)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mixing_scale", choices=["observed", "24h"], default="24h")
    ap.add_argument("--home_protection_scope", choices=["population", "agent"], default="population")
    ap.add_argument("--importation_mode", choices=["phase", "constant", "observed_date_phase", "official_curve", "official_curve_seed"], default="phase")
    ap.add_argument("--official_import_signal", choices=sorted(OFFICIAL_IMPORT_SIGNALS), default="official_active_cases_7dma")
    ap.add_argument("--official_import_floor", type=float, default=0.0)
    ap.add_argument("--enforce_monotonic_phase_beta", action="store_true")
    ap.add_argument("--tie_phase2_to_pre", action="store_true")
    ap.add_argument("--w_incidence_corr", type=float, default=1.20)
    ap.add_argument("--w_cumulative_corr", type=float, default=0.80)
    ap.add_argument("--w_incidence_nrmse", type=float, default=1.00)
    ap.add_argument("--w_cumulative_nrmse", type=float, default=0.60)
    ap.add_argument("--w_peak_date_error_days", type=float, default=0.20)
    ap.add_argument("--w_phase_share_mae", type=float, default=0.80)
    ap.add_argument("--w_active_corr", type=float, default=1.20)
    ap.add_argument("--w_active_nrmse", type=float, default=1.10)
    ap.add_argument("--w_active_peak_date_error_days", type=float, default=0.35)
    ap.add_argument("--w_inc_peak_mass_mae", type=float, default=1.00)
    ap.add_argument("--w_inc_post_peak14_mae", type=float, default=1.10)
    ap.add_argument("--w_active_tail_share_mae", type=float, default=1.20)
    ap.add_argument("--shape_scale", type=float, default=0.12)
    ap.add_argument("--max_external_seed_share", type=float, default=0.15)
    ap.add_argument("--w_external_seed_share_excess", type=float, default=0.0)
    ap.add_argument("--verbose_trials", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    scratch_dir = Path(args.scratch_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_df = pd.read_csv(args.daily_shares)
    feature_meta = _load_feature_meta(Path(args.feature_meta))
    base_hazard = _read_json(Path(args.base_hazard))
    official = _load_official_curve(Path(args.official_curve_html))
    seed_ids = _build_seed_list(Path(args.seed_list), Path(args.daily_shares), args.seed_pct, args.sim_seed)
    search_bounds = copy.deepcopy(SEARCH_BOUNDS_PROFILES[args.bounds_profile])
    if args.importation_mode in {"constant", "official_curve", "official_curve_seed"}:
        search_bounds = {k: v for k, v in search_bounds.items() if not k.startswith("phase_import_")}
    if args.tie_phase2_to_pre:
        search_bounds = {k: v for k, v in search_bounds.items() if k != "phase_beta_3"}
    objective_weights = {
        "incidence_corr": float(args.w_incidence_corr),
        "cumulative_corr": float(args.w_cumulative_corr),
        "incidence_nrmse": float(args.w_incidence_nrmse),
        "cumulative_nrmse": float(args.w_cumulative_nrmse),
        "peak_date_error_days": float(args.w_peak_date_error_days),
        "phase_share_mae": float(args.w_phase_share_mae),
        "active_corr": float(args.w_active_corr),
        "active_nrmse": float(args.w_active_nrmse),
        "active_peak_date_error_days": float(args.w_active_peak_date_error_days),
        "inc_peak_mass_mae": float(args.w_inc_peak_mass_mae),
        "inc_post_peak14_mae": float(args.w_inc_post_peak14_mae),
        "active_tail_share_mae": float(args.w_active_tail_share_mae),
    }

    rng = np.random.default_rng(args.search_seed)
    manual_v1 = _candidate_from_manual(
        name="manual_v1",
        beta_scale=1.35,
        home_beta_alpha=0.25,
        importation_prob=float(base_hazard.get("importation_prob", 0.0001)),
        phase_beta_mult=base_hazard.get("phase_beta_mult", {"0": 1.1, "1": 0.4, "2": 0.65, "3": 0.78}),
        phase_importation_mult=base_hazard.get("phase_importation_mult", {"0": 1.0, "1": 4.5, "2": 0.8, "3": 2.0}),
    )
    original_hazard = _read_json(ROOT / "dataset" / "MetaData" / "hazard_params.json")
    original_manual = _candidate_from_manual(
        name="original_baseline",
        beta_scale=1.0,
        home_beta_alpha=0.0,
        importation_prob=float(original_hazard.get("importation_prob", 0.001)),
        phase_beta_mult=original_hazard.get("phase_beta_mult", {"0": 1.0, "1": 0.52, "2": 0.9, "3": 1.0}),
        phase_importation_mult={"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0},
    )

    candidates: list[dict[str, Any]] = [manual_v1, original_manual]
    for idx in range(args.trials):
        candidates.append(_sample_random_candidate(rng, idx, search_bounds))

    results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for idx, cand in enumerate(candidates, start=1):
        result = _evaluate_candidate(
            candidate=cand,
            base_hazard=base_hazard,
            daily_df=daily_df,
            feature_meta=feature_meta,
            official=official,
            seed_ids=seed_ids,
            scratch_dir=scratch_dir,
            latent_days=args.latent_days,
            infectious_days=args.infectious_days,
            seed_pct=args.seed_pct,
            rng_seed=args.sim_seed,
            workers=args.workers,
            mixing_scale=args.mixing_scale,
            home_protection_scope=args.home_protection_scope,
            objective_weights=objective_weights,
            search_bounds=search_bounds,
            importation_mode=args.importation_mode,
            enforce_monotonic_phase_beta=bool(args.enforce_monotonic_phase_beta),
            tie_phase2_to_pre=bool(args.tie_phase2_to_pre),
            official_import_signal=args.official_import_signal,
            official_import_floor=float(args.official_import_floor),
            max_external_seed_share=float(args.max_external_seed_share),
            w_external_seed_share_excess=float(args.w_external_seed_share_excess),
            shape_scale=float(args.shape_scale),
            verbose_trials=bool(args.verbose_trials),
        )
        results.append(result)
        if best_result is None or result["loss"] < best_result["loss"]:
            best_result = result
        print(
            f"[{idx:03d}/{len(candidates)}] {cand['name']}: "
            f"loss={result['loss']:.4f}, "
            f"inc_corr={result['metrics']['incidence_corr']:.3f}, "
            f"act_corr={result['metrics']['active_corr']:.3f}, "
            f"cum_corr={result['metrics']['cumulative_corr']:.3f}, "
            f"peak_err={result['metrics']['peak_date_error_days']}d"
        )

    assert best_result is not None
    for idx in range(args.refine_trials):
        cand = _sample_local_candidate(best_result["candidate"], rng, idx, args.refine_sigma, search_bounds)
        result = _evaluate_candidate(
            candidate=cand,
            base_hazard=base_hazard,
            daily_df=daily_df,
            feature_meta=feature_meta,
            official=official,
            seed_ids=seed_ids,
            scratch_dir=scratch_dir,
            latent_days=args.latent_days,
            infectious_days=args.infectious_days,
            seed_pct=args.seed_pct,
            rng_seed=args.sim_seed,
            workers=args.workers,
            mixing_scale=args.mixing_scale,
            home_protection_scope=args.home_protection_scope,
            objective_weights=objective_weights,
            search_bounds=search_bounds,
            importation_mode=args.importation_mode,
            enforce_monotonic_phase_beta=bool(args.enforce_monotonic_phase_beta),
            tie_phase2_to_pre=bool(args.tie_phase2_to_pre),
            official_import_signal=args.official_import_signal,
            official_import_floor=float(args.official_import_floor),
            max_external_seed_share=float(args.max_external_seed_share),
            w_external_seed_share_excess=float(args.w_external_seed_share_excess),
            shape_scale=float(args.shape_scale),
            verbose_trials=bool(args.verbose_trials),
        )
        results.append(result)
        if result["loss"] < best_result["loss"]:
            best_result = result
            print(
                f"[refine {idx + 1:03d}/{args.refine_trials}] new best: "
                f"loss={result['loss']:.4f}, "
                f"inc_corr={result['metrics']['incidence_corr']:.3f}, "
                f"act_corr={result['metrics']['active_corr']:.3f}, "
                f"cum_corr={result['metrics']['cumulative_corr']:.3f}, "
                f"peak_err={result['metrics']['peak_date_error_days']}d"
            )

    flat_results = pd.DataFrame([_flatten_result(res) for res in results]).sort_values("loss").reset_index(drop=True)
    flat_results.to_csv(out_dir / "search_results.csv", index=False)
    flat_results.head(20).to_csv(out_dir / "top20_results.csv", index=False)

    best = best_result
    best["sim_df"].to_csv(out_dir / "best_seir_timeseries.csv", index=False)
    best["phase_share"].to_csv(out_dir / "best_phase_share.csv", index=False)
    (out_dir / "best_metrics.json").write_text(json.dumps(best["metrics"], indent=2), encoding="utf-8")
    (out_dir / "best_candidate.json").write_text(json.dumps(best["candidate"], indent=2), encoding="utf-8")
    if args.importation_mode == "observed_date_phase":
        observed_date_import_path = out_dir / "observed_date_importation_schedule.csv"
        _build_observed_date_importation_schedule(best["candidate"]["phase_importation_mult"]).to_csv(
            observed_date_import_path,
            index=False,
        )
    elif args.importation_mode == "official_curve":
        observed_date_import_path = out_dir / "official_curve_importation_schedule.csv"
        _build_official_curve_importation_schedule(
            official,
            signal_col=args.official_import_signal,
            floor=float(args.official_import_floor),
        ).to_csv(
            observed_date_import_path,
            index=False,
        )
    elif args.importation_mode == "official_curve_seed":
        observed_date_import_path = None
        external_seed_schedule_path = out_dir / "official_curve_external_seed_schedule.csv"
        _build_official_curve_importation_schedule(
            official,
            signal_col=args.official_import_signal,
            floor=float(args.official_import_floor),
        ).to_csv(
            external_seed_schedule_path,
            index=False,
        )
    else:
        observed_date_import_path = None
        external_seed_schedule_path = None
    if args.importation_mode != "official_curve_seed":
        external_seed_schedule_path = None
    (out_dir / "best_effective_hazard_params.json").write_text(
        json.dumps(
            {
                "daily_shares": str(Path(args.daily_shares).resolve()),
                "feature_meta": str(Path(args.feature_meta).resolve()),
                "hazard_params_source": str((out_dir / "hazard_params_auto_calibrated.json").resolve()),
                "base_hazard": str(Path(args.base_hazard).resolve()),
                "official_curve_html": str(Path(args.official_curve_html).resolve()),
                "seed_list": str(Path(args.seed_list).resolve()),
                "latent_days": int(args.latent_days),
                "infectious_days": int(args.infectious_days),
                "seed_pct": float(args.seed_pct),
                "sim_seed": int(args.sim_seed),
                "rng_seed": int(args.sim_seed),
                "workers": int(args.workers),
                "mixing_scale": args.mixing_scale,
                "home_protection_scope": args.home_protection_scope,
                "beta_scale": float(best["candidate"]["beta_scale"]),
                "home_beta_alpha": float(best["candidate"]["home_beta_alpha"]),
                "importation_prob": 0.0 if _is_seed_mode(args.importation_mode) else float(best["candidate"]["importation_prob"]),
                "external_seed_prob": float(best["candidate"]["importation_prob"]) if _is_seed_mode(args.importation_mode) else 0.0,
                "phase_beta_mult": {
                    str(k): float(v) for k, v in best["candidate"]["phase_beta_mult"].items()
                },
                "phase_importation_mult": {
                    str(k): float(v) for k, v in best["candidate"]["phase_importation_mult"].items()
                },
                "importation_mode": args.importation_mode,
                "tie_phase2_to_pre": bool(args.tie_phase2_to_pre),
                "official_import_signal": args.official_import_signal if args.importation_mode in {"official_curve", "official_curve_seed"} else "",
                "official_import_floor": float(args.official_import_floor) if args.importation_mode in {"official_curve", "official_curve_seed"} else 0.0,
                "date_importation_schedule": str(observed_date_import_path.resolve()) if observed_date_import_path else "",
                "date_external_seed_schedule": str(external_seed_schedule_path.resolve()) if external_seed_schedule_path else "",
                "search_seed": int(args.search_seed),
                "search_trials": int(args.trials),
                "refine_trials": int(args.refine_trials),
                "refine_sigma": float(args.refine_sigma),
                "best_candidate": best["candidate"],
                "best_metrics": best["metrics"],
                "hazard_params_effective": best["effective_hazard_params"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "search_config.json").write_text(
        json.dumps(
            {
                "search_bounds": {
                    k: {"low": float(v[0]), "high": float(v[1]), "scale": v[2]}
                    for k, v in search_bounds.items()
                },
                "bounds_profile": args.bounds_profile,
                "importation_mode": args.importation_mode,
                "home_protection_scope": args.home_protection_scope,
                "enforce_monotonic_phase_beta": bool(args.enforce_monotonic_phase_beta),
                "tie_phase2_to_pre": bool(args.tie_phase2_to_pre),
                "official_import_signal": args.official_import_signal,
                "official_import_floor": float(args.official_import_floor),
                "shape_scale": float(args.shape_scale),
                "max_external_seed_share": float(args.max_external_seed_share),
                "w_external_seed_share_excess": float(args.w_external_seed_share_excess),
                "objective_weights": objective_weights,
                "loss_definition": {
                    "incidence_corr": "w_incidence_corr * (1 - corr)",
                    "cumulative_corr": "w_cumulative_corr * (1 - corr)",
                    "incidence_nrmse": "w_incidence_nrmse * nrmse",
                    "cumulative_nrmse": "w_cumulative_nrmse * nrmse",
                    "peak_date_error_days": "w_peak_date_error_days * (days / 7)",
                    "phase_share_mae": "w_phase_share_mae * (mae / 0.25)",
                    "active_corr": "w_active_corr * (1 - corr)",
                    "active_nrmse": "w_active_nrmse * nrmse",
                    "active_peak_date_error_days": "w_active_peak_date_error_days * (days / 7)",
                    "inc_peak_mass_mae": "w_inc_peak_mass_mae * mae / shape_scale",
                    "inc_post_peak14_mae": "w_inc_post_peak14_mae * mae / shape_scale",
                    "active_tail_share_mae": "w_active_tail_share_mae * mae / shape_scale",
                    "external_seed_share_excess": "w_external_seed_share_excess * max(0, share - max_external_seed_share) / max_external_seed_share",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    auto_hazard = copy.deepcopy(base_hazard)
    auto_hazard["importation_prob"] = 0.0 if _is_seed_mode(args.importation_mode) else float(best["candidate"]["importation_prob"])
    auto_hazard["external_seed_prob"] = float(best["candidate"]["importation_prob"]) if _is_seed_mode(args.importation_mode) else 0.0
    auto_hazard["phase_beta_mult"] = {str(k): float(v) for k, v in best["candidate"]["phase_beta_mult"].items()}
    auto_hazard["phase_importation_mult"] = {
        str(k): float(v) for k, v in best["candidate"]["phase_importation_mult"].items()
    }
    (out_dir / "hazard_params_auto_calibrated.json").write_text(
        json.dumps(auto_hazard, indent=2),
        encoding="utf-8",
    )

    print("\n[done] best candidate")
    print(json.dumps(best["candidate"], indent=2))
    print("[done] best metrics")
    print(json.dumps(best["metrics"], indent=2))
    print(f"[done] wrote search artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
