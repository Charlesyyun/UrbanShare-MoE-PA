from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


GLOBAL_SCOPE_ID = "__global__"


def _load_home_cat_idx(feature_meta_path: str | None) -> int:
    if not feature_meta_path:
        return 11
    with open(feature_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return int(meta.get("home_cat_idx", 11))


def _normalize_gender(series: pd.Series) -> pd.Series:
    out = series.fillna("U").astype(str).str.strip().str.upper()
    out = out.replace({"": "U", "UNKNOWN": "U", "NAN": "U", "NONE": "U"})
    return out.where(out.isin(["M", "F"]), "U")


def _infer_age_idx(df: pd.DataFrame) -> pd.Series:
    age_cols = [c for c in df.columns if c.startswith("age_")]
    if not age_cols:
        return pd.Series(np.full(len(df), -1, dtype=np.int16), index=df.index)

    age_block = df[age_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    any_age = age_block.sum(axis=1) > 0
    best_col = age_block.idxmax(axis=1).str.replace("age_", "", regex=False)
    best_idx = pd.to_numeric(best_col, errors="coerce").fillna(-1).astype(np.int16)
    return best_idx.where(any_age, -1)


def _load_agent_profiles(daily_shares_path: str) -> pd.DataFrame:
    daily = pd.read_csv(daily_shares_path)
    if "agent_id" not in daily.columns:
        raise ValueError("--daily_shares must include 'agent_id'")

    keep_cols = ["agent_id"]
    if "gender" in daily.columns:
        keep_cols.append("gender")
    keep_cols.extend([c for c in daily.columns if c.startswith("age_")])

    prof = daily[keep_cols].drop_duplicates(subset=["agent_id"], keep="first").copy()
    prof["gender_norm"] = _normalize_gender(prof["gender"]) if "gender" in prof.columns else "U"
    prof["age_idx"] = _infer_age_idx(prof)
    prof["demo_group"] = prof["gender_norm"] + "_age" + prof["age_idx"].astype(str)
    return prof[["agent_id", "gender_norm", "age_idx", "demo_group"]]


def _weekday_bucket(dates: pd.Series, bucket_mode: str) -> pd.Series:
    weekday = pd.to_datetime(dates).dt.weekday
    if bucket_mode == "weekday_weekend":
        return np.where(weekday >= 5, "weekend", "weekday")
    if bucket_mode == "full_week":
        return weekday.map(
            {
                0: "mon",
                1: "tue",
                2: "wed",
                3: "thu",
                4: "fri",
                5: "sat",
                6: "sun",
            }
        )
    raise ValueError(f"Unsupported bucket_mode: {bucket_mode}")


def _prepare_geo(
    daily_geo_path: str,
    daily_shares_path: str,
    feature_meta_path: str | None,
    include_home: bool,
    bucket_mode: str,
) -> tuple[pd.DataFrame, int]:
    geo = pd.read_csv(
        daily_geo_path,
        usecols=["agent_id", "date", "poi_id_num", "cat_idx", "epi_phase", "duration_s"],
    )
    need = {"agent_id", "date", "poi_id_num", "cat_idx", "epi_phase", "duration_s"}
    if not need.issubset(set(geo.columns)):
        missing = sorted(need.difference(set(geo.columns)))
        raise ValueError(f"--daily_geo missing columns: {missing}")

    geo["date"] = pd.to_datetime(geo["date"], errors="coerce")
    geo["duration_s"] = pd.to_numeric(geo["duration_s"], errors="coerce").fillna(0.0)
    geo["poi_id_num"] = pd.to_numeric(geo["poi_id_num"], errors="coerce").fillna(-1).astype(np.int64)
    geo["cat_idx"] = pd.to_numeric(geo["cat_idx"], errors="coerce").fillna(-1).astype(np.int16)
    geo["epi_phase"] = pd.to_numeric(geo["epi_phase"], errors="coerce").fillna(-1).astype(np.int16)
    geo = geo.loc[geo["date"].notna() & (geo["duration_s"] > 0.0)].copy()

    home_cat_idx = _load_home_cat_idx(feature_meta_path)
    if not include_home:
        geo = geo.loc[(geo["poi_id_num"] >= 0) & (geo["cat_idx"] != home_cat_idx)].copy()

    profiles = _load_agent_profiles(daily_shares_path)
    geo = geo.merge(profiles, on="agent_id", how="left")
    geo["gender_norm"] = geo["gender_norm"].fillna("U")
    geo["age_idx"] = pd.to_numeric(geo["age_idx"], errors="coerce").fillna(-1).astype(np.int16)
    geo["demo_group"] = geo["demo_group"].fillna(geo["gender_norm"] + "_age" + geo["age_idx"].astype(str))
    geo["weekday_bucket"] = _weekday_bucket(geo["date"], bucket_mode)
    return geo, home_cat_idx


def _aggregate_level(
    geo: pd.DataFrame,
    *,
    level_name: str,
    level_rank: int,
    scope_type: str,
    scope_col: str | None,
    phase_specific: bool,
    weekday_specific: bool,
    topk: int,
    min_template_hours: float,
    min_template_days: int,
) -> pd.DataFrame:
    work = geo[["agent_id", "date", "poi_id_num", "cat_idx", "epi_phase", "duration_s", "weekday_bucket", "demo_group"]].copy()
    work["scope_id"] = work[scope_col].astype(str) if scope_col else GLOBAL_SCOPE_ID
    if not phase_specific:
        work["epi_phase"] = -1
    if not weekday_specific:
        work["weekday_bucket"] = "all"

    key_cols = ["scope_id", "cat_idx", "weekday_bucket", "epi_phase"]
    agg = (
        work.groupby(key_cols + ["poi_id_num"], dropna=False)
        .agg(raw_hours=("duration_s", lambda s: float(s.sum()) / 3600.0), support_days=("date", "nunique"))
        .reset_index()
    )
    template_stats = (
        work.groupby(key_cols, dropna=False)
        .agg(template_hours=("duration_s", lambda s: float(s.sum()) / 3600.0), template_days=("date", "nunique"))
        .reset_index()
    )
    agg = agg.merge(template_stats, on=key_cols, how="left")
    agg = agg.loc[
        (agg["template_hours"] >= float(min_template_hours))
        & (agg["template_days"] >= int(min_template_days))
    ].copy()
    if agg.empty:
        return agg

    agg["scope_type"] = scope_type
    agg["level_name"] = level_name
    agg["level_rank"] = int(level_rank)

    agg = agg.sort_values(
        key_cols + ["raw_hours", "support_days", "poi_id_num"],
        ascending=[True, True, True, True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    agg["rank"] = agg.groupby(key_cols, sort=False).cumcount() + 1
    agg = agg.loc[agg["rank"] <= int(topk)].copy()

    topk_hours = agg.groupby(key_cols, dropna=False)["raw_hours"].sum().rename("topk_hours").reset_index()
    agg = agg.merge(topk_hours, on=key_cols, how="left")
    agg["weight"] = np.where(agg["topk_hours"] > 0.0, agg["raw_hours"] / agg["topk_hours"], 0.0)
    agg["coverage_topk"] = np.where(agg["template_hours"] > 0.0, agg["topk_hours"] / agg["template_hours"], 0.0)

    out_cols = [
        "level_name",
        "level_rank",
        "scope_type",
        "scope_id",
        "cat_idx",
        "weekday_bucket",
        "epi_phase",
        "poi_id_num",
        "rank",
        "weight",
        "raw_hours",
        "support_days",
        "template_hours",
        "template_days",
        "topk_hours",
        "coverage_topk",
    ]
    return agg[out_cols].copy()


def _build_all_levels(
    geo: pd.DataFrame,
    *,
    topk: int,
    min_template_hours: float,
    min_template_days: int,
) -> pd.DataFrame:
    level_specs = [
        ("agent_phase_bucket", 0, "agent", "agent_id", True, True),
        ("agent_phase_allweek", 1, "agent", "agent_id", True, False),
        ("agent_allphase_bucket", 2, "agent", "agent_id", False, True),
        ("agent_allphase_allweek", 3, "agent", "agent_id", False, False),
        ("group_phase_bucket", 4, "group", "demo_group", True, True),
        ("group_phase_allweek", 5, "group", "demo_group", True, False),
        ("group_allphase_bucket", 6, "group", "demo_group", False, True),
        ("group_allphase_allweek", 7, "group", "demo_group", False, False),
        ("global_phase_bucket", 8, "global", None, True, True),
        ("global_phase_allweek", 9, "global", None, True, False),
        ("global_allphase_bucket", 10, "global", None, False, True),
        ("global_allphase_allweek", 11, "global", None, False, False),
    ]
    frames: list[pd.DataFrame] = []
    for level_name, level_rank, scope_type, scope_col, phase_specific, weekday_specific in level_specs:
        frame = _aggregate_level(
            geo,
            level_name=level_name,
            level_rank=level_rank,
            scope_type=scope_type,
            scope_col=scope_col,
            phase_specific=phase_specific,
            weekday_specific=weekday_specific,
            topk=topk,
            min_template_hours=min_template_hours,
            min_template_days=min_template_days,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "level_name",
                "level_rank",
                "scope_type",
                "scope_id",
                "cat_idx",
                "weekday_bucket",
                "epi_phase",
                "poi_id_num",
                "rank",
                "weight",
                "raw_hours",
                "support_days",
                "template_hours",
                "template_days",
                "topk_hours",
                "coverage_topk",
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(
        ["level_rank", "scope_id", "cat_idx", "epi_phase", "weekday_bucket", "rank"],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def _write_table(df: pd.DataFrame, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        df.to_parquet(out, index=False)
        return
    df.to_csv(out, index=False, float_format="%.8f")


def _build_summary(
    templates: pd.DataFrame,
    *,
    daily_geo_path: str,
    daily_shares_path: str,
    out_path: str,
    home_cat_idx: int,
    include_home: bool,
    bucket_mode: str,
    topk: int,
    min_template_hours: float,
    min_template_days: int,
) -> dict:
    by_level = []
    if not templates.empty:
        for level_name, g in templates.groupby("level_name", sort=False):
            by_level.append(
                {
                    "level_name": str(level_name),
                    "rows": int(len(g)),
                    "n_templates": int(
                        g[["scope_id", "cat_idx", "weekday_bucket", "epi_phase"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "mean_coverage_topk": float(g["coverage_topk"].mean()),
                }
            )

    return {
        "daily_geo": str(Path(daily_geo_path)),
        "daily_shares": str(Path(daily_shares_path)),
        "out_path": str(Path(out_path)),
        "home_cat_idx": int(home_cat_idx),
        "include_home": bool(include_home),
        "bucket_mode": str(bucket_mode),
        "topk": int(topk),
        "min_template_hours": float(min_template_hours),
        "min_template_days": int(min_template_days),
        "n_rows_templates": int(len(templates)),
        "n_levels": int(templates["level_name"].nunique()) if not templates.empty else 0,
        "levels": by_level,
        "fallback_order": [
            "agent_phase_bucket",
            "agent_phase_allweek",
            "agent_allphase_bucket",
            "agent_allphase_allweek",
            "group_phase_bucket",
            "group_phase_allweek",
            "group_allphase_bucket",
            "group_allphase_allweek",
            "global_phase_bucket",
            "global_phase_allweek",
            "global_allphase_bucket",
            "global_allphase_allweek",
        ],
    }


def _write_summary(summary: dict, summary_json: str) -> None:
    out = Path(summary_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build multi-level POI allocation templates from enriched geo stay records."
    )
    ap.add_argument("--daily_geo", required=True, help="Path to enriched daily_geo.csv")
    ap.add_argument("--daily_shares", required=True, help="Path to enriched daily_shares.csv")
    ap.add_argument("--out_path", required=True, help="Output template table (.csv or .parquet)")
    ap.add_argument("--feature_meta", default=None, help="Optional feature_meta.json for home_cat_idx")
    ap.add_argument("--summary_json", default=None, help="Optional summary JSON path")
    ap.add_argument("--topk", type=int, default=8, help="Keep top-K POIs per template")
    ap.add_argument(
        "--bucket_mode",
        choices=["weekday_weekend", "full_week"],
        default="weekday_weekend",
        help="How to bucket weekdays for templates",
    )
    ap.add_argument(
        "--min_template_hours",
        type=float,
        default=0.25,
        help="Drop templates with fewer total hours than this threshold",
    )
    ap.add_argument(
        "--min_template_days",
        type=int,
        default=1,
        help="Drop templates with fewer supporting dates than this threshold",
    )
    ap.add_argument(
        "--include_home",
        action="store_true",
        help="Include home / poi_id_num < 0 rows in the templates",
    )
    args = ap.parse_args()

    geo, home_cat_idx = _prepare_geo(
        daily_geo_path=args.daily_geo,
        daily_shares_path=args.daily_shares,
        feature_meta_path=args.feature_meta,
        include_home=bool(args.include_home),
        bucket_mode=args.bucket_mode,
    )
    templates = _build_all_levels(
        geo,
        topk=args.topk,
        min_template_hours=args.min_template_hours,
        min_template_days=args.min_template_days,
    )
    _write_table(templates, args.out_path)

    summary_json = args.summary_json
    if not summary_json:
        out = Path(args.out_path)
        summary_json = str(out.with_suffix(out.suffix + ".summary.json"))
    summary = _build_summary(
        templates,
        daily_geo_path=args.daily_geo,
        daily_shares_path=args.daily_shares,
        out_path=args.out_path,
        home_cat_idx=home_cat_idx,
        include_home=bool(args.include_home),
        bucket_mode=args.bucket_mode,
        topk=args.topk,
        min_template_hours=args.min_template_hours,
        min_template_days=args.min_template_days,
    )
    _write_summary(summary, summary_json)

    print("[OK] Wrote:")
    print(f"- {Path(args.out_path)}")
    print(f"- {Path(summary_json)}")
    if not templates.empty:
        print(
            f"[info] template rows={len(templates):,}  levels={templates['level_name'].nunique()}  "
            f"unique templates={templates[['level_name','scope_id','cat_idx','weekday_bucket','epi_phase']].drop_duplicates().shape[0]:,}"
        )
    else:
        print("[warn] No templates were produced after filtering.")


if __name__ == "__main__":
    main()
