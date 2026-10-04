from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_poi_allocation_templates import (
    _infer_age_idx,
    _load_home_cat_idx,
    _normalize_gender,
    _weekday_bucket,
    _write_table,
)


UNRESOLVED_POI_BASE = -1000
GLOBAL_SCOPE_ID = "__global__"


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _load_templates(path: str) -> tuple[dict[tuple[str, str, int, str, int], pd.DataFrame], list[str]]:
    tpl = _read_table(path).copy()
    if tpl.empty:
        return {}, []

    need = {
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
        "coverage_topk",
    }
    missing = sorted(need.difference(set(tpl.columns)))
    if missing:
        raise ValueError(f"--templates missing columns: {missing}")

    tpl["scope_id"] = tpl["scope_id"].astype(str)
    tpl["cat_idx"] = pd.to_numeric(tpl["cat_idx"], errors="coerce").fillna(-1).astype(np.int16)
    tpl["epi_phase"] = pd.to_numeric(tpl["epi_phase"], errors="coerce").fillna(-1).astype(np.int16)
    tpl["poi_id_num"] = pd.to_numeric(tpl["poi_id_num"], errors="coerce").fillna(-1).astype(np.int64)
    tpl["rank"] = pd.to_numeric(tpl["rank"], errors="coerce").fillna(0).astype(np.int16)
    tpl["level_rank"] = pd.to_numeric(tpl["level_rank"], errors="coerce").fillna(999).astype(np.int16)
    tpl["weight"] = pd.to_numeric(tpl["weight"], errors="coerce").fillna(0.0).astype(np.float64)
    tpl["coverage_topk"] = pd.to_numeric(tpl["coverage_topk"], errors="coerce").fillna(0.0).astype(np.float64)
    tpl = tpl.sort_values(["level_rank", "rank", "poi_id_num"], kind="mergesort").reset_index(drop=True)

    fallback_order = (
        tpl[["level_name", "level_rank"]]
        .drop_duplicates()
        .sort_values(["level_rank", "level_name"], kind="mergesort")["level_name"]
        .astype(str)
        .tolist()
    )

    lookup: dict[tuple[str, str, int, str, int], pd.DataFrame] = {}
    for key, g in tpl.groupby(["level_name", "scope_id", "cat_idx", "weekday_bucket", "epi_phase"], sort=False):
        lookup[(str(key[0]), str(key[1]), int(key[2]), str(key[3]), int(key[4]))] = g[
            ["poi_id_num", "rank", "weight", "coverage_topk"]
        ].reset_index(drop=True)
    return lookup, fallback_order


def _load_daily_rows(path: str, bucket_mode: str) -> tuple[pd.DataFrame, int]:
    daily = pd.read_csv(path)
    if "agent_id" not in daily.columns or "date" not in daily.columns:
        raise ValueError("--daily_shares must include agent_id and date")

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.loc[daily["date"].notna()].copy()
    daily["epi_phase"] = pd.to_numeric(daily.get("epi_phase", 0), errors="coerce").fillna(0).astype(np.int16)
    daily["travel_frac"] = pd.to_numeric(daily.get("travel_frac", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    daily["day_hours"] = pd.to_numeric(daily.get("day_hours", 24.0), errors="coerce").fillna(24.0).clip(0.0, 24.0)

    cat_cols = sorted(
        [c for c in daily.columns if c.startswith("cat_")],
        key=lambda c: int(c.split("_", 1)[1]),
    )
    if not cat_cols:
        raise ValueError("--daily_shares must include cat_* share columns")

    for c in cat_cols:
        daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    daily["weekday_bucket"] = _weekday_bucket(daily["date"], bucket_mode)
    daily["gender_norm"] = _normalize_gender(daily["gender"]) if "gender" in daily.columns else "U"
    daily["age_idx"] = _infer_age_idx(daily)
    daily["demo_group"] = daily["gender_norm"] + "_age" + daily["age_idx"].astype(str)
    return daily, len(cat_cols)


def _candidate_keys(
    *,
    agent_id: str,
    demo_group: str,
    cat_idx: int,
    weekday_bucket: str,
    epi_phase: int,
) -> list[tuple[str, str, int, str, int]]:
    return [
        ("agent_phase_bucket", agent_id, cat_idx, weekday_bucket, epi_phase),
        ("agent_phase_allweek", agent_id, cat_idx, "all", epi_phase),
        ("agent_allphase_bucket", agent_id, cat_idx, weekday_bucket, -1),
        ("agent_allphase_allweek", agent_id, cat_idx, "all", -1),
        ("group_phase_bucket", demo_group, cat_idx, weekday_bucket, epi_phase),
        ("group_phase_allweek", demo_group, cat_idx, "all", epi_phase),
        ("group_allphase_bucket", demo_group, cat_idx, weekday_bucket, -1),
        ("group_allphase_allweek", demo_group, cat_idx, "all", -1),
        ("global_phase_bucket", GLOBAL_SCOPE_ID, cat_idx, weekday_bucket, epi_phase),
        ("global_phase_allweek", GLOBAL_SCOPE_ID, cat_idx, "all", epi_phase),
        ("global_allphase_bucket", GLOBAL_SCOPE_ID, cat_idx, weekday_bucket, -1),
        ("global_allphase_allweek", GLOBAL_SCOPE_ID, cat_idx, "all", -1),
    ]


def _allocate_rows(
    daily: pd.DataFrame,
    *,
    home_cat_idx: int,
    templates: dict[tuple[str, str, int, str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    records: list[dict] = []
    total_cat_hours = 0.0
    unresolved_cat_hours = 0.0
    unresolved_cat_rows = 0

    cat_cols = sorted(
        [c for c in daily.columns if c.startswith("cat_")],
        key=lambda c: int(c.split("_", 1)[1]),
    )
    alloc_cat_cols = [c for c in cat_cols if int(c.split("_", 1)[1]) != int(home_cat_idx)]

    for row in daily.itertuples(index=False):
        stay_hours = float((1.0 - float(row.travel_frac)) * float(row.day_hours))
        if stay_hours <= 0.0:
            continue

        agent_id = str(row.agent_id)
        weekday_bucket = str(row.weekday_bucket)
        demo_group = str(row.demo_group)
        epi_phase = int(row.epi_phase)
        date_value = pd.Timestamp(row.date)

        for cat_col in alloc_cat_cols:
            cat_idx = int(cat_col.split("_", 1)[1])
            cat_share = float(getattr(row, cat_col, 0.0))
            cat_hours = max(0.0, stay_hours * cat_share)
            if cat_hours <= 1e-12:
                continue

            total_cat_hours += cat_hours
            chosen = None
            chosen_key = None
            for key in _candidate_keys(
                agent_id=agent_id,
                demo_group=demo_group,
                cat_idx=cat_idx,
                weekday_bucket=weekday_bucket,
                epi_phase=epi_phase,
            ):
                if key in templates:
                    chosen = templates[key]
                    chosen_key = key
                    break

            if chosen is None:
                unresolved_cat_hours += cat_hours
                unresolved_cat_rows += 1
                records.append(
                    {
                        "agent_id": agent_id,
                        "date": date_value.strftime("%Y-%m-%d"),
                        "epi_phase": epi_phase,
                        "weekday_bucket": weekday_bucket,
                        "cat_idx": cat_idx,
                        "poi_id_num": int(UNRESOLVED_POI_BASE - cat_idx),
                        "cat_hours": cat_hours,
                        "allocated_hours": cat_hours,
                        "template_level_name": "unresolved",
                        "template_level_rank": 999,
                        "template_scope_type": "none",
                        "template_scope_id": "",
                        "template_rank": 1,
                        "template_weight": 1.0,
                        "template_coverage_topk": 0.0,
                    }
                )
                continue

            level_name, scope_id, _, _, _ = chosen_key
            scope_type = level_name.split("_", 1)[0]
            level_rank = int(
                {
                    "agent_phase_bucket": 0,
                    "agent_phase_allweek": 1,
                    "agent_allphase_bucket": 2,
                    "agent_allphase_allweek": 3,
                    "group_phase_bucket": 4,
                    "group_phase_allweek": 5,
                    "group_allphase_bucket": 6,
                    "group_allphase_allweek": 7,
                    "global_phase_bucket": 8,
                    "global_phase_allweek": 9,
                    "global_allphase_bucket": 10,
                    "global_allphase_allweek": 11,
                }[level_name]
            )

            for tpl_row in chosen.itertuples(index=False):
                weight = float(tpl_row.weight)
                if weight <= 0.0:
                    continue
                records.append(
                    {
                        "agent_id": agent_id,
                        "date": date_value.strftime("%Y-%m-%d"),
                        "epi_phase": epi_phase,
                        "weekday_bucket": weekday_bucket,
                        "cat_idx": cat_idx,
                        "poi_id_num": int(tpl_row.poi_id_num),
                        "cat_hours": cat_hours,
                        "allocated_hours": cat_hours * weight,
                        "template_level_name": level_name,
                        "template_level_rank": level_rank,
                        "template_scope_type": scope_type,
                        "template_scope_id": scope_id,
                        "template_rank": int(tpl_row.rank),
                        "template_weight": weight,
                        "template_coverage_topk": float(tpl_row.coverage_topk),
                    }
                )

    allocated = pd.DataFrame.from_records(records)
    if allocated.empty:
        allocated = pd.DataFrame(
            columns=[
                "agent_id",
                "date",
                "epi_phase",
                "weekday_bucket",
                "cat_idx",
                "poi_id_num",
                "cat_hours",
                "allocated_hours",
                "template_level_name",
                "template_level_rank",
                "template_scope_type",
                "template_scope_id",
                "template_rank",
                "template_weight",
                "template_coverage_topk",
            ]
        )

    summary = {
        "total_cat_hours": float(total_cat_hours),
        "unresolved_cat_hours": float(unresolved_cat_hours),
        "unresolved_cat_rows": int(unresolved_cat_rows),
        "unresolved_share": float(unresolved_cat_hours / total_cat_hours) if total_cat_hours > 0 else 0.0,
    }
    return allocated, summary


def _build_summary(
    allocated: pd.DataFrame,
    *,
    daily_shares_path: str,
    templates_path: str,
    out_path: str,
    bucket_mode: str,
    home_cat_idx: int,
    alloc_summary: dict,
) -> dict:
    by_level = []
    if not allocated.empty:
        for level_name, g in allocated.groupby("template_level_name", sort=False):
            by_level.append(
                {
                    "template_level_name": str(level_name),
                    "rows": int(len(g)),
                    "allocated_hours": float(g["allocated_hours"].sum()),
                    "unique_poi": int(g["poi_id_num"].nunique()),
                }
            )

    return {
        "daily_shares": str(Path(daily_shares_path)),
        "templates": str(Path(templates_path)),
        "out_path": str(Path(out_path)),
        "bucket_mode": str(bucket_mode),
        "home_cat_idx": int(home_cat_idx),
        "n_rows_allocated": int(len(allocated)),
        "n_agent_days": int(allocated[["agent_id", "date"]].drop_duplicates().shape[0]) if not allocated.empty else 0,
        "levels_used": by_level,
        **alloc_summary,
    }


def _write_summary(summary: dict, summary_json: str) -> None:
    out = Path(summary_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Allocate category-level stay hours from daily_shares to concrete POIs using multi-level templates."
    )
    ap.add_argument("--daily_shares", required=True, help="Path to daily_shares.csv or daily_shares_cf.csv")
    ap.add_argument("--templates", required=True, help="Path to POI allocation templates (.csv or .parquet)")
    ap.add_argument("--out_path", required=True, help="Output allocated table (.csv or .parquet)")
    ap.add_argument("--feature_meta", default=None, help="Optional feature_meta.json for home_cat_idx")
    ap.add_argument("--summary_json", default=None, help="Optional summary JSON path")
    ap.add_argument(
        "--bucket_mode",
        choices=["weekday_weekend", "full_week"],
        default="weekday_weekend",
        help="Must match the template bucket mode",
    )
    args = ap.parse_args()

    home_cat_idx = _load_home_cat_idx(args.feature_meta)
    templates, fallback_order = _load_templates(args.templates)
    daily, _ = _load_daily_rows(args.daily_shares, args.bucket_mode)
    allocated, alloc_summary = _allocate_rows(
        daily,
        home_cat_idx=home_cat_idx,
        templates=templates,
    )
    _write_table(allocated, args.out_path)

    summary_json = args.summary_json
    if not summary_json:
        out = Path(args.out_path)
        summary_json = str(out.with_suffix(out.suffix + ".summary.json"))
    summary = _build_summary(
        allocated,
        daily_shares_path=args.daily_shares,
        templates_path=args.templates,
        out_path=args.out_path,
        bucket_mode=args.bucket_mode,
        home_cat_idx=home_cat_idx,
        alloc_summary=alloc_summary,
    )
    summary["fallback_order"] = fallback_order
    _write_summary(summary, summary_json)

    print("[OK] Wrote:")
    print(f"- {Path(args.out_path)}")
    print(f"- {Path(summary_json)}")
    print(
        f"[info] allocated rows={len(allocated):,}  "
        f"total_cat_hours={alloc_summary['total_cat_hours']:.3f}  "
        f"unresolved_share={alloc_summary['unresolved_share']:.6f}"
    )


if __name__ == "__main__":
    main()
