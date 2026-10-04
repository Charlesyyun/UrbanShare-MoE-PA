
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from tqdm.auto import tqdm

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ---------- helpers ----------
def _load_feature_meta(meta_path: Path) -> Dict:
    if not meta_path.exists():
        raise FileNotFoundError(f"feature_meta.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text())
    for k in ["cat_ids", "mode_ids", "home_cat_idx", "unknown_mode_idx"]:
        if k not in meta:
            raise ValueError(f"feature_meta.json missing '{k}'")
    meta["cat_ids"] = sorted(int(x) for x in meta["cat_ids"])
    meta["mode_ids"] = sorted(int(x) for x in meta["mode_ids"])
    meta["home_cat_idx"] = int(meta["home_cat_idx"])
    meta["unknown_mode_idx"] = int(meta["unknown_mode_idx"])
    return meta


def _require_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {name}: {missing}")


def _shard_agents(agent_ids: List[str], nshards: int) -> List[List[str]]:
    nshards = max(1, nshards)
    shards = [[] for _ in range(nshards)]
    for i, a in enumerate(sorted(agent_ids)):
        shards[i % nshards].append(a)
    return shards


def _chunk_reader(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    # iterator yields chunks; dtype hints left to pandas to infer for simplicity
    return pd.read_csv(path, chunksize=chunksize)


# ---------- worker ----------
def _process_shard(args):
    """
    Worker:
    - Filters incoming chunks to shard agent_ids
    - Writes shard-level daily POI + MODE hours CSVs (appending)
    - Builds shard-level per-agent-phase SUMS + COUNTS and pop-phase 24h SUMS + COUNTS
    - Returns paths to shard outputs
    """
    (shard_id, agent_ids, ds_path, meta, out_dir, chunksize) = args
    cat_ids = meta["cat_ids"]
    mode_ids_all = meta["mode_ids"]
    unk_mode = meta["unknown_mode_idx"]
    mode_ids = [m for m in mode_ids_all if m != unk_mode]  # exclude unknown

    shard_dir = Path(out_dir) / f"shard_{shard_id:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    f_poi_shard  = shard_dir / f"per_agent_daily_poi_hours_shard{shard_id:03d}.csv"
    f_mode_shard = shard_dir / f"per_agent_daily_mode_hours_shard{shard_id:03d}.csv"
    # aggregation temps
    poi_agg_parts = []
    mode_agg_parts = []
    count_parts = []
    pop24_sums_parts = []

    # header flags for append
    poi_written = False
    mode_written = False

    # Read in chunks, filter, compute and append
    for chunk in _chunk_reader(ds_path, chunksize):
        chunk["agent_id"] = chunk["agent_id"].astype(str)
        mask = chunk["agent_id"].isin(agent_ids)
        if not mask.any():
            continue
        df = chunk.loc[mask].copy()

        # normalize basics
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # ensure required columns exist in this chunk
        req = (
            ["agent_id", "date", "epi_phase", "day_hours", "travel_frac"]
            + [f"cat_{k}" for k in cat_ids]
            + [f"mode_{m}" for m in mode_ids_all]
        )
        _require_cols(df, req, "daily_shares chunk")

        # ---- per-agent daily POI hours ----
        poi_df = df[["agent_id", "date", "epi_phase", "day_hours"]].copy()
        poi_cols_h = []
        for k in cat_ids:
            s = pd.to_numeric(df[f"cat_{k}"], errors="coerce").fillna(0.0)
            h = f"cat_{k}h"
            poi_df[h] = df["day_hours"] * s
            poi_cols_h.append(h)
        poi_df["total_stay_h"] = poi_df[poi_cols_h].sum(axis=1)

        # append to shard file
        poi_df[["agent_id","date","epi_phase","day_hours","total_stay_h"] + poi_cols_h] \
            .to_csv(f_poi_shard, mode=("a" if poi_written else "w"), header=(not poi_written), index=False)
        poi_written = True

        # ---- per-agent daily MODE hours (exclude unknown) ----
        mode_df = df[["agent_id", "date", "epi_phase", "day_hours", "travel_frac"]].copy()
        mode_cols_h = []
        tf = pd.to_numeric(df["travel_frac"], errors="coerce").fillna(0.0)
        for m in mode_ids:
            s = pd.to_numeric(df[f"mode_{m}"], errors="coerce").fillna(0.0)
            h = f"mode_{m}h"
            mode_df[h] = df["day_hours"] * tf * s
            mode_cols_h.append(h)
        mode_df["total_travel_h"] = mode_df[mode_cols_h].sum(axis=1)

        mode_df[["agent_id","date","epi_phase","day_hours","total_travel_h"] + mode_cols_h] \
            .to_csv(f_mode_shard, mode=("a" if mode_written else "w"), header=(not mode_written), index=False)
        mode_written = True

        # ---- per-agent-phase aggregations (SUMS + COUNTS) ----
        # counts per (agent, phase)
        cnt = df.groupby(["agent_id","epi_phase"], dropna=False).size().rename("n_days").reset_index()
        count_parts.append(cnt)

        # sums for poi hours
        poi_sums = poi_df.groupby(["agent_id","epi_phase"], dropna=False)[poi_cols_h + ["total_stay_h"]].sum().reset_index()
        mode_sums = mode_df.groupby(["agent_id","epi_phase"], dropna=False)[mode_cols_h + ["total_travel_h"]].sum().reset_index()
        poi_agg_parts.append(poi_sums)
        mode_agg_parts.append(mode_sums)

        # ---- population phase 24h sums + counts ----
        base = df.copy()
        for k in cat_ids:
            base[f"cat_{k}h_24"] = 24.0 * pd.to_numeric(base[f"cat_{k}"], errors="coerce").fillna(0.0)
        for m in mode_ids:
            base[f"mode_{m}h_24"] = 24.0 * (
                pd.to_numeric(base["travel_frac"], errors="coerce").fillna(0.0)
                * pd.to_numeric(base[f"mode_{m}"], errors="coerce").fillna(0.0)
            )
        pop_poi_cols = [f"cat_{k}h_24" for k in cat_ids]
        pop_mode_cols = [f"mode_{m}h_24" for m in mode_ids]
        pop_sums = base.groupby("epi_phase", dropna=False)[pop_poi_cols + pop_mode_cols].sum().reset_index()
        pop_sums["n_rows"] = base.groupby("epi_phase", dropna=False).size().values
        pop24_sums_parts.append(pop_sums)

    # --- finalize shard-level aggregations ---
    # counts
    if count_parts:
        counts = pd.concat(count_parts, ignore_index=True) \
                  .groupby(["agent_id","epi_phase"], dropna=False)["n_days"].sum().reset_index()
    else:
        counts = pd.DataFrame(columns=["agent_id","epi_phase","n_days"])

    # sums
    if poi_agg_parts:
        poi_sums_all = pd.concat(poi_agg_parts, ignore_index=True) \
                         .groupby(["agent_id","epi_phase"], dropna=False).sum().reset_index()
    else:
        poi_sums_all = pd.DataFrame(columns=["agent_id","epi_phase"])
    if mode_agg_parts:
        mode_sums_all = pd.concat(mode_agg_parts, ignore_index=True) \
                          .groupby(["agent_id","epi_phase"], dropna=False).sum().reset_index()
    else:
        mode_sums_all = pd.DataFrame(columns=["agent_id","epi_phase"])

    # merge sums + counts, compute means
    agent_phase = counts.copy()
    if not poi_sums_all.empty:
        agent_phase = agent_phase.merge(poi_sums_all, on=["agent_id","epi_phase"], how="left")
    if not mode_sums_all.empty:
        agent_phase = agent_phase.merge(mode_sums_all, on=["agent_id","epi_phase"], how="left")
    agent_phase = agent_phase.fillna(0.0)

    # divide *_h sums by n_days to get means
    h_cols = [c for c in agent_phase.columns if c.endswith("h")]
    for c in h_cols:
        agent_phase[c] = np.where(agent_phase["n_days"] > 0, agent_phase[c] / agent_phase["n_days"], 0.0)
    # drop n_days in the shard output; the global file is concatenation across disjoint agent sets
    f_agent_phase_shard = shard_dir / f"per_agent_phase_summary_shard{shard_id:03d}.csv"
    agent_phase.drop(columns=["n_days"]).to_csv(f_agent_phase_shard, index=False)

    # population phase 24h sums+counts (to be merged globally then averaged)
    if pop24_sums_parts:
        pop24_sums = pd.concat(pop24_sums_parts, ignore_index=True) \
                        .groupby("epi_phase", dropna=False).sum().reset_index()
    else:
        pop24_sums = pd.DataFrame(columns=["epi_phase"])
    f_pop_sums_shard = shard_dir / f"population_phase_summary_24h_sums_shard{shard_id:03d}.csv"
    pop24_sums.to_csv(f_pop_sums_shard, index=False)

    return str(f_poi_shard), str(f_mode_shard), str(f_agent_phase_shard), str(f_pop_sums_shard)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Multiprocessing hours summaries from daily_shares.csv")
    ap.add_argument("--daily_shares", required=True, help="Path to daily_shares.csv")
    ap.add_argument("--feature_meta", default="", help="Path to feature_meta.json (defaults alongside daily_shares)")
    ap.add_argument("--out_dir", default="out_observed_hours_mp", help="Directory for output CSVs")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count()-1))
    ap.add_argument("--chunksize", type=int, default=500_000, help="CSV chunk size per worker when streaming")
    args = ap.parse_args()

    ds_path = Path(args.daily_shares)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.feature_meta) if args.feature_meta else (ds_path.parent / "feature_meta.json")

    meta = _load_feature_meta(meta_path)

    # get unique agent ids (light scan)
    agent_ids = pd.read_csv(ds_path, usecols=["agent_id"])["agent_id"].astype(str).unique().tolist()
    shards = _shard_agents(agent_ids, max(1, int(args.workers)))

    # Launch workers
    work_args = [
        (i, shard, ds_path, meta, out_dir, int(args.chunksize))
        for i, shard in enumerate(shards)
        if len(shard) > 0
    ]
    results = []
    if len(work_args) == 1:
        results = [_process_shard(work_args[0])]
    else:
        with Pool(processes=len(work_args)) as pool:
            for res in tqdm(pool.imap_unordered(_process_shard, work_args), total=len(work_args), desc="Shards"):
                results.append(res)

    # ---------- Merge shard outputs into finals ----------
    poi_files  = [Path(r[0]) for r in results if r]
    mode_files = [Path(r[1]) for r in results if r]
    ap_files   = [Path(r[2]) for r in results if r]
    pop_sum_files = [Path(r[3]) for r in results if r]

    f_poi_final  = out_dir / "per_agent_daily_poi_hours.csv"
    f_mode_final = out_dir / "per_agent_daily_mode_hours.csv"
    f_ap_final   = out_dir / "per_agent_phase_summary.csv"
    f_pop_final  = out_dir / "population_phase_summary_24h.csv"

    # concat daily files
    if f_poi_final.exists(): f_poi_final.unlink()
    if f_mode_final.exists(): f_mode_final.unlink()

    for i, p in enumerate(tqdm(poi_files, desc="Merging POI daily")):
        if p.exists() and p.stat().st_size > 0:
            df = pd.read_csv(p)
            df.to_csv(f_poi_final, mode=("a" if i>0 else "w"), header=(i==0), index=False)

    for i, p in enumerate(tqdm(mode_files, desc="Merging MODE daily")):
        if p.exists() and p.stat().st_size > 0:
            df = pd.read_csv(p)
            df.to_csv(f_mode_final, mode=("a" if i>0 else "w"), header=(i==0), index=False)

    # per_agent_phase_summary: shards are disjoint by agent_id → just concat
    if ap_files:
        ap_frames = [pd.read_csv(p) for p in ap_files if p.exists() and p.stat().st_size > 0]
        if ap_frames:
            pd.concat(ap_frames, ignore_index=True).to_csv(f_ap_final, index=False)
        else:
            pd.DataFrame().to_csv(f_ap_final, index=False)
    else:
        pd.DataFrame().to_csv(f_ap_final, index=False)

    # population_phase_summary_24h: sum shard sums, then divide by total n_rows per phase
    if pop_sum_files:
        pop_frames = [pd.read_csv(p) for p in pop_sum_files if p.exists() and p.stat().st_size > 0]
        if pop_frames:
            pop_sum_all = pd.concat(pop_frames, ignore_index=True) \
                            .groupby("epi_phase", dropna=False).sum().reset_index()
            # compute means = sums / n_rows
            val_cols = [c for c in pop_sum_all.columns if c not in ("epi_phase","n_rows")]
            out = pop_sum_all[["epi_phase"] + val_cols].copy()
            for c in val_cols:
                out[c] = np.where(pop_sum_all["n_rows"] > 0, pop_sum_all[c] / pop_sum_all["n_rows"], 0.0)
            # add totals
            poi_24 = [c for c in out.columns if c.startswith("cat_") and c.endswith("h_24")]
            mode_24 = [c for c in out.columns if c.startswith("mode_") and c.endswith("h_24")]
            out["total_stay_h_24"]   = out[poi_24].sum(axis=1) if poi_24 else 0.0
            out["total_travel_h_24"] = out[mode_24].sum(axis=1) if mode_24 else 0.0
            out.to_csv(f_pop_final, index=False)
        else:
            pd.DataFrame().to_csv(f_pop_final, index=False)
    else:
        pd.DataFrame().to_csv(f_pop_final, index=False)

    print(
        "[OK] Wrote:\n"
        f"- {f_poi_final}\n"
        f"- {f_mode_final}\n"
        f"- {f_ap_final}\n"
        f"- {f_pop_final}"
    )


if __name__ == "__main__":
    main()
