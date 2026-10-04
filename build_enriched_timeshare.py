from __future__ import annotations
import warnings; warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse, csv, json, math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing as mp


def _write_df_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")


def _concat_csv_files(paths: List[Path], out_path: Path):
    if not paths:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with open(out_path, "w", encoding="utf-8", newline="") as fout:
        for path in sorted(paths):
            with open(path, "r", encoding="utf-8", newline="") as fin:
                header = fin.readline()
                if not header:
                    continue
                if not wrote_header:
                    fout.write(header)
                    wrote_header = True
                for line in fin:
                    fout.write(line)


def _write_failure_log(rows: List[Dict[str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["chain_file", "agent_key", "error_type", "error"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _agent_key_from_chain_path(path: Path) -> str:
    return path.stem.replace("_activity_chain", "")


def _read_first_agent_id(path: Path) -> str:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row is None or "agent_id" not in row:
        raise ValueError(f"{path} missing first-row agent_id")
    return str(row["agent_id"])

def _exists(p: str|Path) -> Path:
    p = Path(p)
    if not p.exists(): raise FileNotFoundError(str(p))
    return p

def _read_csv_any(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

def to_int(s: pd.Series, fill=-1) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill).astype(np.int64)

def parse_hms_col(s: pd.Series) -> np.ndarray:
    x = s.astype(str).str.extract(r"(?:(?P<h>\d+)h)?(?:(?P<m>\d+)m)?(?:(?P<s>\d+)s)?").fillna(0).astype(int)
    return (x["h"]*3600 + x["m"]*60 + x["s"]).values.astype(np.float64)


def _safe_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_int(v, default=-1):
    fv = _safe_float(v)
    if fv is None:
        return default
    try:
        return int(fv)
    except Exception:
        return default


def _parse_dt(v: str) -> datetime:
    return datetime.strptime(str(v).strip(), "%Y-%m-%d %H:%M:%S")

@dataclass
class IndexMaps:
    poi_name_to_idx: Dict[str,int]   
    mode_name_to_idx: Dict[str,int]  
    cat_ids: List[int]              
    mode_ids: List[int]             
    home_cat_idx: int            
    unknown_mode_idx: int            

def load_index_maps(path: Path) -> IndexMaps:
    j = json.loads(Path(path).read_text())
    P_raw = {str(k).strip().lower(): int(v) for k, v in j["POIS"].items()}
    M_raw = {str(k).strip().lower(): int(v) for k, v in j["MODES"].items()}

    if "home" not in P_raw:
        raise ValueError("index_maps.json POIS must contain 'home'.")

    cat_ids = sorted(set(P_raw.values()))
    mode_ids = sorted(set(M_raw.values()))
    home_idx = int(P_raw["home"])
    if "unknown" not in M_raw:
        raise ValueError("index_maps.json MODES must contain 'unknown'.")
    unk_idx = int(M_raw["unknown"])

    return IndexMaps(
        poi_name_to_idx=P_raw,
        mode_name_to_idx=M_raw,
        cat_ids=cat_ids,
        mode_ids=mode_ids,
        home_cat_idx=home_idx,
        unknown_mode_idx=unk_idx,
    )

def load_calendar(npi_csv: Path) -> pd.DataFrame:
    df = _read_csv_any(npi_csv)
    if "date" not in df: raise ValueError("npi_calendar.csv must include 'date'")
    if "epi_phase" not in df and "phase_id" in df: df = df.rename(columns={"phase_id":"epi_phase"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")
    if df["epi_phase"].dtype == object:
        name_to_id = {}
        df["phase_name"] = df["epi_phase"].astype(str).str.strip().str.lower()
        for n in df["phase_name"]:
            if n not in name_to_id: name_to_id[n] = len(name_to_id)
        df["epi_phase"] = df["phase_name"].map(name_to_id).astype(int)
    else:
        df["phase_name"] = df["epi_phase"].astype(int).map({0:"pre_npi",1:"lockdown",2:"phase1",3:"phase2"}).fillna("phase")
    if "days_since_npi" not in df:
        start = df["date"].min()
        df["days_since_npi"] = (pd.to_datetime(df["date"]) - pd.to_datetime(start)).dt.days
    if "days_since_phase" not in df:
        df["days_since_phase"] = 0
        for _, g in df.groupby("epi_phase"):
            df.loc[g.index, "days_since_phase"] = (pd.to_datetime(g["date"]) - pd.to_datetime(g["date"].min())).dt.days
    return df[["date","epi_phase","phase_name","days_since_phase","days_since_npi"]]

def load_home(home_csv: Path) -> pd.DataFrame:
    df = _read_csv_any(home_csv)
    need = {"agent_id","home_lat","home_lon"}
    if not need.issubset(df.columns): raise ValueError(f"--home_csv missing {need - set(df.columns)}")
    age_buckets = ["kids","teenagers","young adults","middle-aged adults","the elderly"]
    age_idx = df.get("age_group", "").astype(str).str.strip().str.lower().map({k:i for i,k in enumerate(age_buckets)})
    for i in range(5): df[f"age_{i}"] = (age_idx == i).astype(float)
    df["gender"] = df.get("gender","").astype(str).str.lower().map({"male":1.0,"m":1.0,"female":0.0,"f":0.0}).fillna(0.5)
    return df[["agent_id","home_lat","home_lon","gender","age_0","age_1","age_2","age_3","age_4"]].rename(columns={"lat":"home_lat","lon":"home_lon"})

def load_pois(pois_csv: Path, idx: IndexMaps) -> pd.DataFrame:
    df = _read_csv_any(pois_csv)
    if "poi_id_num" not in df: raise ValueError("--pois_csv must include 'poi_id_num'")
    cat_col = None
    for c in ["poi_category","category","poi_cat","poi_cat_name","cat_name"]:
        if c in df.columns: cat_col=c; break
    if cat_col is not None:
        df["cat_idx"] = df[cat_col].astype(str).str.strip().str.lower().map(idx.poi_name_to_idx)
    else:
        df["cat_idx"] = np.nan
    ren = {}
    for a,b in [("latitude","lat"),("Latitude","lat"),("long","lon"),("longitude","lon"),("Longitude","lon")]:
        if a in df.columns and b not in df.columns: ren[a]=b
    df = df.rename(columns=ren)
    df["poi_id_num"] = to_int(df["poi_id_num"])
    keep = ["poi_id_num","cat_idx"] + [c for c in ("lat","lon") if c in df.columns]
    return df[keep].copy()

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(x) for x in (lat1,lon1,lat2,lon2)): return np.nan
    R=6371.0088; from math import radians,sin,cos,atan2,sqrt
    p1=radians(float(lat1)); p2=radians(float(lat2))
    dphi=radians(float(lat2)-float(lat1)); dl=radians(float(lon2)-float(lon1))
    a=sin(dphi/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
    return R*(2*atan2(sqrt(a), sqrt(1-a)))

# per-agent worker 
def _process_agent(args: Tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, IndexMaps, float, Path]):
    p, pois, home, cal, idx, min_minutes, out_dir = args
    # Prefer the Python CSV parser for per-agent chain files. It is slower but
    # more robust than pandas' C parser on the problematic 911-agent runs.
    df = pd.read_csv(p, engine="python")
    if df.empty: return None
    need = {"agent_id","start_time","end_time","event_type","label","poi_id_num"}
    miss = need - set(df.columns)
    if miss: raise ValueError(f"{p.name}: missing {miss}")

    df["start_dt"] = pd.to_datetime(df["start_time"])
    df["end_dt"]   = pd.to_datetime(df["end_time"])
    dur_from_time = (df["end_dt"] - df["start_dt"]).dt.total_seconds().values
    if "duration(h/m/s)" in df.columns:
        dur_from_hms = parse_hms_col(df["duration(h/m/s)"])
        use_hms = (dur_from_hms > 0).mean() > 0.7
        df["duration_s"] = dur_from_hms if use_hms else dur_from_time
    else:
        df["duration_s"] = dur_from_time

    # split across midnight
    rows=[]
    for _,r in df.iterrows():
        s=r["start_dt"]; e=r["end_dt"]; cur=s
        while cur.date() < e.date():
            cut=pd.Timestamp(cur.date()+pd.Timedelta(days=1))
            rows.append({**r,"start_dt":cur,"end_dt":cut,"date":cur.date(),"duration_s":(cut-cur).total_seconds()})
            cur=cut
        rows.append({**r,"start_dt":cur,"end_dt":e,"date":cur.date(),"duration_s":(e-cur).total_seconds()})
    df = pd.DataFrame(rows)
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    df["label_norm"] = df["label"].astype(str).str.strip().str.lower()
    df["poi_id_num"] = to_int(df["poi_id_num"], fill=-1)

    df = df.merge(pois, on="poi_id_num", how="left", suffixes=("","_poi"))

    is_stay = df["event_type"].eq("stay")
    cat_idx = pd.Series(np.full(len(df), -1, dtype=np.int32), index=df.index)
    mapped = df.loc[is_stay, "label_norm"].map(idx.poi_name_to_idx)
    fallback = df.loc[is_stay, "cat_idx"] 
    cat_idx.loc[is_stay] = pd.to_numeric(mapped.fillna(fallback), errors="coerce").fillna(-1).astype(np.int32)

    valid_cat = set(idx.cat_ids)
    cat_idx.loc[~cat_idx.isin(valid_cat)] = -1
    df["cat_idx"] = cat_idx

    is_travel = df["event_type"].eq("travel")
    midx = pd.Series(np.full(len(df), -1, dtype=np.int32), index=df.index)
    mapped_m = df.loc[is_travel, "label_norm"].map(idx.mode_name_to_idx)
    midx.loc[is_travel] = pd.to_numeric(mapped_m, errors="coerce").fillna(idx.unknown_mode_idx).astype(np.int32)
    # keep only valid IDs from JSON (unknown is valid)
    valid_modes = set(idx.mode_ids)
    midx.loc[~midx.isin(valid_modes)] = idx.unknown_mode_idx
    df["mode_idx"] = midx

    df = df.merge(home, on="agent_id", how="left")

    df = df.sort_values(["agent_id","date","start_dt"]).reset_index(drop=True)
    is_stay   = df["event_type"].eq("stay")
    is_travel = df["event_type"].eq("travel")

    is_home_by_poi   = df["poi_id_num"].astype("Int64").eq(-1)
    is_home_by_label = df["label_norm"].eq("home")
    is_home_by_cat   = df["cat_idx"].astype("Int64").eq(idx.home_cat_idx)
    is_home_stay     = is_stay & (is_home_by_poi | is_home_by_label | is_home_by_cat)
    is_nonhome_stay  = is_stay & (~is_home_stay)

    # Fill coords for HOME stays so daily_geo gets (lat,lon) at home rows
    home_lat_missing = is_home_stay & df["lat"].isna()
    home_lon_missing = is_home_stay & df["lon"].isna()
    df.loc[home_lat_missing, "lat"] = df.loc[home_lat_missing, "home_lat"].to_numpy()
    df.loc[home_lon_missing, "lon"] = df.loc[home_lon_missing, "home_lon"].to_numpy()

    # Distances for non-home stays (home -> POI)
    df["distance_km"] = np.where(
        is_nonhome_stay,
        [haversine_km(a,b,c,d) for a,b,c,d in zip(df["home_lat"], df["home_lon"], df["lat"], df["lon"])],
        np.nan
    )

    # Distances for HOME stays = distance(home -> next non-home stay SAME DAY) iff any travel occurs in between
    grp = df.groupby(["agent_id","date"], sort=False, group_keys=False)
    # next non-home stay's lat/lon per row (bfill over the day sequence)
    df["__lat_cand"] = df["lat"].where(is_nonhome_stay, np.nan)
    df["__lon_cand"] = df["lon"].where(is_nonhome_stay, np.nan)
    df["__lat_next"] = grp["__lat_cand"].transform(lambda s: s.bfill())
    df["__lon_next"] = grp["__lon_cand"].transform(lambda s: s.bfill())
    # cumulative travel count; compare value at current vs at next non-home stay
    df["__travel_cum"] = grp["event_type"].transform(lambda s: (s=="travel").cumsum())
    df["__t_cum_cand"] = df["__travel_cum"].where(is_nonhome_stay, np.nan)
    df["__t_cum_next"] = grp["__t_cum_cand"].transform(lambda s: s.bfill())
    cond = is_home_stay & df["__lat_next"].notna() & df["__lon_next"].notna() & (df["__t_cum_next"] > df["__travel_cum"])
    if cond.any():
        df.loc[cond, "distance_km"] = [
            haversine_km(a,b,c,d)
            for a,b,c,d in zip(
                df.loc[cond, "home_lat"], df.loc[cond, "home_lon"],
                df.loc[cond, "__lat_next"], df.loc[cond, "__lon_next"]
            )
        ]
    # cleanup temps
    df.drop(columns=["__lat_cand","__lon_cand","__lat_next","__lon_next","__travel_cum","__t_cum_cand","__t_cum_next"], inplace=True)

    if min_minutes > 0:
        df = df[df["duration_s"] >= 60.0*float(min_minutes)]
        if df.empty: return None

    df = df.merge(cal, on="date", how="left")
    df["start_time"] = df["start_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["end_time"]   = df["end_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # save events
    ev_cols = ["agent_id","date","start_time","end_time","event_type","label","duration_s",
               "poi_id_num","cat_idx","mode_idx","distance_km","epi_phase","phase_name",
               "days_since_phase","days_since_npi","gender","age_0","age_1","age_2","age_3","age_4"]
    aid = str(df["agent_id"].iloc[0])
    # clean dtypes
    df["epi_phase"] = pd.to_numeric(df["epi_phase"], errors="coerce").fillna(0).astype(np.int16)
    df["days_since_phase"] = pd.to_numeric(df["days_since_phase"], errors="coerce").fillna(0).astype(np.int32)
    df["days_since_npi"] = pd.to_numeric(df["days_since_npi"], errors="coerce").fillna(0).astype(np.int32)
    df["cat_idx"] = pd.to_numeric(df["cat_idx"], errors="coerce").fillna(-1).astype(np.int16)
    df["mode_idx"] = pd.to_numeric(df["mode_idx"], errors="coerce").fillna(-1).astype(np.int16)
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce").fillna(0.0).round(1)
    if "distance_km" in df.columns:
        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").round(3)

    event_path = out_dir/"events"/f"{aid}_events_enriched.csv"
    _write_df_csv(df[ev_cols], event_path)

    # daily shares (exact ID sets from JSON)
    g = df.groupby(["agent_id","date"], dropna=False)
    tot  = g["duration_s"].sum().rename("tot")
    trav = g.apply(lambda x: x.loc[x["event_type"]=="travel","duration_s"].sum()).rename("trav")

    # categories (sum stays with cat_idx == k; ignore -1)
    cat_cols=[]
    for k in idx.cat_ids:
        s = g.apply(lambda x: x.loc[(x["event_type"]=="stay") & (x["cat_idx"]==k), "duration_s"].sum())
        cat_cols.append(s.rename(f"cat_{k}"))
    cat_df = pd.concat(cat_cols, axis=1).div(tot.replace(0,np.nan), axis=0).fillna(0.0)

    # modes (sum travel with mode_idx == k)
    mode_cols=[]
    for k in idx.mode_ids:
        s = g.apply(lambda x: x.loc[(x["event_type"]=="travel") & (x["mode_idx"]==k), "duration_s"].sum())
        mode_cols.append(s.rename(f"mode_{k}"))
    mode_df = pd.concat(mode_cols, axis=1).div(trav.replace(0,np.nan), axis=0).fillna(0.0)

    travel_frac = (trav / tot.replace(0,np.nan)).fillna(0.0).rename("travel_frac")
    demo = g[["epi_phase","days_since_phase","days_since_npi","gender","age_0","age_1","age_2","age_3","age_4"]].first()
    dow = pd.to_datetime(tot.index.get_level_values("date")).weekday.values
    dow_df = pd.DataFrame(index=tot.index)
    for k in range(7): dow_df[f"wk_{k}"] = (dow==k).astype(float)

    daily = pd.concat([cat_df, mode_df, travel_frac, demo, dow_df], axis=1).reset_index()

    # (A) Trip distances: previous stay → current stay (same agent, same day)
    stay_df_all = df[df["event_type"] == "stay"].copy()
    # Ensure per-day temporal order for stays
    stay_df_all = stay_df_all.sort_values(["agent_id","date","start_dt"]).reset_index(drop=True)
    # Previous stay coordinates within the same (agent, date)
    stay_df_all["prev_lat"] = stay_df_all.groupby(["agent_id","date"])["lat"].shift(1)
    stay_df_all["prev_lon"] = stay_df_all.groupby(["agent_id","date"])["lon"].shift(1)
    # Haversine for stay-to-stay hops
    stay_df_all["trip_km"] = [
        haversine_km(a, b, c, d)
        for a, b, c, d in zip(stay_df_all["prev_lat"], stay_df_all["prev_lon"],
                            stay_df_all["lat"],      stay_df_all["lon"])
    ]

    # Aggregate trip distances per agent×date
    trip_agg = (stay_df_all
        .groupby(["agent_id","date"], dropna=False)["trip_km"]
        .agg(total_trip_km="sum", mean_trip_km="mean", max_trip_km="max", n_trips=lambda s: s.notna().sum())
        .reset_index()
        .fillna(0.0)
    )

    # (B) Distance-from-home aggregates for non-home stays (we already computed df['distance_km'])
    nonhome_mask = (df["event_type"].eq("stay")) & (~(df["poi_id_num"].astype("Int64").eq(-1)
                    | df["label_norm"].eq("home")
                    | df["cat_idx"].astype("Int64").eq(idx.home_cat_idx)))
    dist_home_agg = (df.loc[nonhome_mask, ["agent_id","date","distance_km"]]
        .groupby(["agent_id","date"], dropna=False)["distance_km"]
        .agg(mean_nonhome_dist_km="mean", max_nonhome_dist_km="max")
        .reset_index()
        .fillna(0.0)
    )

    # Merge into daily
    daily = daily.merge(trip_agg, on=["agent_id","date"], how="left")
    daily = daily.merge(dist_home_agg, on=["agent_id","date"], how="left")

    for c in ["total_trip_km","mean_trip_km","max_trip_km",
            "mean_nonhome_dist_km","max_nonhome_dist_km"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0.0)
        daily[c] = daily[c].clip(lower=0.0, upper=200.0)
        daily[f"log_{c}"] = np.log1p(daily[c])

    # Fill any missing with zeros
    for c in ["total_trip_km","mean_trip_km","max_trip_km","n_trips","mean_nonhome_dist_km","max_nonhome_dist_km"]:
        if c not in daily.columns:
            daily[c] = 0.0


    daily[["total_trip_km","mean_trip_km","max_trip_km","mean_nonhome_dist_km","max_nonhome_dist_km"]] = \
        daily[["total_trip_km","mean_trip_km","max_trip_km","mean_nonhome_dist_km","max_nonhome_dist_km"]].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0)
    daily["n_trips"] = pd.to_numeric(daily["n_trips"], errors="coerce").fillna(0).astype(np.int16)
 
    # Ensure date sorting within each agent
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["agent_id","date"]).reset_index(drop=True)

    grp = daily.groupby("agent_id", group_keys=False)

    # Travel fraction lags
    daily["prev_travel_frac"] = grp["travel_frac"].shift(1)
    daily["prev7_travel_frac_mean"] = grp["travel_frac"].apply(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )

    # Home share lags (if the home category exists)
    home_col = f"cat_{idx.home_cat_idx}"
    if home_col in daily.columns:
        daily["prev_home_share"] = grp[home_col].shift(1)
        daily["prev7_home_share_mean"] = grp[home_col].apply(
            lambda s: s.shift(1).rolling(7, min_periods=1).mean()
        )
    else:
        daily["prev_home_share"] = 0.0
        daily["prev7_home_share_mean"] = 0.0

    # Clean NaNs introduced by shifts/rolling
    for c in ["prev_travel_frac","prev7_travel_frac_mean","prev_home_share","prev7_home_share_mean"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0.0)

    # Keep 'date' as date (not Timestamp) to match downstream expectations
    daily["date"] = daily["date"].dt.date

    # pick up to 4 non-home category ids
    topK = 4
    lag_cat_ids = [k for k in idx.cat_ids if k != idx.home_cat_idx][:topK]

    for k in lag_cat_ids:
        col = f"cat_{k}"
        if col in daily.columns:
            daily[f"prev_cat_{k}"] = grp[col].shift(1)
            daily[f"prev7_cat_{k}_mean"] = grp[col].apply(
                lambda s: s.shift(1).rolling(7, min_periods=1).mean()
            )
            daily[f"prev_cat_{k}"] = pd.to_numeric(daily[f"prev_cat_{k}"], errors="coerce").fillna(0.0)
            daily[f"prev7_cat_{k}_mean"] = pd.to_numeric(daily[f"prev7_cat_{k}_mean"], errors="coerce").fillna(0.0)

     # bring total seconds and compute observed coverage in hours
    tot_df = tot.rename("tot_s").reset_index()
    daily = daily.merge(tot_df[["agent_id","date","tot_s"]], on=["agent_id","date"], how="left")
    daily["day_hours"] = (pd.to_numeric(daily["tot_s"], errors="coerce").fillna(0.0) / 3600.0).astype(float)
    daily.drop(columns=["tot_s"], inplace=True)

    # final column order (exact IDs)
    extra_cols = [
        "total_trip_km","mean_trip_km","max_trip_km","n_trips",
        "mean_nonhome_dist_km","max_nonhome_dist_km",
        # log versions
        "log_total_trip_km","log_mean_trip_km","log_max_trip_km",
        "log_mean_nonhome_dist_km","log_max_nonhome_dist_km",
        # temporal lags
        "prev_travel_frac","prev7_travel_frac_mean",
        "prev_home_share","prev7_home_share_mean",
    ]+ [f"prev_cat_{k}" for k in lag_cat_ids] \
        + [f"prev7_cat_{k}_mean" for k in lag_cat_ids]

    daily_cols = (
        ["agent_id","date","epi_phase","days_since_phase","days_since_npi"] +
        [f"cat_{i}" for i in idx.cat_ids] +
        [f"mode_{i}" for i in idx.mode_ids] +
        ["travel_frac"] +
        [f"wk_{i}" for i in range(7)] +
        ["age_0","age_1","age_2","age_3","age_4","gender","day_hours"] +
        extra_cols
    )


    for c in daily_cols:
        if c not in daily.columns:
            daily[c] = 0.0
    daily = daily[daily_cols].copy()

    # integer categoricals
    for c in ["epi_phase","days_since_phase","days_since_npi"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0).astype(np.int16)
    for k in range(7):
        daily[f"wk_{k}"] = pd.to_numeric(daily[f"wk_{k}"], errors="coerce").fillna(0).astype(np.int8)
    for k in range(5):
        daily[f"age_{k}"] = pd.to_numeric(daily[f"age_{k}"], errors="coerce").fillna(0).astype(np.int8)
    daily["gender"] = pd.to_numeric(daily["gender"], errors="coerce").fillna(0).astype(np.int8)

    # shares & coverage
    share_cols = [*(f"cat_{i}" for i in idx.cat_ids),
                  *(f"mode_{i}" for i in idx.mode_ids),
                  "travel_frac"]
    daily[share_cols] = daily[share_cols].clip(0, 1).astype(np.float64).round(6)
    daily["day_hours"] = pd.to_numeric(daily["day_hours"], errors="coerce").fillna(0.0).clip(0, 24).round(2)

    daily_path = out_dir/"daily"/f"{aid}_daily.csv"
    _write_df_csv(daily, daily_path)

    # geo (stays only)
    stay_df = df[df["event_type"]=="stay"].copy()
    geo_cols = ["agent_id","date","poi_id_num","cat_idx","epi_phase","duration_s"]
    if "lat" in stay_df.columns and "lon" in stay_df.columns:
        home_rows = stay_df["poi_id_num"].astype("Int64").eq(-1) | stay_df["cat_idx"].astype("Int64").eq(idx.home_cat_idx)
        stay_home_lat_missing = home_rows & stay_df["lat"].isna()
        stay_home_lon_missing = home_rows & stay_df["lon"].isna()
        stay_df.loc[stay_home_lat_missing, "lat"] = stay_df.loc[stay_home_lat_missing, "home_lat"].to_numpy()
        stay_df.loc[stay_home_lon_missing, "lon"] = stay_df.loc[stay_home_lon_missing, "home_lon"].to_numpy()
        geo_cols += ["lat", "lon"]
    geo = (stay_df[geo_cols]
        .groupby([c for c in geo_cols if c != "duration_s"], dropna=False)["duration_s"]
        .sum().reset_index())

    geo["duration_s"] = pd.to_numeric(geo["duration_s"], errors="coerce").fillna(0.0).round(1)
    if "lat" in geo.columns:
        geo["lat"] = pd.to_numeric(geo["lat"], errors="coerce")
        geo["lon"] = pd.to_numeric(geo["lon"], errors="coerce")

    geo_path = out_dir/"geo"/f"{aid}_daily_geo.csv"
    _write_df_csv(geo, geo_path)

    return daily_path, geo_path


def _process_agent_safe(args: Tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, IndexMaps, float, Path]):
    p, pois, home, cal, idx, min_minutes, out_dir = args

    home_lookup = {}
    for r in home.itertuples(index=False):
        home_lookup[str(r.agent_id)] = {
            "home_lat": _safe_float(r.home_lat),
            "home_lon": _safe_float(r.home_lon),
            "gender": _safe_float(r.gender) if _safe_float(r.gender) is not None else 0.0,
            "age_0": float(r.age_0),
            "age_1": float(r.age_1),
            "age_2": float(r.age_2),
            "age_3": float(r.age_3),
            "age_4": float(r.age_4),
        }

    poi_lookup = {}
    for r in pois.itertuples(index=False):
        poi_lookup[int(r.poi_id_num)] = {
            "cat_idx": _safe_int(getattr(r, "cat_idx", -1), default=-1),
            "lat": _safe_float(getattr(r, "lat", None)),
            "lon": _safe_float(getattr(r, "lon", None)),
        }

    cal_lookup = {}
    for r in cal.itertuples(index=False):
        cal_lookup[r.date] = {
            "epi_phase": int(r.epi_phase),
            "phase_name": str(r.phase_name),
            "days_since_phase": int(r.days_since_phase),
            "days_since_npi": int(r.days_since_npi),
        }

    raw_rows = []
    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(row)
    if not raw_rows:
        return None

    events = []
    for row in raw_rows:
        start_dt = _parse_dt(row["start_time"])
        end_dt = _parse_dt(row["end_time"])
        if end_dt < start_dt:
            continue
        cur = start_dt
        while cur.date() < end_dt.date():
            cut = datetime.combine(cur.date() + timedelta(days=1), datetime.min.time())
            events.append({**row, "start_dt": cur, "end_dt": cut, "date": cur.date(), "duration_s": (cut - cur).total_seconds()})
            cur = cut
        events.append({**row, "start_dt": cur, "end_dt": end_dt, "date": cur.date(), "duration_s": (end_dt - cur).total_seconds()})

    if min_minutes > 0:
        events = [e for e in events if e["duration_s"] >= 60.0 * float(min_minutes)]
    if not events:
        return None

    aid = str(events[0]["agent_id"])
    home_info = home_lookup.get(aid, {
        "home_lat": None, "home_lon": None, "gender": 0.0,
        "age_0": 0.0, "age_1": 0.0, "age_2": 0.0, "age_3": 0.0, "age_4": 0.0,
    })

    valid_cat = set(idx.cat_ids)
    valid_modes = set(idx.mode_ids)
    by_date = defaultdict(list)

    for ev in events:
        ev["event_type"] = str(ev.get("event_type", "")).strip().lower()
        ev["label"] = str(ev.get("label", ""))
        ev["label_norm"] = ev["label"].strip().lower()
        ev["poi_id_num"] = _safe_int(ev.get("poi_id_num"), default=-1)
        poi_info = poi_lookup.get(ev["poi_id_num"], {})
        ev["lat"] = poi_info.get("lat")
        ev["lon"] = poi_info.get("lon")
        ev["home_lat"] = home_info["home_lat"]
        ev["home_lon"] = home_info["home_lon"]
        ev["gender"] = home_info["gender"]
        for k in range(5):
            ev[f"age_{k}"] = home_info[f"age_{k}"]

        if ev["event_type"] == "stay":
            mapped = idx.poi_name_to_idx.get(ev["label_norm"])
            cat_idx = mapped if mapped is not None else poi_info.get("cat_idx", -1)
            ev["cat_idx"] = int(cat_idx) if int(cat_idx) in valid_cat else -1
            ev["mode_idx"] = -1
        elif ev["event_type"] == "travel":
            mapped_m = idx.mode_name_to_idx.get(ev["label_norm"], idx.unknown_mode_idx)
            ev["mode_idx"] = int(mapped_m) if int(mapped_m) in valid_modes else idx.unknown_mode_idx
            ev["cat_idx"] = -1
        else:
            ev["cat_idx"] = -1
            ev["mode_idx"] = idx.unknown_mode_idx if idx.unknown_mode_idx in valid_modes else -1

        cal_info = cal_lookup.get(ev["date"], {"epi_phase": 0, "phase_name": "phase", "days_since_phase": 0, "days_since_npi": 0})
        ev.update(cal_info)
        by_date[ev["date"]].append(ev)

    events = []
    for day in sorted(by_date):
        day_events = sorted(by_date[day], key=lambda x: x["start_dt"])
        for ev in day_events:
            is_stay = ev["event_type"] == "stay"
            is_home_stay = is_stay and (
                ev["poi_id_num"] == -1 or ev["label_norm"] == "home" or ev["cat_idx"] == idx.home_cat_idx
            )
            ev["is_home_stay"] = is_home_stay
            ev["is_nonhome_stay"] = is_stay and (not is_home_stay)
            if is_home_stay:
                if ev["lat"] is None:
                    ev["lat"] = ev["home_lat"]
                if ev["lon"] is None:
                    ev["lon"] = ev["home_lon"]
            if ev["is_nonhome_stay"]:
                ev["distance_km"] = haversine_km(ev["home_lat"], ev["home_lon"], ev["lat"], ev["lon"])
            else:
                ev["distance_km"] = None
            ev["trip_km"] = None

        for i, ev in enumerate(day_events):
            if not ev["is_home_stay"]:
                continue
            travel_seen = False
            for nxt in day_events[i + 1:]:
                if nxt["event_type"] == "travel":
                    travel_seen = True
                if nxt["is_nonhome_stay"]:
                    if travel_seen:
                        ev["distance_km"] = haversine_km(ev["home_lat"], ev["home_lon"], nxt["lat"], nxt["lon"])
                    break

        stay_events = [ev for ev in day_events if ev["event_type"] == "stay"]
        for i in range(1, len(stay_events)):
            prev = stay_events[i - 1]
            cur = stay_events[i]
            cur["trip_km"] = haversine_km(prev["lat"], prev["lon"], cur["lat"], cur["lon"])

        events.extend(day_events)

    ev_cols = ["agent_id","date","start_time","end_time","event_type","label","duration_s",
               "poi_id_num","cat_idx","mode_idx","distance_km","epi_phase","phase_name",
               "days_since_phase","days_since_npi","gender","age_0","age_1","age_2","age_3","age_4"]
    event_path = out_dir / "events" / f"{aid}_events_enriched.csv"
    with open(event_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ev_cols)
        writer.writeheader()
        for ev in events:
            writer.writerow({
                "agent_id": aid,
                "date": ev["date"].isoformat(),
                "start_time": ev["start_dt"].strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": ev["end_dt"].strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": ev["event_type"],
                "label": ev["label"],
                "duration_s": round(float(ev["duration_s"]), 1),
                "poi_id_num": ev["poi_id_num"],
                "cat_idx": ev["cat_idx"],
                "mode_idx": ev["mode_idx"],
                "distance_km": "" if ev["distance_km"] is None or pd.isna(ev["distance_km"]) else round(float(ev["distance_km"]), 3),
                "epi_phase": ev["epi_phase"],
                "phase_name": ev["phase_name"],
                "days_since_phase": ev["days_since_phase"],
                "days_since_npi": ev["days_since_npi"],
                "gender": ev["gender"],
                "age_0": ev["age_0"],
                "age_1": ev["age_1"],
                "age_2": ev["age_2"],
                "age_3": ev["age_3"],
                "age_4": ev["age_4"],
            })

    daily_rows = []
    lag_cat_ids = [k for k in idx.cat_ids if k != idx.home_cat_idx][:4]
    hist_travel = []
    hist_home = []
    hist_cat = {k: [] for k in lag_cat_ids}

    for day in sorted(by_date):
        day_events = [ev for ev in events if ev["date"] == day]
        tot = sum(float(ev["duration_s"]) for ev in day_events)
        trav = sum(float(ev["duration_s"]) for ev in day_events if ev["event_type"] == "travel")
        first = day_events[0]
        row = {
            "agent_id": aid,
            "date": day.isoformat(),
            "epi_phase": int(first["epi_phase"]),
            "days_since_phase": int(first["days_since_phase"]),
            "days_since_npi": int(first["days_since_npi"]),
        }
        for k in idx.cat_ids:
            cat_dur = sum(float(ev["duration_s"]) for ev in day_events if ev["event_type"] == "stay" and ev["cat_idx"] == k)
            row[f"cat_{k}"] = 0.0 if tot <= 0 else min(max(cat_dur / tot, 0.0), 1.0)
        for k in idx.mode_ids:
            mode_dur = sum(float(ev["duration_s"]) for ev in day_events if ev["event_type"] == "travel" and ev["mode_idx"] == k)
            row[f"mode_{k}"] = 0.0 if trav <= 0 else min(max(mode_dur / trav, 0.0), 1.0)
        row["travel_frac"] = 0.0 if tot <= 0 else min(max(trav / tot, 0.0), 1.0)
        for wk in range(7):
            row[f"wk_{wk}"] = 1 if day.weekday() == wk else 0
        for k in range(5):
            row[f"age_{k}"] = int(float(first[f"age_{k}"]))
        row["gender"] = int(float(first["gender"]))
        row["day_hours"] = round(tot / 3600.0, 2)

        trip_vals = [float(ev["trip_km"]) for ev in day_events if ev.get("trip_km") is not None and not pd.isna(ev["trip_km"])]
        dist_vals = [float(ev["distance_km"]) for ev in day_events if ev["is_nonhome_stay"] and ev.get("distance_km") is not None and not pd.isna(ev["distance_km"])]
        row["total_trip_km"] = min(sum(trip_vals), 200.0)
        row["mean_trip_km"] = min((sum(trip_vals) / len(trip_vals)) if trip_vals else 0.0, 200.0)
        row["max_trip_km"] = min(max(trip_vals) if trip_vals else 0.0, 200.0)
        row["n_trips"] = len(trip_vals)
        row["mean_nonhome_dist_km"] = min((sum(dist_vals) / len(dist_vals)) if dist_vals else 0.0, 200.0)
        row["max_nonhome_dist_km"] = min(max(dist_vals) if dist_vals else 0.0, 200.0)
        row["log_total_trip_km"] = math.log1p(row["total_trip_km"])
        row["log_mean_trip_km"] = math.log1p(row["mean_trip_km"])
        row["log_max_trip_km"] = math.log1p(row["max_trip_km"])
        row["log_mean_nonhome_dist_km"] = math.log1p(row["mean_nonhome_dist_km"])
        row["log_max_nonhome_dist_km"] = math.log1p(row["max_nonhome_dist_km"])

        row["prev_travel_frac"] = hist_travel[-1] if hist_travel else 0.0
        row["prev7_travel_frac_mean"] = (sum(hist_travel[-7:]) / len(hist_travel[-7:])) if hist_travel else 0.0
        home_cur = row.get(f"cat_{idx.home_cat_idx}", 0.0)
        row["prev_home_share"] = hist_home[-1] if hist_home else 0.0
        row["prev7_home_share_mean"] = (sum(hist_home[-7:]) / len(hist_home[-7:])) if hist_home else 0.0
        for k in lag_cat_ids:
            row[f"prev_cat_{k}"] = hist_cat[k][-1] if hist_cat[k] else 0.0
            row[f"prev7_cat_{k}_mean"] = (sum(hist_cat[k][-7:]) / len(hist_cat[k][-7:])) if hist_cat[k] else 0.0

        hist_travel.append(row["travel_frac"])
        hist_home.append(home_cur)
        for k in lag_cat_ids:
            hist_cat[k].append(row.get(f"cat_{k}", 0.0))
        daily_rows.append(row)

    extra_cols = [
        "total_trip_km","mean_trip_km","max_trip_km","n_trips",
        "mean_nonhome_dist_km","max_nonhome_dist_km",
        "log_total_trip_km","log_mean_trip_km","log_max_trip_km",
        "log_mean_nonhome_dist_km","log_max_nonhome_dist_km",
        "prev_travel_frac","prev7_travel_frac_mean",
        "prev_home_share","prev7_home_share_mean",
    ] + [f"prev_cat_{k}" for k in lag_cat_ids] + [f"prev7_cat_{k}_mean" for k in lag_cat_ids]
    daily_cols = (
        ["agent_id","date","epi_phase","days_since_phase","days_since_npi"] +
        [f"cat_{i}" for i in idx.cat_ids] +
        [f"mode_{i}" for i in idx.mode_ids] +
        ["travel_frac"] +
        [f"wk_{i}" for i in range(7)] +
        ["age_0","age_1","age_2","age_3","age_4","gender","day_hours"] +
        extra_cols
    )
    daily_path = out_dir / "daily" / f"{aid}_daily.csv"
    with open(daily_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=daily_cols)
        writer.writeheader()
        for row in daily_rows:
            writer.writerow({c: row.get(c, 0.0) for c in daily_cols})

    geo_cols = ["agent_id","date","poi_id_num","cat_idx","epi_phase","duration_s","lat","lon"]
    geo_agg = defaultdict(float)
    geo_meta = {}
    for ev in events:
        if ev["event_type"] != "stay":
            continue
        key = (aid, ev["date"].isoformat(), ev["poi_id_num"], ev["cat_idx"], ev["epi_phase"], ev["lat"], ev["lon"])
        geo_agg[key] += float(ev["duration_s"])
        geo_meta[key] = {
            "agent_id": aid,
            "date": ev["date"].isoformat(),
            "poi_id_num": ev["poi_id_num"],
            "cat_idx": ev["cat_idx"],
            "epi_phase": ev["epi_phase"],
            "lat": ev["lat"],
            "lon": ev["lon"],
        }

    geo_path = out_dir / "geo" / f"{aid}_daily_geo.csv"
    with open(geo_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=geo_cols)
        writer.writeheader()
        for key in sorted(geo_meta.keys(), key=lambda x: (x[1], x[2], x[3])):
            row = dict(geo_meta[key])
            row["duration_s"] = round(geo_agg[key], 1)
            writer.writerow(row)

    return daily_path, geo_path


def _process_agent_entry(args: Tuple[Tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, IndexMaps, float, Path], bool]):
    work_args, use_safe = args
    chain_path = work_args[0]
    try:
        result = _process_agent_safe(work_args) if use_safe else _process_agent(work_args)
        return {
            "ok": True,
            "chain_file": chain_path.name,
            "agent_key": _agent_key_from_chain_path(chain_path),
            "result": result,
        }
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        return {
            "ok": False,
            "chain_file": chain_path.name,
            "agent_key": _agent_key_from_chain_path(chain_path),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains_dir", required=True)
    ap.add_argument("--pois_csv", required=True)
    ap.add_argument("--home_csv", required=True)
    ap.add_argument("--npi_csv", required=True)
    ap.add_argument("--index_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_minutes", type=float, default=0.0)
    ap.add_argument("--jobs", type=int, default=max(mp.cpu_count()-1, 1))
    ap.add_argument("--maxtasksperchild", type=int, default=8)
    ap.add_argument("--resume_existing", action="store_true")
    ap.add_argument("--only_file", default="", help="Optional single chain CSV filename to process, e.g. 999_activity_chain.csv")
    ap.add_argument("--safe_mode", action="store_true",
                    help="Use the robust CSV/DictReader worker for every file. Slower, but safer for malformed chains.")
    ap.add_argument("--skip_bad_files", action="store_true",
                    help="Log malformed chain files and continue building outputs from the remaining agents.")
    ap.add_argument("--failed_log", default="",
                    help="Optional CSV path for failed chain files. Defaults to <out_dir>/failed_chain_files.csv.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir/"events").mkdir(parents=True, exist_ok=True)
    (out_dir/"daily").mkdir(parents=True, exist_ok=True)
    (out_dir/"geo").mkdir(parents=True, exist_ok=True)

    idx = load_index_maps(_exists(args.index_json))
    cal = load_calendar(_exists(args.npi_csv))
    home = load_home(_exists(args.home_csv))
    pois = load_pois(_exists(args.pois_csv), idx)

    (out_dir/"feature_meta.json").write_text(json.dumps({
        "poi_name_to_idx": idx.poi_name_to_idx,
        "mode_name_to_idx": idx.mode_name_to_idx,
        "cat_ids": idx.cat_ids,
        "mode_ids": idx.mode_ids,
        "home_cat_idx": idx.home_cat_idx,
        "unknown_mode_idx": idx.unknown_mode_idx
    }, indent=2))

    chain_paths = sorted(Path(_exists(args.chains_dir)).glob("*.csv"))
    if args.only_file:
        target = Path(args.chains_dir) / str(args.only_file)
        chain_paths = [target]

    if args.resume_existing:
        filtered = []
        for p in chain_paths:
            agent_key = _read_first_agent_id(p)
            daily_done = (out_dir / "daily" / f"{agent_key}_daily.csv").exists()
            geo_done = (out_dir / "geo" / f"{agent_key}_daily_geo.csv").exists()
            if daily_done and geo_done:
                continue
            filtered.append(p)
        print(f"[builder] resume_existing kept {len(filtered)}/{len(chain_paths)} files to process")
        if len(filtered) <= 10:
            print("[builder] pending files:", [p.name for p in filtered])
        chain_paths = filtered

    work_args = [(p, pois, home, cal, idx, args.min_minutes, out_dir) for p in chain_paths]
    failed_rows: List[Dict[str, str]] = []
    failed_log = Path(args.failed_log).resolve() if str(args.failed_log).strip() else (out_dir / "failed_chain_files.csv")

    if int(args.jobs) <= 1 or len(chain_paths) <= 1:
        bar = tqdm(chain_paths, total=len(chain_paths), desc="Enriching", dynamic_ncols=True)
        use_safe_seq = bool(
            args.safe_mode
            or args.only_file
            or len(chain_paths) <= 1
            or (args.resume_existing and len(chain_paths) <= 16)
        )
        for p in bar:
            bar.set_postfix_str(p.name)
            print(f"[builder] processing {p.name}")
            try:
                if use_safe_seq:
                    _process_agent_safe((p, pois, home, cal, idx, args.min_minutes, out_dir))
                else:
                    _process_agent((p, pois, home, cal, idx, args.min_minutes, out_dir))
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                failure = {
                    "chain_file": p.name,
                    "agent_key": _agent_key_from_chain_path(p),
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
                failed_rows.append(failure)
                _write_failure_log(failed_rows, failed_log)
                if not args.skip_bad_files:
                    raise
                print(f"[builder] skipped bad chain file {p.name}: {exc}")
    else:
        worker_items = [(wa, bool(args.safe_mode)) for wa in work_args]
        with mp.Pool(processes=args.jobs, maxtasksperchild=max(1, int(args.maxtasksperchild))) as pool:
            for res in tqdm(pool.imap_unordered(_process_agent_entry, worker_items, chunksize=1), total=len(work_args),
                            desc="Enriching", dynamic_ncols=True):
                if res.get("ok") and res.get("result") is None:
                    continue
                if res.get("ok"):
                    continue
                failure = {
                    "chain_file": str(res.get("chain_file", "")),
                    "agent_key": str(res.get("agent_key", "")),
                    "error_type": str(res.get("error_type", "")),
                    "error": str(res.get("error", "")),
                }
                failed_rows.append(failure)
                _write_failure_log(failed_rows, failed_log)
                if not args.skip_bad_files:
                    raise SystemExit(
                        f"Failed on chain file {failure['chain_file']} ({failure['error_type']}): {failure['error']}\n"
                        f"Rerun with --resume_existing --jobs 1, or inspect {failed_log}."
                    )
                print(f"[builder] skipped bad chain file {failure['chain_file']}: {failure['error_type']}: {failure['error']}")
    daily_paths = sorted((out_dir / "daily").glob("*_daily.csv"))
    geo_paths = sorted((out_dir / "geo").glob("*_daily_geo.csv"))
    _concat_csv_files(daily_paths, out_dir/"daily_shares.csv")
    _concat_csv_files(geo_paths, out_dir/"daily_geo.csv")
    if failed_rows:
        print(f"[builder] completed with {len(failed_rows)} failed chain file(s); see {failed_log}")
    print(f"[builder] wrote {out_dir}")

if __name__ == "__main__":
    main()
