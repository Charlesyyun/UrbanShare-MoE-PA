from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _ensure_cols(df: pd.DataFrame, cols: List[str], fill: float = 0.0):
    for c in cols:
        if c not in df.columns:
            df[c] = fill


def _coerce_gender(val) -> str:
    if pd.isna(val):
        return "U"
    if isinstance(val, (int, float)):
        return "M" if int(val) == 1 else ("F" if int(val) == 0 else "U")
    s = str(val).strip().upper()
    if s in ("M", "MALE", "1"):
        return "M"
    if s in ("F", "FEMALE", "0"):
        return "F"
    return "U"


def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)


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


def _vector_from_params_by_ids(param, ids: List[int], name: str) -> np.ndarray:
    if isinstance(param, dict):
        return np.array([float(param.get(int(i), 1.0)) for i in ids], dtype=float)
    arr = np.array(param, dtype=float)
    if arr.shape[0] == len(ids):
        return arr
    raise ValueError(
        f"hazard_params['{name}'] length={arr.shape[0]} != number of IDs={len(ids)}. "
        f"Provide as dict keyed by IDs or reorder/resize to match IDs."
    )


class SEIRTimeseriesPOI:
    def __init__(
        self,
        *,
        daily_df: pd.DataFrame,
        alloc_df: pd.DataFrame,
        feature_meta: Dict,
        hazard_params: Dict,
        latent_days: int,
        infectious_days: int,
        seed_pct: float,
        seed_agent_ids: Optional[List[str]],
        rng_seed: int,
        n_workers: int,
    ):
        self.df = daily_df.copy()
        self.alloc = alloc_df.copy()
        self.meta = feature_meta
        self.hp = hazard_params
        self.L = int(latent_days)
        self.D = int(infectious_days)
        self.seed_pct = float(seed_pct)
        self.seed_list = set(seed_agent_ids) if seed_agent_ids else None
        self.rng = np.random.default_rng(rng_seed)
        self.n_workers = max(1, int(n_workers))

        self.phase_beta_mult = self.hp.get("phase_beta_mult", {})
        self.phase_cat_mult = self.hp.get("phase_cat_mult", {})

        self.cat_ids: List[int] = self.meta["cat_ids"]
        self.mode_ids_all: List[int] = self.meta["mode_ids"]
        self.home_cat_idx: int = self.meta["home_cat_idx"]
        self.unknown_mode_idx: int = self.meta["unknown_mode_idx"]
        self.mode_ids_no_unk: List[int] = [m for m in self.mode_ids_all if m != self.unknown_mode_idx]

        self.CAT_COLS = [f"cat_{i}" for i in self.cat_ids]
        self.MODE_COLS_ALL = [f"mode_{i}" for i in self.mode_ids_all]
        self.MODE_COLS = [f"mode_{i}" for i in self.mode_ids_no_unk]

        _ensure_cols(self.df, ["day_hours", "travel_frac", "epi_phase"], 0.0)
        _ensure_cols(self.df, self.MODE_COLS_ALL, 0.0)
        self.df["agent_id"] = self.df["agent_id"].astype(str)
        self.df["date"] = pd.to_datetime(self.df["date"]).dt.date
        self.df["epi_phase"] = pd.to_numeric(self.df["epi_phase"], errors="coerce").fillna(-1).astype(np.int16)

        need_alloc = {"agent_id", "date", "poi_id_num", "cat_idx", "allocated_hours"}
        missing_alloc = sorted(need_alloc.difference(set(self.alloc.columns)))
        if missing_alloc:
            raise ValueError(f"allocated_poi_hours missing columns: {missing_alloc}")
        self.alloc["agent_id"] = self.alloc["agent_id"].astype(str)
        self.alloc["date"] = pd.to_datetime(self.alloc["date"]).dt.date
        self.alloc["poi_id_num"] = pd.to_numeric(self.alloc["poi_id_num"], errors="coerce").fillna(-1).astype(np.int64)
        self.alloc["cat_idx"] = pd.to_numeric(self.alloc["cat_idx"], errors="coerce").fillna(-1).astype(np.int16)
        self.alloc["allocated_hours"] = (
            pd.to_numeric(self.alloc["allocated_hours"], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
        self.alloc = self.alloc.loc[self.alloc["allocated_hours"] > 0.0].copy()

        self.agents = sorted(self.df["agent_id"].unique().tolist())
        self.agent_idx = {aid: i for i, aid in enumerate(self.agents)}
        self.n_agents = len(self.agents)
        self.dates = sorted(self.df["date"].unique().tolist())
        self.n_days = len(self.dates)

        if "gender" in self.df.columns:
            self.df["gender"] = self.df["gender"].map(_coerce_gender)
        else:
            self.df["gender"] = "U"
        self.has_age_oh = all(f"age_{i}" in self.df.columns for i in range(5))

        self.beta = float(self.hp.get("beta_base", 0.06))
        self.cat_mult = _vector_from_params_by_ids(
            self.hp.get("cat_mult", [1.0] * len(self.cat_ids)),
            self.cat_ids,
            "cat_mult",
        )
        self.mode_mult_all = _vector_from_params_by_ids(
            self.hp.get("mode_mult", [1.0] * len(self.mode_ids_all)),
            self.mode_ids_all,
            "mode_mult",
        )
        self.mode_mult = self.mode_mult_all[[self.mode_ids_all.index(m) for m in self.mode_ids_no_unk]]
        self.gender_mult = self.hp.get("gender_mult", {"M": 1.0, "F": 1.0, "U": 1.0})
        self.age_mult = np.array(self.hp.get("age_mult", [1.0] * 5), dtype=float)
        default_c_cat = np.ones(len(self.cat_ids), dtype=float)
        self.c_cat = _vector_from_params_by_ids(
            self.hp.get("contact_rate_cat", default_c_cat.tolist()),
            self.cat_ids,
            "contact_rate_cat",
        )
        self.import_p = float(self.hp.get("importation_prob", 0.0))

        self.state = np.zeros(self.n_agents, dtype=np.int8)  # 0=S,1=E,2=I,3=R
        self.timer = np.zeros(self.n_agents, dtype=np.int16)

    def _seed_initial(self):
        if self.seed_list:
            idx = [self.agent_idx[a] for a in self.agents if a in self.seed_list]
        else:
            k = max(1, int(round(self.seed_pct * self.n_agents)))
            idx = self.rng.choice(self.n_agents, size=k, replace=False)
        self.state[idx] = 2
        self.timer[idx] = self.D

    def _build_day_views(self, day):
        day_df = (
            self.df[self.df["date"] == day]
            .copy()
            .sort_values("agent_id")
            .reset_index(drop=True)
        )
        travel_frac = day_df["travel_frac"].to_numpy(dtype=float)
        day_hours = day_df["day_hours"].to_numpy(dtype=float)
        modes_all = day_df[self.MODE_COLS_ALL].to_numpy(dtype=float)
        mode_idx_map = [self.mode_ids_all.index(m) for m in self.mode_ids_no_unk]
        modes = modes_all[:, mode_idx_map]
        hours_mode_present = day_hours[:, None] * travel_frac[:, None] * modes

        genders_present = day_df["gender"].astype(str).to_numpy()
        if self.has_age_oh:
            age_mat = day_df[[f"age_{i}" for i in range(5)]].to_numpy(float)
            age_idx_present = np.where(age_mat.sum(axis=1) > 0, age_mat.argmax(axis=1), -1)
        else:
            age_idx_present = np.full(len(day_df), -1, dtype=int)

        idx_present = [self.agent_idx[a] for a in day_df["agent_id"].tolist()]
        nM = len(self.mode_ids_no_unk)
        full_hours_mode = np.zeros((self.n_agents, nM), dtype=float)
        full_gender = np.array(["U"] * self.n_agents, dtype=object)
        full_age = np.full(self.n_agents, -1, dtype=int)
        full_hours_mode[idx_present, :] = hours_mode_present
        full_gender[idx_present] = genders_present
        full_age[idx_present] = age_idx_present

        day_alloc = self.alloc[self.alloc["date"] == day].copy()
        if not day_alloc.empty:
            day_alloc["agent_idx"] = day_alloc["agent_id"].map(self.agent_idx)
            day_alloc = day_alloc.loc[day_alloc["agent_idx"].notna()].copy()
            day_alloc["agent_idx"] = day_alloc["agent_idx"].astype(int)
        return day_df, day_alloc, full_hours_mode, (full_gender, full_age)

    def _cat_risk_today(self, day_phase: int) -> np.ndarray:
        cat_mult_today = self.cat_mult.copy()
        phase_key = str(day_phase)
        if phase_key in self.phase_cat_mult:
            overrides = self.phase_cat_mult[phase_key]
            for k_str, v in overrides.items():
                k = int(k_str)
                if k in self.cat_ids:
                    pos = self.cat_ids.index(k)
                    cat_mult_today[pos] = float(v)
        return cat_mult_today * self.c_cat

    def _compute_mode_prevalence(self, hours_mode: np.ndarray, infectious_mask: np.ndarray):
        presence_mode = hours_mode.sum(axis=0)
        inf_presence_mode = hours_mode[infectious_mask].sum(axis=0)
        p_mode = _safe_div(inf_presence_mode, presence_mode)
        return p_mode, presence_mode

    def _compute_poi_lambda(
        self,
        *,
        day_alloc: pd.DataFrame,
        infectious_mask: np.ndarray,
        poi_cat_risk: np.ndarray,
        beta_eff: float,
    ) -> tuple[np.ndarray, float, int]:
        lam_poi = np.zeros(self.n_agents, dtype=float)
        if day_alloc.empty:
            return lam_poi, 0.0, 0

        infectious_idx = np.flatnonzero(infectious_mask)
        inf_set = set(int(x) for x in infectious_idx.tolist())
        alloc = day_alloc.copy()
        alloc["risk_cat"] = alloc["cat_idx"].map(
            {int(cat_id): float(poi_cat_risk[pos]) for pos, cat_id in enumerate(self.cat_ids)}
        ).fillna(0.0)

        presence = alloc.groupby("poi_id_num", as_index=False)["allocated_hours"].sum().rename(
            columns={"allocated_hours": "presence_hours"}
        )
        if inf_set:
            inf_alloc = alloc.loc[alloc["agent_idx"].isin(inf_set)]
            inf_presence = inf_alloc.groupby("poi_id_num", as_index=False)["allocated_hours"].sum().rename(
                columns={"allocated_hours": "inf_presence_hours"}
            )
        else:
            inf_presence = pd.DataFrame(columns=["poi_id_num", "inf_presence_hours"])

        poi_prev = presence.merge(inf_presence, on="poi_id_num", how="left")
        poi_prev["inf_presence_hours"] = pd.to_numeric(
            poi_prev["inf_presence_hours"], errors="coerce"
        ).fillna(0.0)
        poi_prev["p_poi"] = _safe_div(
            poi_prev["inf_presence_hours"].to_numpy(dtype=float),
            poi_prev["presence_hours"].to_numpy(dtype=float),
        )
        poi_prev_map = dict(zip(poi_prev["poi_id_num"].tolist(), poi_prev["p_poi"].tolist()))

        alloc["p_poi"] = alloc["poi_id_num"].map(poi_prev_map).fillna(0.0)
        alloc["hazard_term"] = (
            (alloc["allocated_hours"].to_numpy(dtype=float) / 24.0)
            * alloc["risk_cat"].to_numpy(dtype=float)
            * alloc["p_poi"].to_numpy(dtype=float)
        )
        agent_term = alloc.groupby("agent_idx", as_index=False)["hazard_term"].sum()
        lam_poi[agent_term["agent_idx"].to_numpy(dtype=int)] = (
            beta_eff * agent_term["hazard_term"].to_numpy(dtype=float)
        )
        return lam_poi, float(poi_prev["presence_hours"].sum()), int(poi_prev["poi_id_num"].nunique())

    def run(self, out_dir: Path, rng_seed: int = 1234):
        out_dir.mkdir(parents=True, exist_ok=True)
        f_seir = out_dir / "seir_timeseries.csv"
        dbg_path = out_dir / "debug_days.csv"
        if dbg_path.exists():
            dbg_path.unlink()

        self._seed_initial()
        print(f"[INFO] SEIR-POI: {self.n_days} days over {self.n_agents} agents")
        seir_rows = []

        for di, day in enumerate(tqdm(self.dates, desc="Days")):
            day_df, day_alloc, hours_mode, (genders, ages) = self._build_day_views(day)
            infectious_mask = self.state == 2
            p_mode, presence_mode = self._compute_mode_prevalence(hours_mode, infectious_mask)

            day_phase = int(pd.Series(day_df["epi_phase"]).mode(dropna=False).iloc[0]) if len(day_df) else -1
            beta_eff = self.beta * float(self.phase_beta_mult.get(str(day_phase), self.phase_beta_mult.get(day_phase, 1.0)))
            poi_cat_risk = self._cat_risk_today(day_phase)
            lam_poi_all, presence_poi_sum, unique_poi = self._compute_poi_lambda(
                day_alloc=day_alloc,
                infectious_mask=infectious_mask,
                poi_cat_risk=poi_cat_risk,
                beta_eff=float(beta_eff),
            )

            hm = hours_mode / 24.0
            lam_mode_all = beta_eff * (hm * (self.mode_mult[None, :] * p_mode[None, :])).sum(axis=1)

            g_arr = np.array([self.gender_mult.get(g, 1.0) for g in genders], dtype=float)
            a_arr = np.array(
                [self.age_mult[a] if (a >= 0 and a < len(self.age_mult)) else 1.0 for a in ages],
                dtype=float,
            )
            demo = g_arr * a_arr
            lam_mix = lam_poi_all + lam_mode_all
            lam = 1.0 - np.exp(-lam_mix * demo)
            lam = 1.0 - (1.0 - lam) * (1.0 - self.import_p)
            lam = np.clip(lam, 0.0, 1.0)

            s_mask = self.state == 0
            draws = self.rng.uniform(0.0, 1.0, size=self.n_agents)
            new_E_mask = (draws < lam) & s_mask
            prev_I = max(int((self.state == 2).sum()), 1)
            Rt_hat = (int(new_E_mask.sum()) / prev_I) * self.D
            den = float(lam_poi_all.sum() + lam_mode_all.sum())

            if (di == 0) or (di % 7 == 0):
                dbg = dict(
                    date=str(day),
                    phase=day_phase,
                    beta_eff=float(beta_eff),
                    infectious_count=int(infectious_mask.sum()),
                    mean_lambda=float(lam.mean()),
                    max_lambda=float(lam.max()),
                    sum_presence_poi=float(presence_poi_sum),
                    sum_presence_mode=float(presence_mode.sum()),
                    unique_poi=int(unique_poi),
                    new_E=int(new_E_mask.sum()),
                    mean_lam_poi=float(lam_poi_all.mean()),
                    mean_lam_mode=float(lam_mode_all.mean()),
                    frac_mode=float(lam_mode_all.sum() / den) if den > 1e-12 else 0.0,
                    Rt_hat=float(Rt_hat),
                )
                pd.DataFrame([dbg]).to_csv(
                    dbg_path,
                    mode=("a" if dbg_path.exists() else "w"),
                    header=(not dbg_path.exists()),
                    index=False,
                )

            new_R = (self.state == 2) & (self.timer <= 1)
            still_I = (self.state == 2) & (self.timer > 1)
            self.timer[still_I] -= 1
            self.state[new_R] = 3
            self.timer[new_R] = 0

            new_I = (self.state == 1) & (self.timer <= 1)
            still_E = (self.state == 1) & (self.timer > 1)
            self.timer[still_E] -= 1
            self.state[new_I] = 2
            self.timer[new_I] = self.D

            self.state[new_E_mask] = 1
            self.timer[new_E_mask] = self.L

            S = int((self.state == 0).sum())
            E = int((self.state == 1).sum())
            I = int((self.state == 2).sum())
            R = int((self.state == 3).sum())
            seir_rows.append(dict(date=str(day), S=S, E=E, I=I, R=R, new_E=int(new_E_mask.sum())))

        pd.DataFrame(seir_rows).to_csv(f_seir, index=False)
        print(f"[OK] Wrote: {f_seir}")


def main():
    ap = argparse.ArgumentParser(description="SEIR timeseries with POI-level mixing.")
    ap.add_argument("--daily_shares", required=True)
    ap.add_argument("--allocated_poi_hours", required=True)
    ap.add_argument("--hazard_params", required=True)
    ap.add_argument("--out_dir", default="out_seir_timeseries_poi")
    ap.add_argument("--feature_meta", default="")
    ap.add_argument("--latent_days", type=int, default=3)
    ap.add_argument("--infectious_days", type=int, default=7)
    ap.add_argument("--seed_pct", type=float, default=0.007)
    ap.add_argument("--seed_list", default="")
    ap.add_argument("--rng_seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--beta_scale", type=float, default=1.0)
    ap.add_argument("--phase_beta_json", default="")
    ap.add_argument("--importation_prob", type=float, default=None)
    args = ap.parse_args()

    ds = pd.read_csv(args.daily_shares)
    alloc = pd.read_csv(args.allocated_poi_hours)

    meta_path = Path(args.feature_meta) if args.feature_meta else (Path(args.daily_shares).parent / "feature_meta.json")
    meta = _load_feature_meta(meta_path)

    req = ["agent_id", "date", "epi_phase", "day_hours", "travel_frac"] + [f"mode_{i}" for i in meta["mode_ids"]]
    missing = [c for c in req if c not in ds.columns]
    if missing:
        raise SystemExit(f"Missing columns in --daily_shares: {missing}")

    with open(args.hazard_params, "r", encoding="utf-8") as f:
        hp = json.load(f)
    hp["beta_base"] = hp.get("beta_base", 0.06) * args.beta_scale

    phase_beta_mult = None
    if args.phase_beta_json and Path(args.phase_beta_json).exists():
        with open(args.phase_beta_json, "r", encoding="utf-8") as f:
            phase_beta_mult = json.load(f)
    elif "phase_beta_mult" in hp:
        phase_beta_mult = hp["phase_beta_mult"]

    seed_ids = None
    if args.seed_list:
        if Path(args.seed_list).exists():
            tmp = pd.read_csv(args.seed_list)
            if "agent_id" not in tmp.columns:
                raise SystemExit("--seed_list CSV must contain an 'agent_id' column.")
            seed_ids = tmp["agent_id"].astype(str).tolist()
        else:
            warnings.warn(f"--seed_list {args.seed_list} not found, falling back to --seed_pct")

    if args.importation_prob is not None:
        hp["importation_prob"] = args.importation_prob

    sim = SEIRTimeseriesPOI(
        daily_df=ds,
        alloc_df=alloc,
        feature_meta=meta,
        hazard_params=hp,
        latent_days=args.latent_days,
        infectious_days=args.infectious_days,
        seed_pct=args.seed_pct,
        seed_agent_ids=seed_ids,
        rng_seed=args.rng_seed,
        n_workers=args.workers,
    )
    sim.phase_beta_mult = {int(k): float(v) for k, v in (phase_beta_mult or {}).items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_cfg = {
        "daily_shares": str(Path(args.daily_shares).resolve()),
        "allocated_poi_hours": str(Path(args.allocated_poi_hours).resolve()),
        "hazard_params_source": str(Path(args.hazard_params).resolve()),
        "feature_meta": str(Path(args.feature_meta).resolve()) if args.feature_meta else "",
        "latent_days": int(args.latent_days),
        "infectious_days": int(args.infectious_days),
        "seed_pct": float(args.seed_pct),
        "rng_seed": int(args.rng_seed),
        "workers": int(args.workers),
        "beta_scale": float(args.beta_scale),
        "phase_beta_mult": {str(k): float(v) for k, v in sim.phase_beta_mult.items()},
        "hazard_params_effective": hp,
    }
    with open(out_dir / "effective_hazard_params.json", "w", encoding="utf-8") as f:
        json.dump(effective_cfg, f, ensure_ascii=False, indent=2)
    sim.run(out_dir=out_dir, rng_seed=args.rng_seed)


if __name__ == "__main__":
    main()
