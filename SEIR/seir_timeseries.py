from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count


def _ensure_cols(df: pd.DataFrame, cols: List[str], fill: float = 0.0):
    for c in cols:
        if c not in df.columns:
            df[c] = fill

def _coerce_gender(val) -> str:
    if pd.isna(val): return "U"
    if isinstance(val, (int, float)):
        return "M" if int(val) == 1 else ("F" if int(val) == 0 else "U")
    s = str(val).strip().upper()
    if s in ("M", "MALE", "1"): return "M"
    if s in ("F", "FEMALE", "0"): return "F"
    return "U"

def _safe_div(a, b, eps=1e-12): return a / (b + eps)
def _init_seed(seed: int): np.random.seed(seed)

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
        out = np.array([float(param.get(int(i), 1.0)) for i in ids], dtype=float)
        return out
    arr = np.array(param, dtype=float)
    if arr.shape[0] == len(ids):
        return arr
    raise ValueError(
        f"hazard_params['{name}'] length={arr.shape[0]} != number of IDs={len(ids)}. "
        f"Provide as dict keyed by IDs or reorder/resize to match IDs."
    )

class SEIRTimeseries:
    def __init__(self,
                 daily_df: pd.DataFrame,
                 feature_meta: Dict,
                 hazard_params: Dict,
                 latent_days: int,
                 infectious_days: int,
                 seed_pct: float,
                 seed_agent_ids: Optional[List[str]],
                 rng_seed: int,
                 n_workers: int):
        self.df = daily_df.copy()
        self.meta = feature_meta
        self.hp = hazard_params
        self.L = int(latent_days)
        self.D = int(infectious_days)
        self.seed_pct = float(seed_pct)
        self.seed_list = set(seed_agent_ids) if seed_agent_ids else None
        self.rng = np.random.default_rng(rng_seed)
        self.n_workers = max(1, n_workers)

        self.mixing_scale = "observed"        
        self.home_beta_alpha = 0.0            
        self.home_protection_scope = "population"
        self.phase_beta_mult = {}          
        self.phase_importation_mult = {}
        self.phase_schedule_by_date = {}
        self.disable_phase_beta = False
        self.disable_phase_importation = False
        self.disable_phase_cat = False
        self.date_importation_mult = {}
        self.date_external_seed_mult = {}
        self.crowding_eta = 0.0
        self.crowding_reference = {}
        self.crowding_min = 0.25
        self.crowding_max = 4.0
        self.phase_cat_mult = self.hp.get("phase_cat_mult", {})

        self.cat_ids: List[int] = self.meta["cat_ids"]
        self.mode_ids_all: List[int] = self.meta["mode_ids"]
        self.home_cat_idx: int = self.meta["home_cat_idx"]
        self.unknown_mode_idx: int = self.meta["unknown_mode_idx"]
        self.mode_ids_no_unk: List[int] = [m for m in self.mode_ids_all if m != self.unknown_mode_idx]

        self.CAT_COLS = [f"cat_{i}" for i in self.cat_ids]
        self.MODE_COLS_ALL = [f"mode_{i}" for i in self.mode_ids_all]
        self.MODE_COLS = [f"mode_{i}" for i in self.mode_ids_no_unk]

        _ensure_cols(self.df, ["day_hours", "travel_frac"], 0.0)
        _ensure_cols(self.df, ["epi_phase"], -1)
        _ensure_cols(self.df, self.CAT_COLS, 0.0)
        _ensure_cols(self.df, self.MODE_COLS_ALL, 0.0)

        self.df["agent_id"] = self.df["agent_id"].astype(str)
        self.df["date"] = pd.to_datetime(self.df["date"]).dt.date
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
        self.cat_mult = _vector_from_params_by_ids(self.hp.get("cat_mult", [1.0]*len(self.cat_ids)),
                                                   self.cat_ids, "cat_mult")
        self.mode_mult_all = _vector_from_params_by_ids(self.hp.get("mode_mult", [1.0]*len(self.mode_ids_all)),
                                                        self.mode_ids_all, "mode_mult")
        self.mode_mult = self.mode_mult_all[[self.mode_ids_all.index(m) for m in self.mode_ids_no_unk]]
        self.gender_mult = self.hp.get("gender_mult", {"M":1.0,"F":1.0,"U":1.0})
        self.age_mult = np.array(self.hp.get("age_mult", [1.0]*5), dtype=float)
        default_c_cat = np.ones(len(self.cat_ids), dtype=float)
        c_cat_param = self.hp.get("contact_rate_cat", default_c_cat.tolist())
        self.c_cat = _vector_from_params_by_ids(c_cat_param, self.cat_ids, "contact_rate_cat")
        self.import_p = float(self.hp.get("importation_prob", 0.0))
        self.external_seed_prob = float(self.hp.get("external_seed_prob", 0.0))

        # states
         # 0=S,1=E,2=I,3=R
        self.state = np.zeros(self.n_agents, dtype=np.int8)  
        self.timer = np.zeros(self.n_agents, dtype=np.int16)

    def set_crowding_reference(self, reference_df: pd.DataFrame):
        ref = reference_df.copy()
        _ensure_cols(ref, ["day_hours", "travel_frac"], 0.0)
        _ensure_cols(ref, self.CAT_COLS, 0.0)
        _ensure_cols(ref, self.MODE_COLS_ALL, 0.0)
        ref["date"] = pd.to_datetime(ref["date"]).dt.date

        mode_idx_map = [self.mode_ids_all.index(m) for m in self.mode_ids_no_unk]
        self.crowding_reference = {}
        for day, day_df in ref.groupby("date", sort=True):
            cat_shares = day_df[self.CAT_COLS].to_numpy(dtype=float)
            modes_all = day_df[self.MODE_COLS_ALL].to_numpy(dtype=float)
            modes = modes_all[:, mode_idx_map]
            travel_frac = day_df["travel_frac"].to_numpy(dtype=float)
            if self.mixing_scale == "24h":
                ref_cat = (24.0 * cat_shares).sum(axis=0)
                ref_mode = (24.0 * (travel_frac[:, None] * modes)).sum(axis=0)
            else:
                day_hours = day_df["day_hours"].to_numpy(dtype=float)
                ref_cat = (day_hours[:, None] * cat_shares).sum(axis=0)
                ref_mode = (day_hours[:, None] * travel_frac[:, None] * modes).sum(axis=0)
            self.crowding_reference[day] = (ref_cat, ref_mode)

    def _crowding_multipliers(self, day, presence_cat, presence_mode):
        if self.crowding_eta <= 0.0 or not self.crowding_reference:
            return np.ones_like(presence_cat), np.ones_like(presence_mode)
        ref = self.crowding_reference.get(day)
        if ref is None:
            return np.ones_like(presence_cat), np.ones_like(presence_mode)
        ref_cat, ref_mode = ref
        eps = 1e-6
        cat_ratio = np.clip((presence_cat + eps) / (ref_cat + eps), self.crowding_min, self.crowding_max)
        mode_ratio = np.clip((presence_mode + eps) / (ref_mode + eps), self.crowding_min, self.crowding_max)
        return cat_ratio ** self.crowding_eta, mode_ratio ** self.crowding_eta

    def _resolve_day_phase(self, day, day_df: pd.DataFrame) -> Tuple[int, str]:
        if day in self.phase_schedule_by_date:
            return int(self.phase_schedule_by_date[day]), "override"
        if len(day_df):
            return int(pd.Series(day_df["epi_phase"]).mode(dropna=False).iloc[0]), "daily_shares"
        return -1, "missing"

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
        day_hours   = day_df["day_hours"].to_numpy(dtype=float)
        cats        = day_df[[f"cat_{k}" for k in self.cat_ids]].to_numpy(dtype=float)
        hours_cat_present = day_hours[:, None] * cats

        travel_frac = day_df["travel_frac"].to_numpy(dtype=float)
        modes_all   = day_df[[f"mode_{m}" for m in self.mode_ids_all]].to_numpy(dtype=float)
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
        nC = len(self.cat_ids); nM = len(self.mode_ids_no_unk)
        full_hours_cat = np.zeros((self.n_agents, nC), dtype=float)
        full_hours_mode = np.zeros((self.n_agents, nM), dtype=float)
        full_gender = np.array(["U"]*self.n_agents, dtype=object)
        full_age = np.full(self.n_agents, -1, dtype=int)

        full_hours_cat[idx_present, :] = hours_cat_present
        full_hours_mode[idx_present, :] = hours_mode_present
        full_gender[idx_present] = genders_present
        full_age[idx_present] = age_idx_present
        return day_df, full_hours_cat, full_hours_mode, (full_gender, full_age)

    def _compute_prevalence(self, day, hours_cat, hours_mode, infectious_mask, present_ids=None):
        if self.mixing_scale == "24h":
            day_mask = (self.df["date"] == day)
            day_sub = self.df.loc[day_mask, ["agent_id","travel_frac"] + [f"cat_{k}" for k in self.cat_ids] + [f"mode_{m}" for m in self.mode_ids_all]].copy()
            day_sub["agent_id"] = day_sub["agent_id"].astype(str)
            day_sub = day_sub.set_index("agent_id").reindex(present_ids)  # align to present order

            shares_cat = day_sub[[f"cat_{k}" for k in self.cat_ids]].to_numpy(dtype=float)

            modes_all = day_sub[[f"mode_{m}" for m in self.mode_ids_all]].to_numpy(dtype=float)
            mode_idx_map = [self.mode_ids_all.index(m) for m in self.mode_ids_no_unk]
            modes = modes_all[:, mode_idx_map]

            travel_frac = day_sub["travel_frac"].to_numpy(dtype=float)

            presence_cat  = (24.0 * shares_cat).sum(axis=0)
            presence_mode = (24.0 * (travel_frac[:, None] * modes)).sum(axis=0)

            present_idx = [self.agent_idx[a] for a in present_ids]
            inf_present_mask = infectious_mask[present_idx]

            inf_presence_cat  = (24.0 * shares_cat[inf_present_mask]).sum(axis=0)
            inf_presence_mode = (24.0 * (travel_frac[inf_present_mask][:, None] * modes[inf_present_mask])).sum(axis=0)
        else:
            presence_cat  = hours_cat.sum(axis=0)
            inf_presence_cat = hours_cat[infectious_mask].sum(axis=0)
            presence_mode = hours_mode.sum(axis=0)
            inf_presence_mode = hours_mode[infectious_mask].sum(axis=0)


        p_cat = _safe_div(inf_presence_cat, presence_cat)
        p_mode = _safe_div(inf_presence_mode, presence_mode)
        return p_cat, p_mode, presence_cat, presence_mode

    def _hazard_chunk(self, args):
        (agent_slice, hours_cat, hours_mode, genders, ages, p_cat, p_mode,
         beta, home_factor, cat_mult, mode_mult, crowding_cat, crowding_mode,
         gender_mult_map, age_mult, c_cat, import_p) = args
        start, end = agent_slice
        hc = hours_cat[start:end, :] / 24.0
        hm = hours_mode[start:end, :] / 24.0
        hf = home_factor[start:end]

        g_arr = np.array([gender_mult_map.get(g, 1.0) for g in genders[start:end]], dtype=float)
        a_arr = np.array([age_mult[a] if (a >= 0 and a < len(age_mult)) else 1.0 for a in ages[start:end]], dtype=float)
        demo = g_arr * a_arr

        lam_cat = (beta * hf * (hc * (
            cat_mult[None, :] * c_cat[None, :] * crowding_cat[None, :] * p_cat[None, :]
        )).sum(axis=1))
        lam_mode = (beta * hf * (hm * (
            mode_mult[None, :] * crowding_mode[None, :] * p_mode[None, :]
        )).sum(axis=1))
        lam = 1.0 - np.exp(-(lam_cat + lam_mode) * demo)
        lam = 1.0 - (1.0 - lam) * (1.0 - import_p)
        lam = np.clip(lam, 0.0, 1.0)
        return (start, end, lam, lam_cat, lam_mode)

    def run(self, out_dir: Path, rng_seed: int = 123):
        out_dir.mkdir(parents=True, exist_ok=True)
        f_seir = out_dir / "seir_timeseries.csv"
        dbg_path = out_dir / "debug_days.csv"
        if dbg_path.exists(): dbg_path.unlink()

        self._seed_initial()
        print(f"[INFO] SEIR-only: {self.n_days} days over {self.n_agents} agents (workers={self.n_workers})")
        seir_rows = []

        for di, day in enumerate(tqdm(self.dates, desc="Days")):
            day_df, hours_cat, hours_mode, (genders, ages) = self._build_day_views(day)
            infectious_mask = (self.state == 2)

            present_ids = day_df["agent_id"].astype(str).tolist()
            p_cat, p_mode, presence_cat, presence_mode = self._compute_prevalence(day, hours_cat, hours_mode,
                                                                                   infectious_mask, present_ids)
            crowding_cat, crowding_mode = self._crowding_multipliers(day, presence_cat, presence_mode)

            day_phase, phase_source = self._resolve_day_phase(day, day_df)
            beta_eff = self.beta
            if not self.disable_phase_beta:
                beta_eff *= float(self.phase_beta_mult.get(day_phase, 1.0))
            if self.date_importation_mult:
                import_mult_today = float(self.date_importation_mult.get(day, 1.0))
            else:
                import_mult_today = (
                    1.0 if self.disable_phase_importation
                    else float(self.phase_importation_mult.get(day_phase, 1.0))
                )
            import_p_eff = self.import_p * import_mult_today
            if self.date_external_seed_mult:
                external_seed_mult_today = float(self.date_external_seed_mult.get(day, 1.0))
            else:
                external_seed_mult_today = 1.0
            external_seed_p_eff = self.external_seed_prob * external_seed_mult_today

            cat_mult_today = self.cat_mult.copy()
            phase_key = str(day_phase)
            if (not self.disable_phase_cat) and phase_key in self.phase_cat_mult:
                overrides = self.phase_cat_mult[phase_key]
                for k_str, v in overrides.items():
                    k = int(k_str)
                    if k in self.cat_ids:
                        pos = self.cat_ids.index(k)
                        cat_mult_today[pos] = float(v)

            day_mask = (self.df["date"] == day)
            home_col = f"cat_{self.home_cat_idx}"
            home_share_mean = float(self.df.loc[day_mask, home_col].mean()) if home_col in self.df.columns else 0.0
            home_factor = np.ones(self.n_agents, dtype=float)
            if self.home_beta_alpha > 0.0:
                if self.home_protection_scope == "agent":
                    if home_col in day_df.columns:
                        home_share_agent = (
                            day_df.set_index("agent_id")[home_col]
                            .reindex(self.agents)
                            .fillna(home_share_mean)
                            .to_numpy(dtype=float)
                        )
                    else:
                        home_share_agent = np.full(self.n_agents, home_share_mean, dtype=float)
                    home_factor = np.maximum(0.0, 1.0 - self.home_beta_alpha * home_share_agent)
                else:
                    home_factor[:] = max(0.0, 1.0 - self.home_beta_alpha * home_share_mean)
            chunks = []
            step = max(1, self.n_agents // (self.n_workers * 8)) or 1
            for start in range(0, self.n_agents, step):
                end = min(self.n_agents, start + step)
                chunks.append(((start, end), hours_cat, hours_mode, genders, ages,
                               p_cat, p_mode, float(beta_eff), home_factor, cat_mult_today, self.mode_mult,
                               crowding_cat, crowding_mode, self.gender_mult, self.age_mult,
                               self.c_cat, float(import_p_eff)))
            if self.n_workers > 1:
                with Pool(processes=self.n_workers, initializer=_init_seed, initargs=(rng_seed,)) as pool:
                    results = list(pool.imap_unordered(self._hazard_chunk, chunks))
            else:
                results = [self._hazard_chunk(ch) for ch in chunks]

            lam = np.zeros(self.n_agents, dtype=float)
            lam_cat_all = np.zeros(self.n_agents)
            lam_mode_all = np.zeros(self.n_agents)
            for start, end, l, lc, lm in results:
                lam[start:end] = l
                lam_cat_all[start:end] = lc
                lam_mode_all[start:end] = lm

            s_mask = (self.state == 0)
            draws = self.rng.uniform(0.0, 1.0, size=self.n_agents)
            new_E_internal_mask = (draws < lam) & s_mask
            remaining_s_mask = s_mask & (~new_E_internal_mask)
            new_E_external_mask = np.zeros(self.n_agents, dtype=bool)
            if external_seed_p_eff > 0.0:
                remaining_idx = np.flatnonzero(remaining_s_mask)
                n_remaining = int(remaining_idx.size)
                if n_remaining > 0:
                    seed_p = min(max(float(external_seed_p_eff), 0.0), 1.0)
                    n_external = int(self.rng.binomial(n_remaining, seed_p))
                    if n_external > 0:
                        picked = self.rng.choice(remaining_idx, size=n_external, replace=False)
                        new_E_external_mask[picked] = True
            new_E_mask = new_E_internal_mask | new_E_external_mask
            prev_I = max(int((self.state == 2).sum()), 1)
            Rt_hat = (int(new_E_internal_mask.sum()) / prev_I) * self.D
            den = float(lam_cat_all.sum() + lam_mode_all.sum())
            mean_lam_mix = float((1.0 - np.exp(-(lam_cat_all + lam_mode_all))).mean())

            if (di == 0) or (di % 7 == 0):
                dbg = dict(
                    date=str(day),
                    phase=day_phase,
                    phase_source=phase_source,
                    phase_beta_enabled=(not self.disable_phase_beta),
                    phase_importation_enabled=(not self.disable_phase_importation),
                    phase_cat_enabled=(not self.disable_phase_cat),
                    beta_eff=float(beta_eff),
                    mean_beta_eff_after_home=float(beta_eff * home_factor.mean()),
                    home_share_mean=home_share_mean,
                    home_protection_scope=self.home_protection_scope,
                    mean_home_factor=float(home_factor.mean()),
                    min_home_factor=float(home_factor.min()),
                    max_home_factor=float(home_factor.max()),
                    infectious_count=int(infectious_mask.sum()),
                    mean_lambda=float(lam.mean()),
                    max_lambda=float(lam.max()),
                    sum_presence_cat=float(presence_cat.sum()),
                    sum_presence_mode=float(presence_mode.sum()),
                    new_E=int(new_E_mask.sum()),
                    new_E_internal=int(new_E_internal_mask.sum()),
                    new_E_external=int(new_E_external_mask.sum()),
                    mean_lam_cat=float(lam_cat_all.mean()),
                    mean_lam_mode=float(lam_mode_all.mean()),
                    frac_mode=float(lam_mode_all.sum() / den) if den > 1e-12 else 0.0,
                    Rt_hat=float(Rt_hat),
                    mean_lambda_mix=mean_lam_mix,
                    import_p_eff=float(import_p_eff),
                    external_seed_p_eff=float(external_seed_p_eff),
                    crowding_eta=float(self.crowding_eta),
                    mean_crowding_cat=float(crowding_cat.mean()),
                    mean_crowding_mode=float(crowding_mode.mean()),
                )
                pd.DataFrame([dbg]).to_csv(dbg_path, mode=("a" if dbg_path.exists() else "w"),
                                           header=(not dbg_path.exists()), index=False)

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
            seir_rows.append(
                dict(
                    date=str(day),
                    phase_used=int(day_phase),
                    phase_source=phase_source,
                    S=S,
                    E=E,
                    I=I,
                    R=R,
                    new_E=int(new_E_mask.sum()),
                    new_E_internal=int(new_E_internal_mask.sum()),
                    new_E_external=int(new_E_external_mask.sum()),
                )
            )

        pd.DataFrame(seir_rows).to_csv(f_seir, index=False)
        print(f"[OK] Wrote: {f_seir}")

# ------------------------------ CLI ------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SEIR timeseries only — dynamic IDs.")
    ap.add_argument("--daily_shares", required=True)
    ap.add_argument("--hazard_params", required=True)
    ap.add_argument("--out_dir", default="out_seir_timeseries")
    ap.add_argument("--feature_meta", default="")
    ap.add_argument("--latent_days", type=int, default=3)
    ap.add_argument("--infectious_days", type=int, default=7)
    ap.add_argument("--seed_pct", type=float, default=0.007)
    ap.add_argument("--seed_list", default="")
    ap.add_argument("--rng_seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count()//2))
    ap.add_argument("--beta_scale", type=float, default=1.0)
    ap.add_argument("--mixing_scale", choices=["observed","24h"], default="observed")
    ap.add_argument("--home_beta_alpha", type=float, default=0.0)
    ap.add_argument("--home_protection_scope", choices=["population", "agent"], default="population")
    ap.add_argument("--phase_beta_json", default="")
    ap.add_argument("--phase_importation_json", default="")
    ap.add_argument("--phase_schedule_csv", default="")
    ap.add_argument("--phase_schedule_col", default="phase_id")
    ap.add_argument("--date_importation_schedule_csv", default="")
    ap.add_argument("--date_external_seed_schedule_csv", default="")
    ap.add_argument("--importation_prob", type=float, default=None)
    ap.add_argument("--disable_phase_beta", action="store_true")
    ap.add_argument("--disable_phase_importation", action="store_true")
    ap.add_argument("--disable_phase_cat", action="store_true")
    ap.add_argument("--phase_invariant", action="store_true",
                    help="Disable all phase-specific SEIR multipliers so transmission is driven only by behaviour/crowding.")
    ap.add_argument("--crowding_eta", type=float, default=0.0)
    ap.add_argument("--crowding_reference_daily_shares", default="")
    args = ap.parse_args()

    ds = pd.read_csv(args.daily_shares)

    meta_path = Path(args.feature_meta) if args.feature_meta else (Path(args.daily_shares).parent / "feature_meta.json")
    meta = _load_feature_meta(meta_path)

    req = ["agent_id","date","epi_phase","day_hours","travel_frac"] \
          + [f"cat_{i}" for i in meta["cat_ids"]] \
          + [f"mode_{i}" for i in meta["mode_ids"]]
    missing = [c for c in req if c not in ds.columns]
    if missing:
        raise SystemExit(f"Missing columns in --daily_shares: {missing}")

    with open(args.hazard_params, "r") as f:
        hp = json.load(f)
    hp["beta_base"] = hp.get("beta_base", 0.06) * args.beta_scale

    phase_beta_mult = None
    if args.phase_beta_json and Path(args.phase_beta_json).exists():
        with open(args.phase_beta_json, "r") as f:
            phase_beta_mult = json.load(f)
    elif "phase_beta_mult" in hp:
        phase_beta_mult = hp["phase_beta_mult"]

    phase_importation_mult = None
    if args.phase_importation_json and Path(args.phase_importation_json).exists():
        with open(args.phase_importation_json, "r") as f:
            phase_importation_mult = json.load(f)
    elif "phase_importation_mult" in hp:
        phase_importation_mult = hp["phase_importation_mult"]

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

    sim = SEIRTimeseries(
        daily_df=ds,
        feature_meta=meta,
        hazard_params=hp,
        latent_days=args.latent_days,
        infectious_days=args.infectious_days,
        seed_pct=args.seed_pct,
        seed_agent_ids=seed_ids,
        rng_seed=args.rng_seed,
        n_workers=args.workers,
    )
    sim.mixing_scale = args.mixing_scale
    sim.home_beta_alpha = float(args.home_beta_alpha)
    sim.home_protection_scope = args.home_protection_scope
    sim.phase_beta_mult = {int(k): float(v) for k, v in (phase_beta_mult or {}).items()}
    sim.phase_importation_mult = {int(k): float(v) for k, v in (phase_importation_mult or {}).items()}
    sim.disable_phase_beta = bool(args.disable_phase_beta or args.phase_invariant)
    sim.disable_phase_importation = bool(args.disable_phase_importation or args.phase_invariant)
    sim.disable_phase_cat = bool(args.disable_phase_cat or args.phase_invariant)
    sim.crowding_eta = float(args.crowding_eta)
    phase_schedule_path = Path(args.phase_schedule_csv) if args.phase_schedule_csv else None
    if phase_schedule_path:
        if not phase_schedule_path.exists():
            raise FileNotFoundError(f"Phase schedule CSV not found: {phase_schedule_path}")
        phase_sched_df = pd.read_csv(phase_schedule_path)
        if "date" not in phase_sched_df.columns:
            raise ValueError(f"{phase_schedule_path} must contain a 'date' column.")
        phase_col = None
        for candidate in [args.phase_schedule_col, "phase_id", "epi_phase"]:
            if candidate and candidate in phase_sched_df.columns:
                phase_col = candidate
                break
        if phase_col is None:
            raise ValueError(
                f"{phase_schedule_path} must contain one of: {args.phase_schedule_col}, phase_id, epi_phase."
            )
        phase_sched_df["date"] = pd.to_datetime(phase_sched_df["date"]).dt.date
        phase_sched_df[phase_col] = pd.to_numeric(phase_sched_df[phase_col], errors="coerce")
        phase_sched_df = phase_sched_df.dropna(subset=[phase_col])
        sim.phase_schedule_by_date = {
            day: int(pd.Series(vals).mode(dropna=False).iloc[0])
            for day, vals in phase_sched_df.groupby("date")[phase_col]
        }
    crowding_reference_path = Path(args.crowding_reference_daily_shares) if args.crowding_reference_daily_shares else None
    if sim.crowding_eta > 0.0:
        if crowding_reference_path is None:
            raise ValueError("--crowding_reference_daily_shares is required when --crowding_eta > 0")
        if not crowding_reference_path.exists():
            raise FileNotFoundError(f"Crowding reference daily_shares not found: {crowding_reference_path}")
        sim.set_crowding_reference(pd.read_csv(crowding_reference_path))
    if args.date_importation_schedule_csv:
        sched_path = Path(args.date_importation_schedule_csv)
        if not sched_path.exists():
            raise FileNotFoundError(f"Date importation schedule not found: {sched_path}")
        sched_df = pd.read_csv(sched_path)
        if "date" not in sched_df.columns or "importation_mult" not in sched_df.columns:
            raise ValueError(
                f"{sched_path} must contain 'date' and 'importation_mult' columns."
            )
        sched_df["date"] = pd.to_datetime(sched_df["date"]).dt.date
        sim.date_importation_mult = {
            day: float(mult)
            for day, mult in zip(sched_df["date"], sched_df["importation_mult"])
        }
    if args.date_external_seed_schedule_csv:
        sched_path = Path(args.date_external_seed_schedule_csv)
        if not sched_path.exists():
            raise FileNotFoundError(f"Date external seed schedule not found: {sched_path}")
        sched_df = pd.read_csv(sched_path)
        if "date" not in sched_df.columns or "importation_mult" not in sched_df.columns:
            raise ValueError(
                f"{sched_path} must contain 'date' and 'importation_mult' columns."
            )
        sched_df["date"] = pd.to_datetime(sched_df["date"]).dt.date
        sim.date_external_seed_mult = {
            day: float(mult)
            for day, mult in zip(sched_df["date"], sched_df["importation_mult"])
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_cfg = {
        "daily_shares": str(Path(args.daily_shares).resolve()),
        "hazard_params_source": str(Path(args.hazard_params).resolve()),
        "feature_meta": str(Path(args.feature_meta).resolve()) if args.feature_meta else "",
        "latent_days": int(args.latent_days),
        "infectious_days": int(args.infectious_days),
        "seed_pct": float(args.seed_pct),
        "rng_seed": int(args.rng_seed),
        "workers": int(args.workers),
        "beta_scale": float(args.beta_scale),
        "mixing_scale": args.mixing_scale,
        "home_beta_alpha": float(args.home_beta_alpha),
        "home_protection_scope": args.home_protection_scope,
        "phase_beta_mult": {str(k): float(v) for k, v in sim.phase_beta_mult.items()},
        "phase_importation_mult": {str(k): float(v) for k, v in sim.phase_importation_mult.items()},
        "phase_schedule_csv": str(phase_schedule_path.resolve()) if phase_schedule_path else "",
        "phase_schedule_col": args.phase_schedule_col,
        "disable_phase_beta": bool(sim.disable_phase_beta),
        "disable_phase_importation": bool(sim.disable_phase_importation),
        "disable_phase_cat": bool(sim.disable_phase_cat),
        "phase_invariant": bool(args.phase_invariant),
        "date_importation_schedule_csv": str(Path(args.date_importation_schedule_csv).resolve()) if args.date_importation_schedule_csv else "",
        "date_external_seed_schedule_csv": str(Path(args.date_external_seed_schedule_csv).resolve()) if args.date_external_seed_schedule_csv else "",
        "crowding_eta": float(sim.crowding_eta),
        "crowding_reference_daily_shares": str(crowding_reference_path.resolve()) if crowding_reference_path else "",
        "crowding_min": float(sim.crowding_min),
        "crowding_max": float(sim.crowding_max),
        "hazard_params_effective": hp,
    }
    with open(out_dir / "effective_hazard_params.json", "w", encoding="utf-8") as f:
        json.dump(effective_cfg, f, ensure_ascii=False, indent=2)
    sim.run(out_dir=out_dir, rng_seed=args.rng_seed)

if __name__ == "__main__":
    main()
