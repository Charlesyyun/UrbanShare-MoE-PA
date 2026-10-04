from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

EPS = 1e-8

def _load_feature_meta(meta_path: Path) -> Dict:
    meta = json.loads(meta_path.read_text())
    for k in ["cat_ids", "mode_ids", "home_cat_idx", "unknown_mode_idx"]:
        if k not in meta:
            raise ValueError(f"feature_meta.json missing '{k}'")
    meta["cat_ids"] = [int(x) for x in meta["cat_ids"]]
    meta["mode_ids"] = [int(x) for x in meta["mode_ids"]]
    meta["home_cat_idx"] = int(meta["home_cat_idx"])
    meta["unknown_mode_idx"] = int(meta["unknown_mode_idx"])
    return meta

@dataclass
class Columns:
    cat_cols: List[str]
    mode_cols_all: List[str]
    mode_cols_known: List[str]  # excludes unknown travel mode
    ctx_cols: List[str]

class DailyShareDataset(Dataset):
    """
    Builds per-day samples with inputs (agent, phase, weekday, days_since, demographics)
    and targets: category shares (K), mode shares over known modes (M'), and travel_frac.
    """
    def __init__(self, daily_csv: Path, feature_meta_path: Optional[Path] = None):
        self.daily = pd.read_csv(daily_csv)
        if feature_meta_path and feature_meta_path.exists():
            self.meta = _load_feature_meta(feature_meta_path)
            cat_ids = sorted(self.meta["cat_ids"])
            mode_ids = sorted(self.meta["mode_ids"])
            self.unknown_id = self.meta["unknown_mode_idx"]
        else:
            # derive IDs from columns if meta missing
            def ids_from_cols(cols, pref):
                out=set()
                for c in cols:
                    if c.startswith(pref):
                        try: out.add(int(c.split("_")[1]))
                        except: pass
                return sorted(out)
            cat_ids = ids_from_cols(self.daily.columns.tolist(), "cat")
            mode_ids = ids_from_cols(self.daily.columns.tolist(), "mode")
            # best-effort: unknown assumed = max id if present but we will drop if column not used
            self.unknown_id = None

        self.daily["date"] = pd.to_datetime(self.daily["date"]).dt.date
        self.agents = sorted(self.daily["agent_id"].astype(str).unique().tolist())
        self.agent2idx = {a:i for i,a in enumerate(self.agents)}

        # columns
        self.cat_cols = [f"cat_{i}" for i in cat_ids]
        self.mode_cols_all = [f"mode_{i}" for i in mode_ids]
        # exclude unknown if present
        self.mode_cols_known = [c for c in self.mode_cols_all if not c.endswith(f"_{self.unknown_id}")]
        # train on ALL modes (including unknown)
        self.mode_cols_train = list(self.mode_cols_all)
        # base context (distance, temporal)
        ctx_cols = [
            "epi_phase","days_since_phase","days_since_npi","gender",
            "wk_0","wk_1","wk_2","wk_3","wk_4","wk_5","wk_6",
            "age_0","age_1","age_2","age_3","age_4",
            "day_hours",
            "total_trip_km","mean_trip_km","max_trip_km","n_trips",
            "mean_nonhome_dist_km","max_nonhome_dist_km",
            "log_total_trip_km","log_mean_trip_km","log_max_trip_km",
            "log_mean_nonhome_dist_km","log_max_nonhome_dist_km",
            "prev_travel_frac","prev7_travel_frac_mean",
            "prev_home_share","prev7_home_share_mean",
        ]

        dyn_lag_cols = [c for c in self.daily.columns
                        if c.startswith("prev_cat_") or c.startswith("prev7_cat_")]
        self.ctx_cols = [c for c in ctx_cols if c in self.daily.columns] + sorted(dyn_lag_cols)


        # ensure required columns exist
        req = ["agent_id","date","epi_phase","day_hours","travel_frac"] + self.cat_cols + self.mode_cols_all
        missing = [c for c in req if c not in self.daily.columns]
        if missing:
            raise ValueError(f"Missing columns in daily_shares: {missing}")

        cats = self.daily[self.cat_cols].to_numpy(dtype=float)
        cats = np.clip(cats, 0.0, 1.0)
        K = cats.shape[1]
        cat_sum = cats.sum(axis=1, keepdims=True)
        
        cats = np.divide(cats, np.maximum(cat_sum, EPS), out=np.zeros_like(cats), where=cat_sum>EPS)
        self.daily[self.cat_cols] = cats


        modes_train = self.daily[self.mode_cols_train].to_numpy(dtype=float)
        modes_train = np.clip(modes_train, 0.0, 1.0)
        M = modes_train.shape[1]
        mt_sum = modes_train.sum(axis=1, keepdims=True)
        self.mode_loss_mask = (mt_sum.squeeze(-1) > EPS).astype(np.float32)
        modes_train = np.divide(modes_train, np.maximum(mt_sum, EPS), out=np.zeros_like(modes_train), where=mt_sum>EPS)
        self.daily[self.mode_cols_train] = modes_train

        for c in self.ctx_cols:
            if c == "gender":
                g = pd.to_numeric(self.daily[c], errors="coerce").fillna(0.5).astype(float)
                self.daily[c] = g
            else:
                self.daily[c] = pd.to_numeric(self.daily[c], errors="coerce").fillna(0.0).astype(float)

        self.index = self.daily.index.to_numpy()
        self.n_cat = len(self.cat_cols)
        self.n_mode = len(self.mode_cols_train)

        # Build phase order for prev_phase lookup (used in training augmentation)
        phase_order_df = (
            self.daily[["epi_phase","date"]].drop_duplicates()
            .groupby("epi_phase", as_index=False)["date"].min()
            .sort_values("date")
        )
        ordered_phases = [int(x) for x in phase_order_df["epi_phase"].tolist()]
        self._prev_phase_map = {ph: (ordered_phases[i-1] if i > 0 else ph)
                                for i, ph in enumerate(ordered_phases)}
        self._has_prev_phase_map = {ph: (1.0 if i > 0 else 0.0)
                                    for i, ph in enumerate(ordered_phases)}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i = self.index[idx]
        row = self.daily.loc[i]

        agent_id = str(row["agent_id"])
        agent_idx = self.agent2idx.get(agent_id, 0)

        epi_phase = int(row["epi_phase"]) if not pd.isna(row["epi_phase"]) else -1
        # Build context vector (fixed ordering)
        ctx_vals = [row.get(c, 0.0) for c in self.ctx_cols]

        x = np.array(ctx_vals, dtype=np.float32)
        a = np.int64(agent_idx)
        p = np.int64(max(epi_phase, 0))

        # Targets
        y_cat = row[self.cat_cols].to_numpy(dtype=np.float32)
        y_mode = row[self.mode_cols_train].to_numpy(dtype=np.float32)
        travel_frac = float(row.get("travel_frac", 0.0))
        day_hours = float(row.get("day_hours", 24.0))
        mode_mask = float(self.mode_loss_mask[i])

        return {
            "agent_idx": torch.tensor(a, dtype=torch.long),
            "phase_id": torch.tensor(p, dtype=torch.long),
            "x_ctx": torch.tensor(x, dtype=torch.float32),
            "y_cat": torch.tensor(y_cat, dtype=torch.float32),
            "y_mode": torch.tensor(y_mode, dtype=torch.float32),
            "travel_frac": torch.tensor([travel_frac], dtype=torch.float32),
            "day_hours": torch.tensor([day_hours], dtype=torch.float32),
            "mode_mask": torch.tensor([mode_mask], dtype=torch.float32),
            "prev_phase_id": torch.tensor(self._prev_phase_map.get(int(p), int(p)), dtype=torch.long),
            "has_prev_phase": torch.tensor(self._has_prev_phase_map.get(int(p), 0.0), dtype=torch.float32),
            "meta": {
                "agent_id": agent_id,
                "date": str(row["date"]),
                "epi_phase": int(row["epi_phase"]) if not pd.isna(row["epi_phase"]) else -1
            }
        }

    def schema(self) -> Columns:
        return Columns(
            cat_cols=self.cat_cols,
            mode_cols_all=self.mode_cols_all,
            mode_cols_known=self.mode_cols_known,
            ctx_cols=self.ctx_cols
        )

    def agent_index_df(self) -> pd.DataFrame:
        return pd.DataFrame({"agent_id": self.agents, "agent_idx": list(range(len(self.agents)))})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily_shares", required=True)
    ap.add_argument("--feature_meta", default="")
    args = ap.parse_args()
    meta_path = Path(args.feature_meta) if args.feature_meta else Path(args.daily_shares).parent / "feature_meta.json"
    ds = DailyShareDataset(Path(args.daily_shares), meta_path)
    print("Rows:", len(ds), "Agents:", len(ds.agents), "Cats:", ds.n_cat, "Modes:", ds.n_mode)
    print("Cat cols:", ds.schema().cat_cols)
    print("Mode (known) cols:", ds.schema().mode_cols_known)
    print("Ctx cols:", ds.schema().ctx_cols)


