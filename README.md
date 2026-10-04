# UrbanShare-MoE-PA

This repository contains the paper-aligned UrbanShare-MoE-PA code and the activity-aware SEIR analysis used in the accompanying paper.

The public release is intentionally code-first. It contains source code, policy calendars, calibrated SEIR parameters, aggregated paper tables, and figure artifacts. It does **not** contain individual-level mobility trajectories, raw activity chains, checkpoints, agent-level predictions, or private training outputs.

## What is reproducible from this repository

- Inspect the activity-aware SEIR implementation in `SEIR/`.
- Rebuild the published ten-seed Table 9 summary and Figure 15 from the retained SEIR time series.
- Inspect the four policy calendars, calibrated parameters, and model contracts.
- Run syntax and release-integrity checks with `python tools/validate_release.py`.

The ten-seed summary includes the four policy calendars plus the generated original-policy reference. `Observed reference` is an external calibration/reference quantity and is not treated as another simulated calendar. See `docs/PAPER_ALIGNMENT.md` for the exact mapping between paper terminology and repository artifacts.

## What is not reproducible without private inputs

End-to-end behavioral training and the 50 stochastic policy-calendar rollouts require restricted agent-day daily-share inputs and model checkpoints. Those inputs are deliberately excluded from this public repository. The scripts document the expected interfaces and fail explicitly when the private manifest is not supplied; no private path is required for the public validation checks.

## Quick start

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python tools/validate_release.py
```

The released aggregate table, figure, and 50 compact SEIR time series are stored in the repository. Regenerating the trajectories from behavioral daily shares still requires a local private-input manifest; see `SEIR/results/policy_calendar_eta1_reproduction/random_seed_table9/README.md`.

## Repository map

```text
.
├── daily_share_model.py              # UrbanShare-MoE model and training interface
├── preference_dataset.py             # daily-share dataset contract
├── preference_alignment_finetune.py  # preference-alignment fine-tuning
├── ├── ├── SEIR/                             # activity-aware SEIR code and paper artifacts
├── dataset/MetaData/                 # public model configuration
├── early|late|short|long/            # policy calendars and release placeholders
├── docs/                              # architecture, workflow, and paper alignment
└── tools/                             # public release validation utilities
```

The repository intentionally omits unrelated baseline and ablation trees, legacy single-seed figures, and optional POI branches. The retained files are the main behavioral interface plus the paper's SEIR reproduction path.

## Citation, license, and data use

The software is released under the MIT License in `LICENSE`. The license covers the code and repository documentation only. No individual-level research data or redistribution permission is granted by publishing this repository; users must obtain and comply with the relevant source-data agreements before supplying private inputs.

