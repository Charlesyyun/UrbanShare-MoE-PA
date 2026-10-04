# UrbanShare-MoE-PA

This repository contains the paper-aligned implementation of the UrbanShare-MoE-PA behavioral model and the activity-aware SEIR analysis used in the accompanying paper.

The public release is intentionally code-first. It contains source code, policy calendars, calibrated SEIR parameters, aggregated paper tables, and figure artifacts. It does **not** contain individual-level mobility trajectories, raw activity chains, checkpoints, agent-level predictions, or private training outputs.

## What is reproducible from this repository

- Inspect and run the activity-aware SEIR implementation in `SEIR/`.
- Reproduce the released aggregate Table 9 and Figure 15 artifacts from `SEIR/results/policy_calendar_eta1_reproduction/random_seed_table9/` and `SEIR/figures/`.
- Inspect the four policy calendars and the model/configuration contracts.
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

The released aggregate table and figures are already stored in the repository. Regenerating the 50 stochastic trajectories requires a local private-input manifest; see `SEIR/results/policy_calendar_eta1_reproduction/random_seed_table9/README.md`.

## Repository map

```text
.
├── daily_share_model.py              # UrbanShare-MoE model and training interface
├── daily_share_model_baseline.py     # dense UrbanShare baseline
├── preference_dataset.py             # daily-share dataset contract
├── preference_alignment_finetune.py  # preference-alignment fine-tuning
├── counterfactual_pipeline.py        # policy-calendar rollout interface
├── run_full_pipeline.py              # private-data end-to-end pipeline
├── SEIR/                             # activity-aware SEIR code and paper artifacts
├── dataset/MetaData/                 # public model configuration
├── early|late|short|long/            # policy calendars and release placeholders
├── baselines/                         # baseline implementations and protocols
├── MoE-PA-ablations/                 # ablation code and compact design tables
├── docs/                              # architecture, workflow, and paper alignment
└── tools/                             # public release validation utilities
```

## Citation, license, and data use

The software is released under the MIT License in `LICENSE`. The license covers the code and repository documentation only. No individual-level research data or redistribution permission is granted by publishing this repository; users must obtain and comply with the relevant source-data agreements before supplying private inputs.
