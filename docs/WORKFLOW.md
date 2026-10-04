# Code workflow

The repository has two reproducibility tracks.

## Paper SEIR track

This self-contained track is the recommended public entry point:

```text
ten-seed SEIR time series + calibrated parameters
                    │
                    ▼
SEIR/results/policy_calendar_eta1_reproduction/run_10_seed_table9.py
                    │
                    ├──► Table 9 summary CSV/Markdown
                    └──► SEIR/analysis/plot_figure12_10seed.py
```

The runner reads the retained 50 compact SEIR time series and the published
economic proxy components. The release validator checks calendars, JSON and
Python syntax, absence of private paths and model weights, required files,
GitHub's file-size limit, and paper artifacts.

## Full training track

`
1. `build_enriched_timeshare.py` converts restricted activity chains into daily features.
2. `daily_share_model.py` trains UrbanShare-MoE.
3. `preference_alignment_finetune.py` trains the preference scorer and aligns the policy.
4. rolls checkpoints forward under a new calendar.
5. `SEIR/seir_timeseries.py` maps behavior to epidemic states.

The full track requires private inputs documented in `DATA.md`; they are
protected by `.gitignore`. Run commands from the repository root. Packaged SEIR
analysis paths are resolved from script locations and do not depend on the shell's directory.

