# Code workflow

The repository has two reproducibility tracks.

## Packaged-data track

This self-contained track is the recommended public entry point:

```text
policy calendar + released behavior trajectory
                    │
                    ▼
SEIR/analysis/run_counterfactual_scenarios.py
                    │
                    ▼
SEIR/seir_timeseries.py
                    │
                    ├──► SEIR/results/scenarios/<policy>/<model>/seir_timeseries.csv
                    ├──► SEIR/analysis/plot_paper_figure15.py
                    └──► SEIR/analysis/generate_table6_from_paper.py
```

The runner reads four calendars and twelve compressed behavior files, applies
the released calibrated hazard parameters, and records a summary. The release
validator checks hashes, dimensions, privacy-sensitive columns, calendars,
Python syntax, required files, GitHub's file-size limit, and paper artifacts.

## Full training track

`run_full_pipeline.py` coordinates the raw-data workflow:

1. `build_enriched_timeshare.py` converts restricted activity chains into daily features.
2. `make_hours_from_daily_shares_mp.py` materializes optional hourly allocations.
3. `daily_share_model.py` trains UrbanShare-MoE; `daily_share_model_baseline.py`
   supplies the dense comparison.
4. `preference_alignment_finetune.py` trains the preference scorer and aligns the policy.
5. `counterfactual_pipeline.py` rolls checkpoints forward under a new calendar.
6. `SEIR/seir_timeseries.py` maps behavior to epidemic states. The optional POI
   branch also uses the two POI-allocation scripts and `seir_timeseries_poi.py`.

The full track requires private inputs documented in `DATA.md`; they are
protected by `.gitignore`. Run commands from the repository root. Packaged SEIR
analysis paths are resolved from script locations and do not depend on the shell's directory.
