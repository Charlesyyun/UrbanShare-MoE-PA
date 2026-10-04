# Code workflow

## Behavioral stage

1. `build_enriched_timeshare.py` converts authorized mobility inputs into daily features.
2. `daily_share_model.py` trains UrbanShare-MoE.
3. `preference_alignment_finetune.py` trains the preference scorer and alignment objective.
4. The resulting daily-share files are passed to the SEIR stage.

## SEIR stage

1. Provide daily shares, feature metadata, hazard parameters, policy calendars, and the seed list locally.
2. Run `SEIR/seir_timeseries.py` for individual simulations or `SEIR/analysis/run_10_seed_table9.py` for the ten-seed comparison.
3. Run `SEIR/analysis/plot_figure12_10seed.py` to generate the paper-format policy figure.
4. For parameter fitting, provide `official_curve.html` or pass `--official_curve_html` to `SEIR/calibration/auto_calibrate_seir.py`.

All numerical outputs should be written outside the repository. The public tree contains only source code and documentation.

