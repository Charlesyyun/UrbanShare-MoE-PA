# SEIR release

This directory contains the activity-aware SEIR implementation, calibration utilities, policy-calendar definitions, and aggregate paper artifacts. Individual-level behavior trajectories and the 50 per-seed simulation files are excluded from the public release.

## Released artifacts

- `figures/fig15_seir_policy_compare_four_calendars_10seed.png` and `.pdf`: the paper-format ten-seed comparison figure.
- `results/policy_calendar_eta1_reproduction/random_seed_table9/table9_10_seed_summary.csv` and `.md`: aggregate Table 9 output.
- `results/policy_calendar_eta1_reproduction/random_seed_table9/epidemic_summary_by_calendar.csv`: mean, standard deviation, minimum, and maximum epidemic metrics by calendar.
- `results/hazard_params_recalibrated_2026.json` and related calibration files: calibrated population-level SEIR parameters.

## Re-running

The summary files can be inspected directly and are checked by `python tools/validate_release.py`. Rebuilding the stochastic trajectories requires a private daily-share manifest and the corresponding restricted inputs; `run_10_seed_table9.py` documents that interface and will report a clear missing-input error when `--rerun` is requested without it.
