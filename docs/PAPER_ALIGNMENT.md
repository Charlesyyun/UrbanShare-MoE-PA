# Paper alignment

## Reference

This public tree is aligned to the 2026 TRA manuscript (`2026_TRA_MoE_PA.pdf`) and the ten-seed SEIR policy-calendar analysis used for Figure 15 and Table 9.

## Model and policy mapping

- `daily_share_model.py`, `preference_alignment_finetune.py`, and implement the UrbanShare-MoE-PA behavioral pipeline and its policy-calendar rollout interface.
- `SEIR/seir_timeseries.py` is the activity-aware SEIR engine.
- `early`, `late`, `short`, and `long` retain the paper's calendar definitions. Their individual behavior files are intentionally omitted from the public release.
- The paper-aligned ten-seed aggregate is in `SEIR/results/policy_calendar_eta1_reproduction/random_seed_table9/`.

## Table 9 and Figure 15

The public Table 9 artifact uses the ten seeds listed in `inputs/model_metadata/seir_seed_list_911_trend_v1.csv`. `table9_10_seed_summary.csv` reports the mean and standard deviation across those seeds for the model-based original policy and the four counterfactual calendars. The economic proxy is taken from `inputs/model_metadata/paper_table9_components.csv`.

The observed reference value used for calibration is conceptually separate from the PA-generated original-policy simulation. It should not be read as a sixth simulated calendar or substituted into the stochastic mean/std calculation.

`SEIR/figures/fig15_seir_policy_compare_four_calendars_10seed.png` is the released paper-format figure. The public tree preserves the aggregate figure and summary tables, but not the agent-day behavior or per-seed trajectory files from which they were generated.

## Reproducibility limits

A user with the restricted daily-share inputs, the calibrated parameters, and compatible software can use the supplied scripts to regenerate the stochastic runs. A user with this public repository alone can audit code structure, calendars, parameter files, aggregate outputs, and figure artifacts, but cannot retrain the behavioral model or independently regenerate agent-level rollouts.

