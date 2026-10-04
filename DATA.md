# Data and reproducibility

## Public contents

This release contains no individual-level behavior CSV/CSV.GZ files, raw mobility chains, checkpoints, agent-level predictions, or private training outputs. The `data/behavior/` and policy `behavior/` directories are intentionally empty release placeholders. The policy directories retain only their 184-day calendar definitions and short README files.

Public numerical artifacts are aggregate or population-level outputs, including calibrated SEIR parameters, epidemic summaries, Table 9 components, and paper figures. These artifacts are suitable for checking the published calculations but are not substitutes for the restricted agent-day inputs used to train the behavioral model.

## Calendars

Each policy directory includes a `calendar.csv` covering 2020-03-01 through 2020-08-31. The intervention windows are:

| Scenario | Start | End |
|---|---:|---:|
| Early lockdown | 2020-03-31 | 2020-05-25 |
| Late lockdown | 2020-04-14 | 2020-06-08 |
| Short lockdown | 2020-04-07 | 2020-05-18 |
| Long lockdown | 2020-04-07 | 2020-06-15 |

## Paper-output provenance

The ten-seed Table 9 summary is stored in `SEIR/results/policy_calendar_eta1_reproduction/random_seed_table9/table9_10_seed_summary.csv` and `.md`. Its underlying population-level summary is `epidemic_summary_by_calendar.csv`; the economic proxy components are in `inputs/model_metadata/paper_table9_components.csv`. Figure 15 is stored in `SEIR/figures/fig15_seir_policy_compare_four_calendars_10seed.png` and `.pdf`.

`Observed reference`, `PA-generated original`, and the four counterfactual calendars are distinct quantities. The former is an external observed/calibration reference; the latter five are model-based policy-calendar outputs. They must not be merged when interpreting Table 9.

## Responsible use

The code is public, but the source data used for behavioral training are restricted. Do not infer that pseudonymous identifiers, if used in a private input, guarantee anonymity. Before supplying any local data to the scripts, verify that redistribution, processing, and derived-output sharing are allowed by the applicable data agreement.
