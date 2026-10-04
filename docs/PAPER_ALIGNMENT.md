# Paper alignment

The source files implement the behavioral and activity-aware SEIR methods described in the 2026 TRA manuscript. The repository is code-only: the numerical inputs and outputs used for the paper are intentionally not included.

`daily_share_model.py`, `preference_dataset.py`, `preference_scorer.py`, and `preference_alignment_finetune.py` define the behavioral model and preference-alignment stages. `SEIR/seir_timeseries.py` defines the epidemic simulation. The scripts under `SEIR/analysis/` provide the Table 9 and Figure 15 workflows once authorized local inputs are supplied.

The paper's calibrated SEIR parameters, ten-seed trajectories, policy calendars, official epidemic curve, and economic proxy components are external artifacts. They are not reconstructed by placeholder defaults and must not be inferred from this repository alone.

