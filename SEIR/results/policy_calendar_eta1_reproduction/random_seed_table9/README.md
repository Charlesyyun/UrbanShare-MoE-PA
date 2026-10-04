# Ten-seed SEIR calendar comparison

This result set contains 50 simulated runs: five calendars, each evaluated
with the same ten random seeds (`1234`-`1243`). The archived timeseries are
the input for the paper's epidemic means and standard deviations. Economic
output proxy `O` is deterministic and comes from
`../inputs/paper_table9_components.csv`, using the
`mobility_augmented_output_all_travel_w031` column. It is unchanged across
SEIR seeds.

From the repository root, regenerate Table 9 statistics and the original
2-by-4 Figure 15 layout with:

```powershell
python SEIR/results/policy_calendar_eta1_reproduction/run_10_seed_table9.py
python SEIR/analysis/plot_figure12_10seed.py
```

The epidemic columns report the mean and sample standard deviation across
the ten seeds. Peak new exposure is the maximum centered 7-day mean of `new_E`;
area metrics are daily sums; epidemic duration counts days with `I > 1`.
The plot is written to `SEIR/figures/`. It keeps observed and other model
curves as their archived references;
the UrbanShare-MoE-PA curve is the ten-seed mean with a plus/minus one standard
deviation band. PNG and PDF outputs use the filename referenced by the paper.

To rerun all 50 simulations, create a local JSON file with paths to the five
calendar daily-share CSVs and the crowding-reference daily-share CSV, then run:

```powershell
python SEIR/results/policy_calendar_eta1_reproduction/run_10_seed_table9.py `
  --rerun --daily-shares-manifest PRIVATE_INPUT_REQUIRED
```

The manifest schema is:

```json
{
  "crowding_reference_daily_shares": "C:/local-data/reference_daily_shares.csv",
  "daily_shares": {
    "Original policy": "C:/local-data/original_daily_shares.csv",
    "Early lockdown": "C:/local-data/early_daily_shares.csv",
    "Late lockdown": "C:/local-data/late_daily_shares.csv",
    "Short lockdown": "C:/local-data/short_daily_shares.csv",
    "Long lockdown": "C:/local-data/long_daily_shares.csv"
  }
}
```

Those agent-day inputs remain external to this public repository. The
calibrated parameter file, model metadata, seed list, economic components,
archived simulation results, and compact figure reference curves are included.

