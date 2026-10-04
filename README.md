# UrbanShare-MoE-PA

This repository is a code-only release of the UrbanShare-MoE-PA behavioral model and its activity-aware SEIR implementation.

It intentionally contains no datasets, policy calendars, calibrated results, figures, checkpoints, agent-level predictions, or third-party epidemic curves. Those artifacts must be supplied locally under the data-use permissions of the researcher running the code.

## Code layout

```text
.
├── daily_share_model.py              # UrbanShare-MoE model and training interface
├── build_enriched_timeshare.py       # private mobility-to-daily-feature preparation
├── preference_dataset.py             # daily-share dataset contract
├── preference_scorer.py              # preference score model
├── preference_alignment_finetune.py  # preference-alignment training
├── SEIR/seir_timeseries.py           # activity-aware SEIR engine
├── SEIR/calibration/                 # epidemic-curve calibration utility
├── SEIR/analysis/                    # Table 9 and Figure 15 generation code
├── docs/                             # architecture, workflow, and paper mapping
└── tools/validate_release.py         # code-only integrity checks
```

## Required external inputs

Behavioral training requires restricted agent-day activity data and, where applicable, model checkpoints. SEIR runs require daily-share inputs, feature metadata, hazard parameters, policy calendars, and a seed list. The automatic calibration utility additionally requires an archived official epidemic curve in HTML format, readable by `pandas.read_html`.

None of these inputs are redistributed here. The repository supports code inspection and local reproduction by authorized users, but it cannot reproduce the paper's numerical results from a fresh clone alone.

## Checks

```bash
python -m pip install -r requirements.txt
python tools/validate_release.py
python -m compileall -q .
```

See [SEIR/README.md](SEIR/README.md), [DATA.md](DATA.md), and [docs/PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md) for input interfaces and the relationship to the paper.

## License and data use

The code is released under the MIT License in [LICENSE](LICENSE). The license does not grant access to restricted research data or permission to redistribute them.
