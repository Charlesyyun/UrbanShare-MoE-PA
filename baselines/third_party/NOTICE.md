# Third-party source snapshots

This directory contains only the upstream files needed to run the TabM and
TimeXer adapters. The snapshots were copied from the commits pinned in
`../sources.lock.json`; unrelated examples, datasets, environments, Git
history, and build artefacts are intentionally excluded.

- `tabm/`: selected files from Yandex Research TabM, commit
  `28e47ae301c92ec37787dde1ce923a0793f405b4`. Its upstream license is kept at
  `tabm/LICENSE`.
- `TimeXer/`: the model and dependency modules used by the TimeXer adapter,
  commit `76011909357972bd55a27adba2e1be994d81b327`. See its included upstream
  README and repository URL in `../sources.lock.json`.

CatBoost is installed as an official Python dependency rather than copied into
this repository. GRU, LSTM, vanilla Transformer, logistic-normal AR, and the
gamma-profile MDCEV implementation are contained in `../models.py`.
