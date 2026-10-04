# Baseline model inventory

All models share `data.py`, `run.py`, `evaluate.py`, and `paper_metrics.py`.
They produce the same 18-dimensional nonnegative time-share composition and are
evaluated with the same reconstruction and distribution metrics.

| Model | Implementation used in this repository | External source |
|---|---|---|
| Persistence | Deterministic predictor in `run.py` | None |
| Mean-7 | Deterministic rolling-mean predictor in `run.py` | None |
| CatBoost | Training and prediction adapter in `run.py` | Official `catboost` Python package pinned through `requirements.txt` |
| GRU | `GRUAdapter` in `models.py` | PyTorch standard layers |
| LSTM | `LSTMAdapter` in `models.py` | PyTorch standard layers |
| Vanilla Transformer | `TransformerAdapter` in `models.py` | PyTorch standard layers |
| TabM | `TabMAdapter` in `models.py`; runtime snapshot in `third_party/tabm/` | Commit recorded in `sources.lock.json` |
| TimeXer | `TimeXerAdapter` in `models.py`; runtime snapshot in `third_party/TimeXer/` | Commit recorded in `sources.lock.json` |
| Logistic-normal AR | `LogisticNormalARAdapter` in `models.py` | Independent transparent implementation |
| Gamma-profile MDCEV | `MDCEVAdapter` in `models.py` | Independent Python implementation of the documented likelihood and allocation rule |

None of these adapters imports or reuses UrbanShare's MoE, FiLM, lag gate,
policy-alignment loss, three-head decoder, or phase-reset template. This is also
checked by `test_protocol.py`.

The ignored `vendor/` directory may contain full upstream clones used during
development and source auditing. It is not required at runtime and should not
be committed. The compact `third_party/` directory is the runtime source
snapshot intended for GitHub.
