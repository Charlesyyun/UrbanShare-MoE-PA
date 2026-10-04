# Model architecture

## UrbanShare-MoE behavior model

For each agent-day, `DailyShareDataset` supplies an agent index, epidemic phase,
temporal/demographic context, lagged behavior, category shares, travel-mode
shares, and the total travel fraction.

```text
agent id ─► agent embedding ───────────────┐
phase ────► phase embedding ─► FiLM ──────┤
context ─► phase-gated lags ─► context MLP├─► fused representation
                                            │
                                            ├─► sigmoid travel head
                                            ├─► top-k MoE category head ─► sparsemax
                                            └─► top-k MoE mode head ─────► sparsemax
```

Category and mode heads contain configurable MLP experts and learned routers;
only top-k expert outputs are mixed. A balancing loss discourages router collapse.
Phase embeddings modulate the context encoder through FiLM and can suppress
lagged features after policy transitions. Outputs are category shares, travel
mode shares, and a scalar travel fraction. The baseline keeps the same interface
without the MoE specialization.

## Preference alignment

`preference_alignment_finetune.py` first learns `PreferenceScoreNet`, a
state-action scorer trained on rankings between observed actions, policy outputs,
wrong-phase references, and perturbed negatives. It then updates the policy with
preference, behavior-cloning, anchoring, phase-response, sparsity, and optional
one-step rollout objectives. Factual and counterfactual profiles expose different
conservative/alignment presets.

## Counterfactual rollout

`counterfactual_pipeline.py` loads a dense, MoE, or PA checkpoint, replaces the
factual policy calendar, and updates lag features sequentially. Every variant
emits the same daily-share schema consumed by SEIR.

## Activity-aware SEIR

`SEIRTimeseries` maintains agent-level susceptible, exposed, infectious, and
recovered states. Daily risk combines base hazard, phase multipliers, category
contact rates, travel modes, demographics, importations, and home protection.
It emits compartment counts and internal/external exposures. The POI variant
adds explicit allocated POI hours when those restricted inputs are available.
