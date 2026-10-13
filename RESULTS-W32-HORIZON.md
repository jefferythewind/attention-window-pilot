# W32 beyond-horizon test: one paired seed

Eight layers, ordinary window32: baseline horizon 8 × 31 = **248 tokens**. Fresh models train on **320-token sequences**, including distance **280**. Both use seed515843, width48, heads4, batch16, 400 retrieval updates, the original AdamW schedule and full-sequence backpropagation. No text models are trained.

Training uniformly samples distances8/24/48/280 in expectation, preserving four equally sampled distance conditions. Short distances check learnability; distance280 is beyond the sliding architecture’s horizon. Each model processes 2,048,000 tokens and 6,400 answer targets. This is a new longer-sequence experiment, not an out-of-distribution evaluation of the 128-token checkpoints.

| Condition | Distance | Accuracy | Answer NLL | Changed-source accuracy | Prediction changes | Max source logit difference |
|---|---:|---:|---:|---:|---:|---:|
| sliding-32 | 8 | 84.0% | 0.2642 | 91.8% | 91.8% | 7.88419 |
| sliding-32 | 24 | 88.7% | 0.2581 | 87.1% | 87.1% | 7.78276 |
| sliding-32 | 48 | 85.2% | 0.2636 | 91.0% | 91.0% | 7.72593 |
| sliding-32 | 280 | 14.8% | 2.0864 | 15.2% | 0.0% | 0 |
| recurrent-32 | 8 | 85.5% | 0.2797 | 89.1% | 100.0% | 9.2553 |
| recurrent-32 | 24 | 86.7% | 0.2569 | 87.1% | 100.0% | 9.17987 |
| recurrent-32 | 48 | 100.0% | 0.3147 | 100.0% | 100.0% | 8.72524 |
| recurrent-32 | 280 | 27.0% | 1.4348 | 27.0% | 51.2% | 6.25663 |

Chance is 12.5% among eight values; evaluation uses 256 fresh examples per distance, validation seed2026. Answer NLL is in nats. Changed-source evaluation changes only the stored value to value+1 modulo8 and checks the new answer. Cached and whole-sequence logits agree. The answer token is excluded from the prompt.

Initialization, ordered inputs, labels, parameter counts and budgets match by hashes. Both models are evaluated with rolling inference. The sliding model must have exactly zero source influence at distance280; the driver asserts this. One seed provides no between-seed standard deviation and does not establish reliability.

## Timing and state

| Condition | Training seconds | Amortized logical KiB/sequence | Amortized backing KiB/sequence |
|---|---:|---:|---:|
| sliding-32 | 179.3 | 93.008 | 138.011 |
| recurrent-32 | 1766.4 | 94.508 | 97.508 |

End-to-end wall time: 2131.5s. Serial worker execution, one PyTorch thread. Training time includes data creation, forward/backward, clipping and optimizer updates; excludes final evaluation/save. Backing storage includes sliced allocations and shared positions; excludes model weights and allocator/Python overhead. Per-sequence values divide measured batched totals by256; position metadata is shared, so these are amortized sizes rather than batch-one sizes. Raw cache bytes specify their batch size.

## Interpretation

In this single paired trial, recurrent32 achieves 27.0% distance-280 accuracy versus 14.8% for sliding32 (chance12.5%), with changed-source accuracy27.0% versus15.2%. Source substitution changes recurrent predictions on51.2% of examples while sliding logits are exactly invariant. Answer NLL falls from2.0864 to1.4348. This supports partial learned source-sensitive relay beyond horizon248, rather than merely a connected graph. Exact-value retrieval remains far from reliable; one seed does not establish reproducibility. Local recall stays strong in both models (84.0–100.0% across distances8/24/48).

## Reproduce

```bash
.venv/bin/python run_w32_horizon.py --output-dir results-w32-horizon-reproduction
.venv/bin/python summarize_w32_horizon.py
.venv/bin/pip install -r requirements-plots.txt
.venv/bin/python plot_w32_horizon.py
```

The runner reuses the existing training/evaluation helpers but requires no prior checkpoint binaries. Its default output is results-w32-horizon/. Use a new output directory to retrain; the summary reads the preserved default results. Configuration: [w32_horizon_config.json](w32_horizon_config.json). Raw measurements: [results-w32-horizon/runs.json](results-w32-horizon/runs.json).
