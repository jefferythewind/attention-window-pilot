# Doubled-depth and doubled-context CPU experiment

8 layers, 128-token sequences, width 48, four attention heads.
Two separately trained tasks and two paired seeds, following the first pilot.

**Headline:** Restricted windows retain substantial information relay, but the fourfold
reduction does not reliably match full-attention retrieval at this budget. W=32 achieves
100% at distance 80 in seed 17 and 60.9% in seed 29, versus 100% for full attention in both.
Separately trained text models still show lower early held-out loss for smaller windows.

| Window | Text NLL, mean (seed range) | Recall d=8 | Recall d=24 | Recall d=48 | Recall d=80 | KV KiB/sequence* |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2.5433 (2.5331–2.5535) | 100.0% | 100.0% | 100.0% | 100.0% | 381.0 |
| 32 | 2.5124 (2.5009–2.5240) | 81.6% | 80.5% | 81.6% | 80.5% | 93.0 |
| 16 | 2.4775 (2.4683–2.4866) | 53.1% | 31.6% | 28.7% | 27.1% | 45.0 |
| 8 | 2.4414 (2.4302–2.4526) | 17.6% | 18.8% | 11.3% | 12.7% | 21.0 |

*Steady-state float32 KV tensors; excludes metadata, activations and transient allocations.

Text NLL is nats per character. Text and retrieval columns use separately trained models.
Training does not cache or detach states: all complete sequences retain their graph.

## Retrieval controls and variation across seeds

| Window | Distance | Accuracy per seed | Answer NLL per seed | Final-window recomputation per seed | Changed-source accuracy per seed |
|---:|---:|---|---|---|---|
| 128 | 8 | 100.0% / 100.0% | 0.0357 / 0.1024 | 100.0% / 100.0% | 100.0% / 100.0% |
| 128 | 24 | 100.0% / 100.0% | 0.0350 / 0.1065 | 100.0% / 100.0% | 100.0% / 100.0% |
| 128 | 48 | 100.0% / 100.0% | 0.0326 / 0.1023 | 100.0% / 100.0% | 100.0% / 100.0% |
| 128 | 80 | 100.0% / 100.0% | 0.0323 / 0.0919 | 100.0% / 100.0% | 100.0% / 100.0% |
| 32 | 8 | 100.0% / 63.3% | 0.1446 / 0.8538 | 100.0% / 63.3% | 100.0% / 59.0% |
| 32 | 24 | 100.0% / 60.9% | 0.1389 / 0.8720 | 100.0% / 60.9% | 100.0% / 61.3% |
| 32 | 48 | 100.0% / 63.3% | 0.1323 / 0.8881 | 9.8% / 11.3% | 100.0% / 62.1% |
| 32 | 80 | 100.0% / 60.9% | 0.1278 / 0.9208 | 10.5% / 9.4% | 100.0% / 68.8% |
| 16 | 8 | 64.8% / 41.4% | 1.1178 / 1.4348 | 60.9% / 40.2% | 62.5% / 54.3% |
| 16 | 24 | 39.5% / 23.8% | 1.1829 / 1.8628 | 14.1% / 14.1% | 36.7% / 23.4% |
| 16 | 48 | 30.1% / 27.3% | 1.4696 / 1.4691 | 15.2% / 11.7% | 21.5% / 25.8% |
| 16 | 80 | 28.1% / 26.2% | 1.4805 / 1.4494 | 11.7% / 10.9% | 19.9% / 24.6% |
| 8 | 8 | 21.5% / 13.7% | 1.7352 / 2.0819 | 17.2% / 9.0% | 28.1% / 9.0% |
| 8 | 24 | 24.6% / 12.9% | 1.7827 / 2.0814 | 11.7% / 12.9% | 25.8% / 11.7% |
| 8 | 48 | 10.9% / 11.7% | 2.0920 / 2.0810 | 10.9% / 11.7% | 11.7% / 10.9% |
| 8 | 80 | 16.4% / 9.0% | 2.0796 / 2.0829 | 16.4% / 9.0% | 10.9% / 16.4% |

Seed order: 17, 29.

The weaker W=32 seed also scores only 63.3% at distance 8, inside its direct window.
Its errors are therefore not confined to long-range relay. These results suggest a
training/optimization limitation may contribute, but do not establish its cause.

## Comparison with the first pilot

| Setting | First pilot | Doubled experiment |
|---|---:|---:|
| Layers | 4 | 8 |
| Sequence length | 64 | 128 |
| Windows | 64, 16, 8, 4 | 128, 32, 16, 8 |
| Retrieval distances | 4, 12, 24, 40 | 8, 24, 48, 80 |
| Updates (text / retrieval) | 200 / 400 | 200 / 400 |
| Processed tokens per run (text / retrieval) | 204,800 / 409,600 | 409,600 / 819,200 |
| Supervised targets per run (text / retrieval) | 201,600 / 6,400 | 406,400 / 6,400 |
| Timed training, all runs | 76.8s | 470.0s |

Width, heads, tokenizer, corpus split, batch size, seeds, optimizer and learning-rate
schedule remain unchanged. Models are trained from scratch, not continued from the first pilot.
The new sweep uses one CPU thread instead of four to reduce threading overhead; timing
comparisons across pilots also reflect this execution-setting difference.
Matching initialization, sequence order and budgets was verified within each new task and seed.

This jointly changes depth, sequence length, windows, retrieval distances and processed-token
budget. It is not an isolated test of depth or a token-matched comparison with the first pilot.
The text evaluations also differ in sequence length and sampled positions, so cross-pilot
loss differences are descriptive rather than controlled effects.

Maximum structural reach is L(W-1): W=128: 1016, W=32: 248, W=16: 120, W=8: 56 positions.
In particular, distance 80 is within the W=16 horizon (120), but outside W=8 (56).
Reachability permits retrieval; it does not guarantee it is learned in this training budget.

Cached and full-sequence logits were checked for equivalence on every retrieval evaluation.
Two seeds and a simple single-value probe do not establish general capability equivalence.
Matching exact-match accuracy need not mean matching answer NLL; both are reported above.
Dense attention masking does not establish sparse-kernel training savings.

Raw metrics: [results-2x/runs.json](results-2x/runs.json).
Protocol and reproduction command: [DESIGN.md](DESIGN.md#doubled-depth-and-context-experiment).
