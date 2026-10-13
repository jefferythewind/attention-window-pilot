# One-slot same-layer recurrent attention: results

V1 is preserved; 36 new recurrent models use the same six initializations, original batches, labels and update budgets. Means ± sample SD across six seeds. Text and retrieval are separately trained models.

## Text loss and retrieval accuracy

| Condition | Text NLL ↓ | Recall d=8 | d=24 | d=48 | d=80 |
|---|---:|---:|---:|---:|---:|
| full-128 | 2.5396 ± 0.0075 | 92.6 ± 11.7% | 91.7 ± 12.9% | 91.8 ± 12.8% | 88.7 ± 18.2% |
| sliding-32 | 2.5069 ± 0.0075 | 93.6 ± 15.8% | 93.6 ± 15.8% | 91.8 ± 20.1% | 92.0 ± 19.6% |
| sliding-16 | 2.4752 ± 0.0043 | 47.5 ± 24.7% | 38.1 ± 27.5% | 33.3 ± 29.5% | 30.4 ± 28.1% |
| sliding-8 | 2.4366 ± 0.0030 | 26.6 ± 10.3% | 20.8 ± 6.2% | 15.7 ± 6.7% | 12.5 ± 3.3% |
| chunk-32 | 2.4554 ± 0.0074 | 97.1 ± 7.0% | 98.0 ± 4.8% | 13.1 ± 2.2% | 12.8 ± 3.2% |
| recurrent-32 | 2.3677 ± 0.0045 | 82.0 ± 15.7% | 83.6 ± 14.4% | 78.6 ± 17.0% | 78.8 ± 20.5% |
| recurrent-16 | 2.3447 ± 0.0067 | 40.9 ± 19.3% | 37.0 ± 26.1% | 24.0 ± 6.9% | 23.7 ± 8.0% |
| recurrent-8 | 2.3035 ± 0.0056 | 15.5 ± 7.0% | 15.0 ± 5.6% | 13.8 ± 4.1% | 14.3 ± 3.1% |

V1 chunk-32 uses native disconnected-block evaluation in this table; other conditions use their full/rolling inference. Chance is 12.5% among eight values.

## Retrieval answer loss

| Condition | d=8 NLL | d=24 NLL | d=48 NLL | d=80 NLL |
|---|---:|---:|---:|---:|
| full-128 | 0.3372 ± 0.3053 | 0.3374 ± 0.3186 | 0.3497 ± 0.3230 | 0.3625 ± 0.3450 |
| sliding-32 | 0.2345 ± 0.3804 | 0.2261 ± 0.3674 | 0.2334 ± 0.3888 | 0.2364 ± 0.3956 |
| sliding-16 | 1.3501 ± 0.4981 | 1.4446 ± 0.5737 | 1.4983 ± 0.5832 | 1.5573 ± 0.5588 |
| sliding-8 | 1.6317 ± 0.2557 | 1.7472 ± 0.2902 | 2.0133 ± 0.1958 | 2.1041 ± 0.0204 |
| chunk-32 | 0.3272 ± 0.1918 | 0.3197 ± 0.1833 | 2.0874 ± 0.0147 | 2.0834 ± 0.0111 |
| recurrent-32 | 0.5128 ± 0.3149 | 0.5036 ± 0.3023 | 0.5453 ± 0.3033 | 0.5507 ± 0.3090 |
| recurrent-16 | 1.4767 ± 0.4146 | 1.5349 ± 0.3733 | 1.7151 ± 0.2573 | 1.7203 ± 0.2811 |
| recurrent-8 | 1.9911 ± 0.1619 | 2.0444 ± 0.0919 | 2.0710 ± 0.0659 | 2.0722 ± 0.0333 |

## Changed-source accuracy

The source value is incremented modulo eight; success requires predicting the new value.

| Condition | d=8 | d=24 | d=48 | d=80 |
|---|---:|---:|---:|---:|
| full-128 | 90.0 ± 15.4% | 91.7 ± 12.8% | 92.8 ± 11.2% | 89.1 ± 17.2% |
| sliding-32 | 93.2 ± 16.6% | 93.9 ± 15.0% | 92.0 ± 19.6% | 91.4 ± 21.1% |
| sliding-16 | 48.8 ± 25.1% | 36.3 ± 27.5% | 33.9 ± 27.9% | 34.6 ± 28.0% |
| sliding-8 | 29.4 ± 9.8% | 20.6 ± 6.5% | 17.6 ± 5.2% | 13.9 ± 2.5% |
| chunk-32 | 97.9 ± 5.1% | 97.7 ± 5.7% | 12.9 ± 2.8% | 12.9 ± 3.2% |
| recurrent-32 | 82.9 ± 17.2% | 82.8 ± 15.5% | 80.8 ± 16.1% | 81.3 ± 19.9% |
| recurrent-16 | 39.3 ± 16.7% | 37.5 ± 26.8% | 21.8 ± 5.4% | 20.6 ± 5.3% |
| recurrent-8 | 16.6 ± 5.6% | 14.6 ± 5.4% | 15.6 ± 8.0% | 14.3 ± 5.0% |

## Primary comparison: distance 80 beyond horizon 56

| Seed | Sliding8 recall | Recurrent8 recall | Sliding8 NLL | Recurrent8 NLL | Sliding8 changed source | Recurrent8 changed source | Recurrent8 prediction changes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 515843 | 10.9% | 18.8% | 2.1239 | 2.0061 | 15.2% | 23.4% | 19.9% |
| 363550 | 9.0% | 15.2% | 2.1154 | 2.0857 | 16.4% | 9.4% | 0.0% |
| 519935 | 16.4% | 16.4% | 2.0727 | 2.0728 | 10.9% | 10.9% | 0.0% |
| 354526 | 10.9% | 10.9% | 2.1048 | 2.0908 | 15.2% | 15.2% | 0.0% |
| 292582 | 10.9% | 10.9% | 2.0868 | 2.0819 | 15.2% | 15.2% | 0.0% |
| 57613 | 16.8% | 13.3% | 2.1209 | 2.0958 | 10.5% | 11.7% | 21.9% |

Paired recall advantage, recurrent8 − sliding8: 1.8 ± 4.3 percentage points.
Recurrent8 seeds with at least 90% recall at d=80: 0/6; with at least 90% changed-source accuracy: 0/6. These counts are descriptive, not a predeclared success threshold.

The target of reliable source-sensitive distance-80 retrieval is not met across these six seeds under the specified 400-update budget. This configuration failed to learn reliable relay; it is not evidence that the connection cannot support relay.

At distance 80, predictions change after source substitution in 2/6 recurrent8 seeds. Some source dependence can exist without accurate value recovery. Controlled forward/gradient tests establish the path, not successful task learning.

## Every new seed and distance

### recurrent-32: text

| Seed | Held-out NLL | Training seconds |
|---:|---:|---:|
| 515843 | 2.3623 | 318.1 |
| 363550 | 2.3731 | 289.9 |
| 519935 | 2.3665 | 247.7 |
| 354526 | 2.3630 | 297.8 |
| 292582 | 2.3719 | 236.3 |
| 57613 | 2.3696 | 235.1 |

**Recall**

| Seed | d=8 | d=24 | d=48 | d=80 |
|---:|---:|---:|---:|---:|
| 515843 | 100.0% | 100.0% | 100.0% | 100.0% |
| 363550 | 100.0% | 100.0% | 100.0% | 100.0% |
| 519935 | 85.2% | 85.9% | 73.0% | 88.3% |
| 354526 | 62.9% | 64.5% | 61.7% | 59.4% |
| 292582 | 69.9% | 76.2% | 67.6% | 52.3% |
| 57613 | 74.2% | 75.0% | 69.1% | 73.0% |

**Answer NLL**

| Seed | d=8 | d=24 | d=48 | d=80 |
|---:|---:|---:|---:|---:|
| 515843 | 0.3852 | 0.3940 | 0.4257 | 0.4610 |
| 363550 | 0.0254 | 0.0255 | 0.0266 | 0.0261 |
| 519935 | 0.4715 | 0.4269 | 0.5748 | 0.5522 |
| 354526 | 0.6588 | 0.6510 | 0.6312 | 0.6574 |
| 292582 | 0.9785 | 0.9283 | 0.9327 | 0.9691 |
| 57613 | 0.5574 | 0.5957 | 0.6808 | 0.6387 |

**Changed-source accuracy**

| Seed | d=8 | d=24 | d=48 | d=80 |
|---:|---:|---:|---:|---:|
| 515843 | 100.0% | 100.0% | 100.0% | 100.0% |
| 363550 | 100.0% | 100.0% | 100.0% | 100.0% |
| 519935 | 91.0% | 87.1% | 77.3% | 91.0% |
| 354526 | 56.6% | 62.5% | 67.2% | 63.3% |
| 292582 | 72.7% | 71.9% | 62.1% | 52.0% |
| 57613 | 77.3% | 75.4% | 78.1% | 81.6% |

### recurrent-16: text

| Seed | Held-out NLL | Training seconds |
|---:|---:|---:|
| 515843 | 2.3394 | 232.6 |
| 363550 | 2.3511 | 309.8 |
| 519935 | 2.3382 | 223.8 |
| 354526 | 2.3390 | 410.0 |
| 292582 | 2.3473 | 208.0 |
| 57613 | 2.3535 | 289.2 |

**Recall**

| Seed | d=8 | d=24 | d=48 | d=80 |
|---:|---:|---:|---:|---:|
| 515843 | 53.5% | 36.7% | 27.3% | 26.2% |
| 363550 | 47.7% | 37.1% | 27.7% | 21.5% |
| 519935 | 67.2% | 87.1% | 26.6% | 31.6% |
| 354526 | 12.5% | 14.1% | 11.7% | 10.9% |
| 292582 | 32.4% | 23.4% | 30.5% | 32.0% |
| 57613 | 32.0% | 23.8% | 19.9% | 19.9% |

**Answer NLL**

| Seed | d=8 | d=24 | d=48 | d=80 |
|---:|---:|---:|---:|---:|
| 515843 | 1.2240 | 1.2981 | 1.4523 | 1.4727 |
| 363550 | 1.3788 | 1.4980 | 1.8142 | 1.9462 |
| 519935 | 0.9269 | 1.0549 | 1.6497 | 1.5242 |
| 354526 | 2.0783 | 2.0826 | 2.0923 | 2.0954 |
| 292582 | 1.4296 | 1.4282 | 1.4235 | 1.4282 |
| 57613 | 1.8225 | 1.8474 | 1.8585 | 1.8552 |

**Changed-source accuracy**

| Seed | d=8 | d=24 | d=48 | d=80 |
|---:|---:|---:|---:|---:|
| 515843 | 52.0% | 36.7% | 25.8% | 24.6% |
| 363550 | 50.4% | 38.7% | 17.6% | 14.8% |
| 519935 | 54.3% | 88.3% | 21.9% | 20.3% |
| 354526 | 10.9% | 10.9% | 15.6% | 15.2% |
| 292582 | 37.5% | 23.8% | 19.9% | 19.9% |
| 57613 | 30.5% | 26.6% | 30.1% | 28.5% |

### recurrent-8: text

| Seed | Held-out NLL | Training seconds |
|---:|---:|---:|
| 515843 | 2.3029 | 311.5 |
| 363550 | 2.3093 | 305.5 |
| 519935 | 2.3031 | 214.1 |
| 354526 | 2.3030 | 399.6 |
| 292582 | 2.3090 | 190.8 |
| 57613 | 2.2940 | 278.4 |

**Recall**

| Seed | d=8 | d=24 | d=48 | d=80 |
|---:|---:|---:|---:|---:|
| 515843 | 28.5% | 26.2% | 21.5% | 18.8% |
| 363550 | 10.9% | 10.9% | 15.6% | 15.2% |
| 519935 | 17.2% | 11.7% | 10.9% | 16.4% |
| 354526 | 9.0% | 14.1% | 11.7% | 10.9% |
| 292582 | 12.5% | 14.1% | 11.7% | 10.9% |
| 57613 | 14.8% | 12.9% | 11.3% | 13.3% |

**Answer NLL**

| Seed | d=8 | d=24 | d=48 | d=80 |
|---:|---:|---:|---:|---:|
| 515843 | 1.6714 | 1.8570 | 1.9484 | 2.0061 |
| 363550 | 2.0875 | 2.0821 | 2.0760 | 2.0857 |
| 519935 | 2.0711 | 2.0863 | 2.0841 | 2.0728 |
| 354526 | 2.0523 | 2.0831 | 2.0886 | 2.0908 |
| 292582 | 2.0874 | 2.0806 | 2.0800 | 2.0819 |
| 57613 | 1.9771 | 2.0776 | 2.1489 | 2.0958 |

**Changed-source accuracy**

| Seed | d=8 | d=24 | d=48 | d=80 |
|---:|---:|---:|---:|---:|
| 515843 | 26.2% | 25.0% | 30.9% | 23.4% |
| 363550 | 13.7% | 11.7% | 10.2% | 9.4% |
| 519935 | 12.5% | 14.1% | 11.7% | 10.9% |
| 354526 | 17.2% | 10.9% | 15.6% | 15.2% |
| 292582 | 10.9% | 10.9% | 15.6% | 15.2% |
| 57613 | 19.1% | 14.8% | 9.4% | 11.7% |

## Timings

Training durations overlap across three workers; each uses one CPU thread. New recurrent timings exclude evaluation and checkpoint writes. V1 rows retain their historical measurements. They are contextual observations, not a new controlled speed benchmark.

| Condition | Text training seconds | Retrieval training seconds |
|---|---:|---:|
| full-128 | 24.2694 ± 6.9533 | 40.6285 ± 5.7921 |
| sliding-32 | 22.7042 ± 3.8141 | 38.4146 ± 2.5352 |
| sliding-16 | 24.6633 ± 7.7608 | 41.7799 ± 4.9539 |
| sliding-8 | 22.4171 ± 3.3541 | 38.3939 ± 2.6762 |
| chunk-32 | 17.7109 ± 2.8703 | 33.8208 ± 6.5985 |
| recurrent-32 | 270.8026 ± 35.5782 | 627.0919 ± 100.7374 |
| recurrent-16 | 278.9126 ± 75.4362 | 717.8164 ± 110.6501 |
| recurrent-8 | 283.3182 ± 75.0602 | 685.2683 ± 150.4034 |

End-to-end new-run wall time: 6188.8s. Sum of overlapping new-model training durations: 17179.3s.

## Retained inference state

The following logical tensor sizes use batch size one, float32 K/V and completed outputs, int64 positions. W counts ordinary positions including current; only W−1 ordinary positions are retained after a token. Recurrent slot K/V are recomputed from the one completed output rather than stored.

| Condition | Ordinary KV KiB | Recurrent output KiB | Position metadata KiB | Total logical KiB |
|---|---:|---:|---:|---:|
| full-128 | 381.000 | 0.000 | 7.938 | 388.938 |
| sliding-32 | 93.000 | 0.000 | 1.938 | 94.938 |
| sliding-16 | 45.000 | 0.000 | 0.938 | 45.938 |
| sliding-8 | 21.000 | 0.000 | 0.438 | 21.438 |
| recurrent-32 | 93.000 | 1.500 | 2.000 | 96.500 |
| recurrent-16 | 45.000 | 1.500 | 1.000 | 47.500 |
| recurrent-8 | 21.000 | 1.500 | 0.500 | 23.000 |

### Actual cache backing allocations

Batch one, 127 processed tokens, chunk-16 prefill; same input shape and external chunk policy. The baseline vectorizes each prefill chunk; recurrence internally processes single tokens. Logical tensor bytes differ from backing allocations when eviction uses slices; backing sizes therefore depend on execution/chunking, not only architecture.

| Condition | Total logical KiB | Deduplicated backing KiB |
|---|---:|---:|
| full-128 | 388.938 | 388.938 |
| sliding-32 | 94.938 | 140.875 |
| sliding-16 | 45.938 | 91.875 |
| sliding-8 | 21.438 | 67.375 |
| recurrent-32 | 96.500 | 99.562 |
| recurrent-16 | 47.500 | 50.562 |
| recurrent-8 | 23.000 | 26.062 |

Raw JSON also records actual batched-cache logical bytes and deduplicated backing-storage bytes. Position metadata is shared across the batch. Sliced K/V views may retain larger backing allocations; weights, allocator overhead, inference temporaries and the training graph are excluded from this table. No training-memory reduction is claimed.

## Interpretation

The old sliding8 architecture has no input-to-query path at distance 80; its horizon is 56. The recurrent version has a path, verified independently by controlled forward and gradient tests. Useful relay still requires accurate, source-sensitive held-out retrieval across seeds. A connected graph or a tiny nonzero logit change alone does not establish it.

Recurrent8 also struggles at the short distance 8 (15.5% mean accuracy), so this failure does not isolate long-range transmission from broader learning/optimization difficulty. This configuration failed to learn reliable relay under the initial budget; it does not prove the connection cannot support relay. No seeds are excluded, and seeds change both initialization and training sampling. The finite training budget, absolute positional embeddings and single-value synthetic task limit generalization claims.

Protocol: [RECURRENT-DESIGN.md](RECURRENT-DESIGN.md). Config: [recurrent_config.json](recurrent_config.json). Raw metrics: [results-recurrent/runs.json](results-recurrent/runs.json).
