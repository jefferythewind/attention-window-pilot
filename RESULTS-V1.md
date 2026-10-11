# V1 hello world: attention-window experiment

One setting: 128-token sequences, eight layers, width 48, four heads; six new paired seeds.
Full attention is the baseline. Sliding attention uses windows 32/16/8. The independent-chunk
training control is evaluated only at W=32. Text prediction and retrieval train separate models.

## Primary results

Entries are mean ± sample standard deviation across seeds. Each condition uses its native
inference rule: rolling attention for full/sliding, disconnected blocks for chunk-trained.

| Condition | Text NLL ↓ | Recall d=8 | d=24 | d=48 | d=80 |
|---|---:|---:|---:|---:|---:|
| full-128 | 2.5396 ± 0.0075 | 92.6 ± 11.7% | 91.7 ± 12.9% | 91.8 ± 12.8% | 88.7 ± 18.2% |
| sliding-32 | 2.5069 ± 0.0075 | 93.6 ± 15.8% | 93.6 ± 15.8% | 91.8 ± 20.1% | 92.0 ± 19.6% |
| sliding-16 | 2.4752 ± 0.0043 | 47.5 ± 24.7% | 38.1 ± 27.5% | 33.3 ± 29.5% | 30.4 ± 28.1% |
| sliding-8 | 2.4366 ± 0.0030 | 26.6 ± 10.3% | 20.8 ± 6.2% | 15.7 ± 6.7% | 12.5 ± 3.3% |
| chunk-32 | 2.4554 ± 0.0074 | 97.1 ± 7.0% | 98.0 ± 4.8% | 13.1 ± 2.2% | 12.8 ± 3.2% |

NLL is nats per character; recall is exact-match answer accuracy. Chance is 12.5% among eight values.
Seed variation, not the number of held-out examples, defines the dispersion reported above.

## W=32 training/inference comparison

The same two sets of W=32 trained checkpoints are tested with both inference rules.
This is an evaluation matrix, not four separately trained conditions.

| Training / inference | Text NLL ↓ | Recall d=8 | d=24 | d=48 | d=80 |
|---|---:|---:|---:|---:|---:|
| Sliding trained / rolling | 2.5069 ± 0.0075 | 93.6 ± 15.8% | 93.6 ± 15.8% | 91.8 ± 20.1% | 92.0 ± 19.6% |
| Sliding trained / chunks | 2.4978 ± 0.0098 | 93.6 ± 15.8% | 93.6 ± 15.8% | 11.5 ± 2.2% | 12.0 ± 3.0% |
| Chunk trained / rolling | 2.5296 ± 0.0059 | 97.1 ± 7.0% | 98.0 ± 4.8% | 20.6 ± 8.9% | 12.8 ± 3.2% |
| Chunk trained / chunks | 2.4554 ± 0.0074 | 97.1 ± 7.0% | 98.0 ± 4.8% | 13.1 ± 2.2% | 12.8 ± 3.2% |

## Individual retrieval seeds

| Seed | Full d=80 | Sliding32 d=8 | Sliding32 d=80 | Sliding16 d=80 | Sliding8 d=80 | Chunk32 native d=8 | Chunk32 native d=80 | Chunk32 rolling d=80 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 515843 | 100.0% | 100.0% | 100.0% | 85.2% | 10.9% | 100.0% | 10.9% | 10.9% |
| 363550 | 100.0% | 100.0% | 100.0% | 32.0% | 9.0% | 100.0% | 11.7% | 11.7% |
| 519935 | 100.0% | 100.0% | 100.0% | 20.3% | 16.4% | 100.0% | 16.8% | 16.8% |
| 354526 | 57.8% | 100.0% | 100.0% | 9.4% | 10.9% | 82.8% | 10.9% | 10.9% |
| 292582 | 74.6% | 100.0% | 100.0% | 10.9% | 10.9% | 100.0% | 9.4% | 9.4% |
| 57613 | 100.0% | 61.3% | 52.0% | 24.6% | 16.8% | 100.0% | 16.8% | 16.8% |

## Paired differences

Each difference subtracts the other condition from sliding-32 within the same seed.
Positive accuracy differences favor sliding-32; negative NLL differences favor sliding-32.

| Paired comparison | Text NLL difference | d=80 accuracy difference (percentage points) |
|---|---:|---:|
| Sliding32 − full | -0.0328 ± 0.0025 | 3.3 ± 30.6 |
| Sliding32 rolling − chunk32 native | 0.0514 ± 0.0035 | 79.2 ± 21.7 |
| Sliding32 rolling − chunk32 rolling | -0.0228 ± 0.0053 | 79.2 ± 21.7 |

## Answer loss and source counterfactuals at distance 80

| Condition / inference | Answer NLL ↓ | Changed-source accuracy |
|---|---:|---:|
| full-128 native | 0.3625 ± 0.3450 | 89.1 ± 17.2% |
| sliding-32 native | 0.2364 ± 0.3956 | 91.4 ± 21.1% |
| sliding-16 native | 1.5573 ± 0.5588 | 34.6 ± 28.0% |
| sliding-8 native | 2.1041 ± 0.0204 | 13.9 ± 2.5% |
| chunk-32 native | 2.0834 ± 0.0111 | 12.9 ± 3.2% |
| chunk-32 rolling transfer | 2.0813 ± 0.0105 | 12.9 ± 3.2% |

## KV storage and budgets

| Rolling attention window | Retained KV KiB per sequence, float32 |
|---:|---:|
| 128 | 381.0 |
| 32 | 93.0 |
| 16 | 45.0 |
| 8 | 21.0 |

W=32 retains 31 prior keys/values per layer versus 127 for full attention: 24.4% of KV
tensor storage. This is not a claim of quarter total memory or quarter training cost.
Position metadata, weights, activations, allocator overhead and transient prefill are excluded.

- Seeds, selected before training: 515843, 363550, 519935, 354526, 292582, 57613.
- Text: 200 updates × 16 sequences × 128 tokens = 409,600 processed tokens; 406,400 targets/run.
- Retrieval: 400 updates × 16 sequences × 128 tokens = 819,200 processed tokens; 6,400 targets/run.
- Same held-out examples across every condition and seed: 512 text sequences, 256 retrieval examples/distance.
- Completed models: 60; three CPU workers, one PyTorch thread per worker.
- Recorded end-to-end wall time: 758.7s.
- Sum of overlapping per-model timed training durations: 1828.8s.

All input, label, initialization, parameter-count, token, target and optimizer-update checks passed.
Cached/full-sequence equivalence and native chunk source invariance passed in evaluation.

## Interpretation

Local and distant failures may indicate a broader learning/optimization limitation, but the
seed changes both initialization and training-batch sampling; initializer quality is not isolated.
No seeds are excluded or selected based on performance. Six seeds remain a preliminary check,
not a demonstration of converged general capability equivalence.

The native chunk control cannot recover values in other chunks. Its rolling transfer evaluation
holds inference fixed to test training adaptation, but earlier chunks and their learned position
embeddings receive no gradient under answer-only supervision. RoPE or fixed embeddings would
help investigate the contribution of positional learning. Exact-match and answer NLL may differ.

Reach is finite: eight layers give horizons 248, 120, 56 for W=32,16,8. Distance 80 is structurally
unreachable for W=8. Text and retrieval columns refer to different trained models, not one model
demonstrating both capabilities. Dense masks do not establish sparse-attention training savings.

Raw data: [results-v1/runs.json](results-v1/runs.json). Protocol: [DESIGN.md](DESIGN.md).
