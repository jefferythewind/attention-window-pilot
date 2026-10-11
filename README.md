# Attention-window CPU pilot

**A fourfold reduction in attention window preserved controlled long-range retrieval; separately trained text models showed no held-out loss penalty in this small pilot.**

This experiment follows Timothy DeLise and Seth Cromelin’s [*Thinking Outside the Box: Retention and Transmission of Information in Sliding-Window KV Inference*](https://arxiv.org/abs/2609.34049) (arXiv:2609.34049, 2026). It asks whether training with restricted attention from the start lets a transformer learn to carry information beyond its immediate window.

We trained four-layer, approximately 140K-parameter decoder-only transformers from scratch in PyTorch, with a minimal Hugging Face model interface. Every condition processed complete 64-token sequences with full backpropagation. Within each task and seed, initialization, ordered batches, and training-token budgets were identical; only the causal attention window changed. The CPU sweep used two paired seeds and took about 77 seconds of timed training.

## Preliminary results

| Attention window | Held-out text loss ↓ | Retrieval at distance 40 ↑ |
|---:|---:|---:|
| 64 (full attention) | 2.5255 | 100% |
| 16 | 2.4937 | 100% |
| 8 | 2.4673 | 13.1% |
| 4 | 2.3949 | 13.3% |

**These columns describe separately trained models.** Text models predict the next character on held-out Shakespeare text. Retrieval models learn to reproduce one random stored value after random distractors, with only the answer supervised. We have not evaluated retrieval on the text-trained checkpoints.

The 16-token-window probe achieved 100% retrieval at distances 4, 12, 24, and 40 in both seeds. At distance 40, recomputing only the final window gave approximately chance accuracy (12.5%); changing the stored value changed the model’s answer correctly. Smaller windows still succeeded at some shorter distances.

These are early learning results, not evidence of converged general capability equivalence. Four local-attention layers have a maximum reach of `4 × (window − 1)` positions: 60 for the 16-token window. Forty is the longest distance tested. The dense masked training implementation does not establish sparse-attention compute savings.

## Reproduce on CPU

```bash
git clone https://github.com/jefferythewind/attention-window-pilot.git
cd attention-window-pilot
python -m venv .venv
.venv/bin/pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers==4.57.6 numpy==2.5.4
.venv/bin/python checks.py
.venv/bin/python run.py
.venv/bin/python summarize.py
```

The corpus is included; no GPU or Hugging Face account is required. Checks cover causal masking, full-sequence/cached inference equivalence, cache bounds, gradient reach, and checkpoint save/load. Running the sweep writes checkpoints and metrics under `results/`.

See [DESIGN.md](DESIGN.md) for data examples, training objectives and budgets, optimizer settings, cache semantics, reproducibility details, and the RTX 3090 follow-up. [RESULTS.md](RESULTS.md) contains the full results table and controls; [results/runs.json](results/runs.json) records raw metrics, versions, and initialization/data hashes.
