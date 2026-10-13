# Attention-window pilot — V1 hello world

A six-seed CPU experiment testing whether local attention trained over connected sequences supports retrieval beyond its window. A follow-up to Timothy DeLise and Seth Cromelin’s [Thinking Outside the Box: Retention and Transmission of Information in Sliding-Window KV Inference](https://arxiv.org/abs/2609.34049).

One setting: **128-token sequences, eight layers, width 48, four heads**. Full attention over the whole sequence is the baseline. Sliding models train on the same complete sequences with windows 32, 16, or 8. The W=32 chunk control instead trains on four independent blocks of the same data. Within each seed, initialization, batches, targets, and training budgets match.

## Results

Text prediction and synthetic retrieval use separate models. Values are mean ± sample standard deviation across six new seeds, using each condition’s native inference rule (rolling attention or independent chunks). Retrieval asks for a stored value 80 tokens before the query; chance is 12.5%.

| Condition | Held-out text NLL ↓ | Retrieval at distance 80 ↑ |
|---|---:|---:|
| full-128 | 2.5396 ± 0.0075 | 88.7 ± 18.2% |
| sliding-32 | 2.5069 ± 0.0075 | 92.0 ± 19.6% |
| sliding-16 | 2.4752 ± 0.0043 | 30.4 ± 28.1% |
| sliding-8 | 2.4366 ± 0.0030 | 12.5 ± 3.3% |
| chunk-32 | 2.4554 ± 0.0074 | 12.8 ± 3.2% |

Sliding-32 reaches 100% far-distance retrieval in 5/6 seeds; full attention does so in 4/6. Giving chunk-trained models the same rolling inference yields 12.8 ± 3.2%. This supports connected training on this task; it does not establish general capability equivalence. Seeds vary both initialization and training batches, and none are excluded. Learned positional coverage also differs under answer-only chunk supervision.

W=32 retains **93 KiB versus 381 KiB** of KV tensors for the full baseline (24.4%, approximately one quarter). This measures retained KV memory, not total model memory or training cost. Eight layers impose a finite information horizon; the implementation uses dense attention masks.

## Reproduce

The corpus is included. No GPU or account is required. Use a new output directory to rerun all 60 models; published metrics exclude checkpoints. The summary below reads the preserved V1 metrics.

```bash
git clone https://github.com/jefferythewind/attention-window-pilot.git
cd attention-window-pilot
python -m venv .venv
source .venv/bin/activate
pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.57.6 numpy==2.5.4
python checks.py
python checks_chunk.py
python run_v1.py --output-dir results-v1-reproduction
python summarize_v1.py
```

[Design and reproducibility](DESIGN.md) · [Full results and every seed](RESULTS-V1.md) · [Configuration](v1_config.json) · [Raw metrics](results-v1/runs.json) · [Earlier pilots](archive/PILOT-HISTORY.md)

Additional V1 measurements: [learning curves](figures/v1-learning-curves.png) and [training time/memory](RESULTS-TRAINING-COST.md).

**Same-layer recurrent slot:** the default-off feature adds no parameters and retains one completed output per layer. In 36 matched new runs, recurrent-8 lowered text loss to 2.3035 ± 0.0056, but distance-80 recall remained near chance (14.3% ± 3.1%). The verified beyond-horizon path did not learn reliable retrieval at this budget. See [all seeds, timings and state sizes](RESULTS-RECURRENT.md) and [reproduction protocol](RECURRENT-DESIGN.md).

A [single-seed W=32 test on 320-token sequences](RESULTS-W32-HORIZON.md) puts the source 280 tokens away, beyond the sliding horizon of 248. Recurrent recall is 27.0% versus sliding’s 14.8%, with source-sensitive predictions and strong local recall: partial learned relay, not reliable exact retrieval.

A [diagonal one-versus-two-slot comparison](RESULTS-LATERAL-SLOTS.md) scores 36.7% versus 28.1% at distance280, with changed-source accuracy39.5% versus24.2%. This is improved partial relay in one paired seed. [Architecture and correctness checks](DIAGONAL-LATERAL-DESIGN.md).
