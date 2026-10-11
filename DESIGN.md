# Experimental design and reproducibility

Follow-up to Timothy DeLise and Seth Cromelin,
[*Thinking Outside the Box: Retention and Transmission of Information in
Sliding-Window KV Inference*](https://arxiv.org/abs/2609.34049)
(arXiv:2609.34049, 2026).
The paper's abstract motivates testing whether training with sliding windows improves
information relay. This pilot trains models from scratch; it does not fine-tune the
paper's pretrained models.

See [README.md](README.md) for the headline and [RESULTS.md](RESULTS.md) for the recorded results.

## How to interpret the two tasks

This pilot contains **two experiments with separately trained models**, not two
evaluations of the same model. Each task has its own full-attention baseline and
window-restricted conditions. Matching initialization, data and token budgets
applies within a task and seed; the two tasks have different data and objectives.

| Task | Training data and objective | Held-out evaluation | What it tests |
|---|---|---|---|
| Text prediction | Shakespeare characters; predict each next character | Next-character loss on the held-out corpus portion | Learning language patterns that generalize to unseen sequences |
| Controlled retrieval | Random filler, one stored value, then a query; predict only the answer | Accuracy on newly generated examples at fixed distances | Learning to transmit a specific value across positions |

The text task is not explicitly a memorization test. Held-out prediction measures
generalization, although a small literary corpus can contain repeated phrases and
is not a comprehensive measure of language ability. Retrieval is deliberately
simple: one stored value, no competing facts, and no natural-language reasoning.

The current headline is: **A fourfold attention-window reduction preserved simple
controlled retrieval; separately trained text models showed no held-out loss
penalty after the short matched training budget.** We have not shown that one
text-trained model achieves both results, or that the text-loss advantage persists
after longer training. Smaller windows still succeed on some shorter-distance
retrieval cases; their degradation depends on distance.

## Concrete data examples

These are actual first-batch examples reconstructed from the recorded random seeds.
The text examples each contain 64 characters, including spaces and newlines.

Training text (seed 17):

```text
d thy close curtain, love-performing night,
That runaway's eyes 
```

Held-out text:

```text
 you; I will not adventure
my discretion so weakly. Will you lau
```

Every position except the final character supplies an input for a next-character
prediction target. The held-out example comes from the final 10% of the corpus.

The retrieval examples below abbreviate random filler; STORE, QUERY, and each
value are single tokens, not their displayed text spellings:

```text
Training: [49 filler tokens] STORE 3 [11 filler tokens] QUERY 3
Held-out: [21 filler tokens] STORE 2 [39 filler tokens] QUERY 2
```

Both training sequences contain 64 tokens, including the answer. Only the answer
prediction contributes to retrieval training loss. At evaluation, the model sees
the 63-token prefix ending at QUERY and predicts the withheld answer. The example
source-to-query distances are 12 and 40, respectively. Filler tokens are sampled
independently from 16 alternatives, disjoint from the eight possible value tokens
and the two marker tokens. Held-out probe examples use a separate fixed generator;
the answer is uniformly random and can only be determined from the stored value.

## Run

```bash
cd attention-window-pilot
python -m venv .venv
.venv/bin/pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers==4.57.6 numpy==2.5.4
.venv/bin/python checks.py
.venv/bin/python run.py
.venv/bin/python summarize.py
```

The included `shakespeare.txt` is the public Tiny Shakespeare character corpus,
downloaded from Karpathy's char-rnn repository. Its hash and split boundary are in
`results/runs.json`. `requirements.lock` captures all installed dependency versions.
No GPU or Hugging Face account is required. Checkpoints use Hugging Face
`PreTrainedModel` / `PretrainedConfig` and `CausalLMOutputWithPast`; load with
`WindowLM.from_pretrained(path)`. This is a minimal custom HF model, not an
AutoModel-registered architecture or a complete GenerationMixin integration.
Character encoding is saved in `results/tokenizer.json`.

## Matched experiment

Four pre-normalized decoder blocks, width 48, four heads, GELU MLP expansion 4,
learned absolute position embeddings, tied input/output embeddings, no dropout.
Every input has 64 tokens. W=64 is full causal attention at this sequence length;
other windows are 16, 8, and 4. W includes the current token.

Two paired seeds (17, 29). Within each seed and task, every window starts from
the exact same tensors and receives the same ordered batches. Initialization and
batch hashes are recorded. Optimizer: AdamW, betas (0.9, 0.95), matrix weight decay
0.1, peak LR 0.002, warmup followed by cosine decay to 10%, gradient norm clipping
1.0, float32, deterministic PyTorch operations, four CPU threads. No condition
specific hyperparameter tuning. Full sequences retain their entire autograd graph;
there is no cache, detach, or truncated backpropagation during training.

1. **Character next-token LM:** 200 updates, batch 16, 204,800 processed tokens
   per run; 201,600 next-token targets. Contiguous 90/10 corpus split, 512 fixed
   held-out sequences. Report NLL in nats per character and character perplexity.
2. **Separate controlled recall probe:** 400 updates, batch 16, 409,600 processed
   tokens per run. Each sequence contains one STORE marker and uniformly random
   value from eight alternatives, random filler, then a QUERY marker and answer.
   Source-to-query distances 4, 12, 24, 40 are mixed equally in expectation.
   Only the answer next-token target is supervised (6,400 targets/run). This is
   deliberately a task-specific learnability experiment, not general LM training.
   Evaluation uses 256 fixed examples per distance, exact-match accuracy, answer
   NLL, rolling-cache equivalence, final-window recomputation, and a source-value
   counterfactual that changes the correct answer. Chance is 12.5% among values;
   predictions may also choose non-value tokens.

The two tasks train independent models. Probe results are not evidence that the
character LM acquired the same retrieval ability. These models are too small to
support meaningful standard capability benchmark claims.

## Structural limit and cache semantics

An L-layer feedforward decoder with uniform local window W has a maximum input
distance of L(W-1). For L=4, horizons are 252, 60, 28, 12 for the four conditions.
Relay beyond W but within this horizon is possible. Beyond the horizon, source
information has no path to the output, even with an indefinitely rolling cache.
The output at a query predicts the following answer: distances here are measured
to the query, not the answer token.

Cached inference retains W-1 previous keys and values at every layer. Absolute
positions continue increasing after eviction. Chunked prefill computes attention
against the retained prefix and all positions in the incoming chunk before
eviction, preserving full-sequence masked evaluation. No reset of positions or
recomputation of retained states occurs. `checks.py` checks token and chunk prefill
equivalence, causal masking, cache bounds, gradient horizons, propagation beyond
one direct window, and checkpoint save/load.

Steady-state float32 KV tensor bytes per sequence are 2*L*(W-1)*width*4:
96,768 / 23,040 / 10,752 / 4,608 for W=64 / 16 / 8 / 4. Metadata, activations,
allocator overhead, and transient prefill concatenations are excluded. The
63-token retrieval prompt fills the full model's 63-key retained-cache capacity.
`cache_kv_bytes` records actual resident KV tensors, with batch size supplied.

Training uses a dense boolean local mask with PyTorch SDPA. This changes the
attention semantics but does not establish sparse-attention compute savings.
Recorded training times exclude initialization, evaluation and checkpoint writes;
they include batch creation and batch hashing. No inference latency or total
process peak-memory claim is made by this pilot.

## Doubled depth and context experiment

The second CPU sweep doubles depth from 4 to 8 layers and sequence length from
64 to 128 tokens. To preserve relative scales, windows become 128, 32, 16, 8 and
retrieval distances become 8, 24, 48, 80. Width 48, four heads, learned position
capacity 512, seeds 17 and 29, batch size 16, optimizer, LR schedule, and update
counts (200 text / 400 retrieval) remain unchanged. All models are trained from
scratch, with identical initial tensors and ordered batches across windows within
each new task and seed. Data and initialization hashes are checked again.
The scaled sweep uses one CPU thread for all conditions after a timing check
found substantial threading overhead; the original used four. This runtime change
is recorded in the metrics and also limits cross-pilot timing comparisons.

Each text run now processes 409,600 tokens with 406,400 next-character targets;
each retrieval run processes 819,200 tokens with 6,400 answer targets. Retaining
the update counts doubles processed-token budgets relative to the first pilot.
This is a joint scaling experiment, not an isolated depth effect or a comparison
matched in processed tokens across pilots. Held-out text samples are regenerated
at the new length using the same seed, so their positions and available context
also differ from the first pilot.

The text evaluation still uses 512 sequences, and retrieval still uses 256 examples
per distance. Full-sequence training, cached-logit equivalence, final-window
recomputation and changed-source controls are retained. With eight layers the
structural horizons are 1,016, 248, 120 and 56 positions for W=128, 32, 16, 8.
Distance 80 is reachable for W=16, but structurally unreachable for W=8.
This test cannot establish effects at the maximum horizon for the larger windows.

Run from the repository root after installing the dependencies above:

```bash
.venv/bin/python checks.py
.venv/bin/python run.py --layers 8 --sequence-length 128 \
  --windows 128 32 16 8 --distances 8 24 48 80 --threads 1 --output-dir results-2x
.venv/bin/python summarize_scaled.py
```

The original command defaults are preserved. Scaled metrics and checkpoints are
saved under `results-2x/`, and the scaled report is [RESULTS-2X.md](RESULTS-2X.md).
Both sweeps' checkpoint binaries stay local; published JSON contains configurations,
metrics and reproducibility metadata. No standard capability benchmarks are added.

## Next stage on an RTX 3090

First profile a 10-30M parameter model (e.g. 8 layers, width 256, 8 heads, a fixed
8K subword tokenizer) at sequence length 1,024, batch 1-4 with gradient
accumulation, and FP16/BF16 where supported. Sweep W=1024,256,128,64,32 with a
fixed global token batch and training-token budget. Establish full attention's
held-out ability and positive-control retrieval before committing to the sweep.
Use at least three paired seeds for the final sweep, and increase tokens until
the baseline is meaningfully trained. Profile actual fit and throughput before
choosing a budget; no 24GB memory guarantee is implied.

Add a full-trained model evaluated with restricted caches to distinguish training
adaptation from inference-only restriction. Retain source counterfactuals and
final-window recomputation. Evaluate long-range tasks on the actual LM checkpoints
as well as separate diagnostic probes. Choose capability benchmarks appropriate
to baseline strength and include uncertainty, length/distance breakdowns, and
leakage controls. Predeclare practical equivalence margins (for example <=1%
relative held-out NLL increase, with a justified task-accuracy tolerance); two
seeds or a nonsignificant difference do not establish equivalence.

For long sequences use a verified native local-attention kernel, RoPE and consistent
absolute offsets. Confirm its logits and gradients against this dense reference.
Report training peak memory/throughput separately from decode-cache bytes, prefill
costs and decode latency. Keep full input sequences and backpropagation intact.
25K versus 100K windows require a separate scaling study; the CPU pilot tests
mechanics and learning trends, not that extrapolation.
