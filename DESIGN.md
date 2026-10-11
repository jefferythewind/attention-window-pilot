# V1 hello world: experimental design

Follow-up to Timothy DeLise and Seth Cromelin,
[*Thinking Outside the Box: Retention and Transmission of Information in
Sliding-Window KV Inference*](https://arxiv.org/abs/2609.34049) (arXiv:2609.34049).

## One experiment, five conditions

The question is whether a transformer trained with local attention over connected
sequences learns useful information retention beyond its immediate attention window.
For sequence length N, full attention over N establishes the baseline; sliding
conditions reduce the span while preserving the complete computation graph.
An independent-chunk condition at the same shorter span tests disconnected training.

V1 sets N=128 with eight layers:

| Condition | Training inputs / attention | Native inference |
|---|---|---|
| full-128 | Complete 128-token sequences; full causal attention | Full attention |
| sliding-32 | Complete sequences; causal window 32 at every layer | Rolling KV, window 32 |
| sliding-16 | Complete sequences; causal window 16 at every layer | Rolling KV, window 16 |
| sliding-8 | Complete sequences; causal window 8 at every layer | Rolling KV, window 8 |
| chunk-32 | Four disconnected 32-token blocks per original sequence; full causal attention within blocks | Reset at aligned block boundaries |

W includes the current token. Smaller sliding windows do not shorten training inputs
or detach intermediate representations. Chunk training has no information path
between blocks, while retaining every original input and prediction target.
Only W=32 receives the chunk-trained control; V1 has no chunk-16/8 runs.

## Model and paired training

Eight pre-normalized decoder blocks, width 48, four heads (head dimension 12), GELU
MLP expansion 4, learned absolute position capacity 512, tied input/output embeddings,
no dropout. Approximately 254K parameters for text, 252K for retrieval. PyTorch CPU,
float32, deterministic operations, one thread per worker. Three independent worker
processes execute jobs; worker count is an execution setting, not a model change.
The model uses Hugging Face `PreTrainedModel` and `PretrainedConfig`; it is not
AutoModel registered and does not implement a complete generation interface.

Six seeds were selected before training: **515843, 363550, 519935, 354526, 292582,
57613**, drawn with `random.Random(20261011).sample(range(100,1000000),6)`.
Within each task and seed, all five conditions share exact initial tensors, ordered
original sequences, labels, processed-token budget, target count, and optimizer
updates. Input, label and initialization hashes verify pairing. Each seed changes
both initialization and training sampling (`seed+1000`); initializer quality is not
isolated. Every seed is retained. Earlier two-seed pilots are not additional V1 seeds.

| Setting | Value |
|---|---|
| Original batch | 16 × 128-token sequences |
| Text training | 200 updates; 409,600 processed tokens; 406,400 targets |
| Retrieval training | 400 updates; 819,200 processed tokens; 6,400 answer targets |
| Optimizer | AdamW, betas (0.9, 0.95) |
| Weight decay | 0.1 for matrices, zero for biases/norm vectors |
| Learning rate | Peak 0.002; 20-update warmup; cosine decay to 10% |
| Gradient clipping | Global norm 1.0, once per original batch |
| Validation sampling | Fixed seed 2026; identical examples across conditions/seeds |

The tasks train separate models: 6 seeds × 5 conditions × 2 tasks = **60 runs**.
There is no condition-specific tuning. [v1_config.json](v1_config.json) records the
configuration selected before training.

## Data and evaluation

**Text prediction:** Tiny Shakespeare from Karpathy's char-rnn repository, included
as `shakespeare.txt`; encoding is the sorted corpus character set. A contiguous
first-90% / last-10% split supplies training / held-out text. Evaluate 512 fixed
128-character held-out sequences using next-character NLL in nats and perplexity.
This measures held-out prediction, rather than explicitly testing memorization;
literary phrases can repeat and it is not a general capability benchmark. Corpus
hash, split and tokenizer are saved.

**Retrieval:** random filler, STORE and one uniformly random value among eight
alternatives, then QUERY and the answer. Values, filler (16 alternatives), and
markers use disjoint token IDs. Source-to-query distances 8, 24, 48, 80 are mixed
uniformly in expectation during training. Only the answer target is supervised.
Evaluation uses 256 fixed fresh examples per distance; chance among values is
12.5%. Predictions can also select non-value tokens.

A distance-80 example abbreviates filler:

```text
[45 filler tokens] STORE 2 [79 filler tokens] QUERY 2
```

There are 128 tokens: query index 126 predicts answer index 127. Distance is from
the value (index 46) to QUERY. The answer cannot influence query logits under
causal attention. This is a single-value learnability probe, without competing
facts or natural-language reasoning, evaluated on separately trained probe models.
It does not establish retrieval ability in the text models.

Report accuracy and answer NLL, plus a source-value counterfactual that changes the
correct answer. Report every seed, mean ± sample standard deviation, and within-seed
differences. There is no predeclared general-capability equivalence margin; do not
infer equivalence from nonsignificance or exclude seeds that struggle locally.

## Chunk-trained control and inference ablations

An inference ablation recomputes the final W-token window from raw tokens, discarding
retained states. It tests the inference contribution of retained representations.
These metrics accompany rolling evaluation.

The W=32 training control trains chunk-32 on disconnected blocks of the original data. Absolute
positions are preserved: 0..31, 32..63, 64..95, 96..127. Resetting position indices
would introduce another difference. Blocks are batched independently; their
shared-parameter gradients are combined into one update per original batch. A
non-final block's last output predicts the next block's first token as a label;
its representation is not supplied to the preceding block. The shifted loss retains
all 127 text targets, including boundaries. Retrieval still supervises one answer.
Maximum span is matched, but average context and attention-pair counts differ after
resets; compute is not matched.

| W=32 training | Rolling inference | Independent-block inference |
|---|---|---|
| Connected sliding | Primary sliding result | Evaluation of the same checkpoint |
| Independent chunks | Transfer evaluation of the same checkpoint | Primary chunk-control result |

Native chunks cannot retrieve a random value in another block: changing it must
leave query logits exactly unchanged. Comparing native systems measures practical
performance with the same maximum span. Comparing both with rolling inference
holds inference fixed and measures training adaptation. Rolling transfer is outside
the chunk model's training regime: with answer-only supervision, earlier chunks and
their learned positions receive no gradient. RoPE/fixed-position studies would
help separate positional learning from learning to relay. Connected training's
success is not proof that such training is necessary for all models.

## Receptive field, memory and costs

For L layers and causal window W, maximum source distance is L(W−1). Horizons are
248, 120, 56 for W=32,16,8; distance 80 is structurally unreachable for W=8.
Reachability does not guarantee learning. The cache is not unlimited recurrent memory.

Rolling inference retains W−1 previous keys/values per layer, preserving states
and absolute positions. Incoming chunks are computed before eviction; cached logits
are checked against complete masked evaluation. Float32 KV tensors per sequence
occupy `2 × L × (W−1) × width × 4` bytes: 381 / 93 / 45 / 21 KiB for W=128/32/16/8.
W=32 uses **24.4% of full KV storage**, approximately one quarter. This excludes
weights, metadata, activations, temporary prefill and allocator overhead; it is not
a claim about quarter total or training memory.

Training uses dense boolean masks with PyTorch SDPA and establishes no sparse
compute savings. Timed training includes batch creation, hashing and optimizer
work; it excludes initialization, evaluation and checkpoint writes. Worker durations
overlap; end-to-end wall time is reported separately. V1 does not establish results
for 25K versus 100K context.

## Reproduce

Install CPU dependencies in README.md, then:

```bash
.venv/bin/python checks.py
.venv/bin/python checks_chunk.py
.venv/bin/python run_v1.py
.venv/bin/python summarize_v1.py
```

The runner uses [v1_config.json](v1_config.json), resumes completed checkpointed jobs
when configuration/data match, and saves metrics/checkpoints in `results-v1/`.
Original pilot checkpoints are not required. Raw JSON records package versions,
configuration, source/data hashes, pairing hashes, budgets and loss traces. Checkpoint
binaries are excluded from Git; rerunning creates them locally.
`WindowLM.from_pretrained(path)` loads a checkpoint. Direct long-sequence calls use
rolling attention; native chunks use `chunk_logits(model, input_ids, 32)`.

Checks cover causality, cache equivalence/bounds, gradient reach, multi-layer paths,
checkpoint save/load, independent-block equivalence/invariance, full-block loss and
gradient equality, and every prediction target. All seed pairs are checked for
input/label/initialization and budget equality as jobs finish.

[RESULTS-V1.md](RESULTS-V1.md) reports V1. Earlier protocols and scaling ideas are in
[archive/PILOT-HISTORY.md](archive/PILOT-HISTORY.md), with historical reports
[RESULTS.md](RESULTS.md), [RESULTS-2X.md](RESULTS-2X.md), [RESULTS-CHUNK.md](RESULTS-CHUNK.md).
