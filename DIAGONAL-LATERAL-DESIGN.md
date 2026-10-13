# Two lateral completed-output slots and diagonal execution

Enable recurrence with `recurrent_attention_slot=True`. Set `recurrent_attention_slots=2` for completed outputs at t−1 and t−2 from that same layer. Default count1 preserves the original one-slot model. Both extra K/V pairs reuse that block’s ln1 and existing K/V projection weights and biases, join the ordinary attention softmax, and receive no additional positional embedding. No trainable parameters are added. W still counts only ordinary positions, including the current token. Token0 has no extra slot; token1 has one; token2 onward has two. History is stored oldest first.

`recurrent_execution='sequential'` remains the default reference. `recurrent_execution='diagonal'` executes wavefronts indexed by t+layer. Each node depends on the preceding layer at the current token (previous diagonal), its completed output at t−1 (previous diagonal), t−2 (two diagonals earlier), and already computed ordinary K/V inputs. Nodes on the same diagonal are independent. A T-token, L-layer forward pass has T+L−1 waves with at most L active nodes per wave, rather than T×L separate token/layer steps. Waves still execute in dependency order.

The implementation batches active nodes across layers, using their own stacked projection and normalization parameters. Batched matrix multiplications and one SDPA operation compute a wave; this is tensor-level parallelism, not a Python thread per token. Full waves reuse batched history/cache tensors. Startup and drain waves have fewer active nodes. Missing ordinary and recurrent positions are fully masked. State updates become visible only to the next wave. All stacking, slicing and copying remain differentiable; no state is detached and no truncated backpropagation is introduced.

The cache format remains `(ordinary_K, ordinary_V, positions, completed_history, next_position)`. Completed history is batch × min(tokens_seen,slots) × width. Ordinary K/V contain at most W−1 entries. Returned diagonal caches are compact copies so one layer’s views do not retain unrelated diagonal lanes. Both state types reset when no prior state is supplied and persist across chunks of the same sequence. Single-token streaming has little diagonal parallelism; full-sequence training/prefill supplies the independent nodes.

## Checks

```bash
.venv/bin/python checks_recurrent.py
.venv/bin/python checks_diagonal.py
```

The checks compare the default one-slot outputs and parameter gradients exactly with the prior published implementation at commit24f1619. An independent explicit joint-softmax reference verifies one/two-slot forward values and gradients against diagonal execution. Tests cover windows1/3/16, sequence lengths1/2/9, startup missing slots, full/token/chunk inference, no future/answer leakage, independent-example resets, cache/history bounds and checkpoint save/load. A controlled W=1 test isolates the t−2 state: changing only it affects the output and gives a nonzero gradient. An end-to-end controlled source test also verifies nonzero diagonal forward influence and gradient at distance4 beyond the baseline horizon2, versus exact zeros with recurrence disabled. No extra parameter keys are introduced.

Diagonal versus sequential floating-point reductions can differ slightly even when they implement the same mathematics; equivalence is checked at tight float64 tolerances. New training comparisons therefore use diagonal execution for both slot counts, rather than treating an old sequential checkpoint as a perfectly matched numerical-training control.

## CPU benchmark and experiment

[diagonal_benchmark.json](diagonal_benchmark.json) records a short CPU thread-count benchmark: one warmup and three timed forward/backward/clip/AdamW updates with batch16 × 320 tokens and two slots. Two and four threads perform similarly in that small measurement; training uses two threads for both conditions. Reproduce a fresh measurement with `.venv/bin/python benchmark_diagonal.py`; it writes a separate benchmark file. This is a local benchmark, not a general CPU/GPU speedup claim.

[lateral_slots_config.json](lateral_slots_config.json) defines one matched seed515843, eight layers, width48, heads4, W32, max positions512, 320-token complete sequences, and 400 retrieval updates. Training distances8/24/48/280 are mixed uniformly in expectation. Distance280 exceeds the nonrecurrent horizon248. Initialization, batches, labels and optimizer budgets are paired; wall-clock time is not matched. No text models are trained in this comparison.

```bash
.venv/bin/python run_lateral_slots.py --output-dir results-lateral-slots-reproduction
.venv/bin/python summarize_lateral_slots.py
```

Use a new output directory to retrain. The summary reads the preserved default results-lateral-slots/ measurements. No previous checkpoint binaries are required. Historical sequential one-slot and sliding results at280 are contextual references; the primary comparison is the freshly trained diagonal pair. One seed cannot establish across-seed reliability.

## Three-slot extension

`recurrent_attention_slots=3` adds the completed same-layer output at t−3 to the same softmax. The existing diagonal implementation supports the third history entry; its dependency comes from three diagonals earlier. [Three-slot results and reproduction](RESULTS-THREE-SLOTS.md) reuse the preserved controls and train only the new condition.
