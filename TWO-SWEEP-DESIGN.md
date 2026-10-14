# Two-sweep top-layer recurrent attention: separate variant

The intended all-layer architecture is now the [zigzag mode](ZIGZAG-DESIGN.md), with corrected rows feeding subsequent ordinary blocks. The top-only trial here is retained separately.

This is a separate architecture, enabled by `two_sweep_recurrent=True` with `lateral_window=32`. It is mutually exclusive with the older interleaved `recurrent_attention_slot` mode. Both default to off. No parameters are added; initialization and parameter keys match the ordinary model.

The first sweep calls the existing ordinary `Block.forward` at every layer, with all sequence positions parallel under the original causal sliding mask. It records completed outputs U after attention and MLP residuals. The second sweep reads only the top row. At each position it projects the query and current self K/V from `ln1(U_top[t])`, attends jointly to that self slot and at most W−1 historical K/V pairs, and adds the existing output projection to U. There is no second MLP, gate, positional embedding or temporal additive residual. After finishing R[t], it projects `ln1(R[t])` into history K/V exactly once. Later positions read these completed corrected states through their cached projections, rather than U-only history. Missing history is absent at startup.

Logits are `head(norm(R_top))`. There are no lower recurrent rows, auxiliary losses or subsequent vertical integration. The ordinary lower backbone still receives gradients through U_top. This intentionally implements the minimal top-only trial; optional all-row diagnostics are not provided. Consequently there is no lateral cross-layer attention or recurrent-row interaction to hide in the implementation.

Training retains autograd through both sweeps and the complete sequence. No state is detached. Training uses full sequences and rejects streaming caches, as in the original model. Streaming evaluation carries a dictionary with separate `ordinary` per-layer K/V/position tuples, one top-layer `recurrent` K/V/position tuple, and an absolute `next_position` cursor. Each cache retains at most its window minus one prior positions. Returned streaming tensors are compact copies, while internal training histories retain their gradient connections. New independent examples start with no cache; chunks of one example continue both caches and their absolute cursor. No recurrent state overwrites the ordinary grid or cache.

## Acceptance checks

`checks_two_sweep.py` verifies exact flag-off outputs and gradients against the previous published model; first-sweep outputs and caches against the ordinary backbone; an independent explicit joint-softmax R-history reference including parameter gradients; unchanged parameter keys; full/token/chunk agreement; window1 and larger windows; startup, reset, independent-batch and absolute-position behavior; future/answer-token causality; and save/load. It explicitly checks the top-corrected readout and lower-backbone gradients. A controlled intervention changes a completed R, reprojects it, and affects the next R. A hook-based gradient check confirms that the actual cached history depends on the completed R. A short controlled source test beyond L(W−1) gives exact zero forward influence/gradient in the ordinary model and nonzero values in the new architecture.

## Trial and cost measurements

`two_sweep_config.json` preserves the 320-token protocol: eight layers, width48, four heads, ordinary and lateral windows32, batch16, seed515843, 400 retrieval updates, distances8/24/48/280, the existing AdamW schedule and two CPU threads. Only the new model is trained. Saved ordinary and interleaved-recurrent controls are reused. Parameter initialization, ordered inputs, labels and update/token/target budgets must match their hashes. Distance280 exceeds the ordinary horizon248. A single nonrecurrent lateral pass using only U history would extend this finite reach to279, so distance280 also lies beyond that shortcut; reading completed R history is essential to the unbounded path. One seed is exploratory; graph connectivity does not establish learned or arbitrary-distance retention.

`run_two_sweep.py` records the training source commit and hashes, full training duration and peak worker-process RSS before evaluation. `measure_two_sweep.py` loads each saved checkpoint in a fresh process, without optimizer updates or changing parameters. It reports a fixed batch16 × 320 forward/backward pass, peak process RSS during that pass, unique storage saved by autograd, and the separately counted completed first-sweep grid for the new model. The grid count is temporary activation volume, not total training memory. Process RSS includes Python/PyTorch, weights, inputs and allocator storage; saved-tensor storage also includes weights/inputs and is not a peak-memory measurement.

Prefill and cached decoding are timed separately with batch1 and batch16, two warmups, five full 319-token prefill timings, and twenty one-token timings from the same frozen prefix cache. All use two CPU threads. Ordinary/recurrent KV bytes, metadata, logical retained bytes and actual backing-storage bytes are measured separately. Historical full-training times remain historical measurements; the checkpoint cost profiles provide common measurement conditions without rerunning training controls. No speedup or memory reduction is assumed.

```bash
.venv/bin/python checks_two_sweep.py
.venv/bin/python run_two_sweep.py --output-dir results-two-sweep-reproduction
.venv/bin/python measure_two_sweep.py
.venv/bin/python summarize_two_sweep.py
.venv/bin/python plot_two_sweep.py
```

The cost profiler requires local checkpoint binaries for the preserved controls; metrics are published without those binaries. A fresh clone can reproduce the new trial and its costs by passing its saved checkpoint to `measure_two_sweep.py --checkpoint PATH`. Comparison metrics remain available in the preserved raw results. Use a new output directory to retrain; report and plot read the published default directory.
