# Intended all-layer full-window zigzag architecture

Enable `zigzag_recurrent=True` and `lateral_window=32`. The ordinary window is32 as well. This mode is exclusive with the separate top-only `two_sweep_recurrent` variant and the older interleaved recurrent-slot architecture. No trainable parameters are added.

For layer l, the ordinary block receives the previous layer’s corrected row R[l−1], or embeddings at the first layer. Its ordinary attention/residual/MLP operations are unchanged and run with sequence positions parallel, producing U[l]. A same-layer recurrent sweep then proceeds over t: the query and temporary self K/V come from LN1_l(U[l,t]); at most W−1 historical pairs come from completed R[l,j] of this layer. The existing output projection is added to U to finish R. LN1 and Q/K/V/output weights and biases are reused. There is no second MLP, gate or positional embedding. Each completed R is projected once into its layer’s rolling recurrent K/V cache.

The completed corrected row feeds the next ordinary block. Every recurrent row therefore contributes to the final corrected top-row readout `head(norm(R[L]))`, and receives gradients through the actual computed states. Lateral attention itself never reads another layer’s row. Vertical integration occurs only through the next ordinary block. This differs from computing an ordinary grid first and independently correcting its rows: zigzag alternates an ordinary and a lateral phase at each layer. It has L ordinary row-parallel phases and L temporally sequential lateral phases, rather than two global independent sweeps.

The full training graph remains connected through both operations and all sequence positions. State is never detached and backpropagation is not truncated. Streaming retains separate ordinary input-derived K/V caches and corrected-history K/V caches for every layer, plus an absolute-position cursor. Both retain at most W−1 prior pairs and reset when no past state is supplied. Ordinary cached K/Vs at layer l are derived from its corrected previous-layer inputs; recurrent K/Vs come only from its own completed R. Returned caches are compact allocations. The top-only implementation/results remain separately named and unchanged.

## Tests and interpretation

`checks_zigzag.py` compares an independent explicit same-row joint-softmax reference, including all parameter gradients, verifies that each ordinary input is the preceding corrected row, and checks a nonzero loss gradient to every corrected row. It tests full/token/chunk equivalence, future/answer causality, window/startup/reset/independent-example/absolute-position behavior, cache bounds and checkpoint round trips. A controlled completed-R intervention affects a later R. A controlled source at distance5 exceeds the ordinary horizon2 and the U-only two-local-attention horizon4 in the small two-layer W2 test; its forward influence and gradient are zero in the original backbone and nonzero in zigzag. The retained top-only and original flag-off paths also pass their regression checks.

The 320-token exploratory protocol is unchanged: eight layers, width48, four heads, both windows32, batch16, seed515843, 400 retrieval updates, distances8/24/48/280, existing optimizer and two CPU threads. Only the new zigzag condition is trained; older controls and the top-only variant are preserved. Initialization, inputs, labels and budgets are checked by hashes. Training source revision and file hashes are recorded.

Distance280 exceeds the original ordinary horizon8 × 31 =248. However, replacing corrected-history R with U-only history at every lateral layer would still yield two finite local attention operations per layer, with maximum reach2 × 8 × 31 =496. Consequently success at280 cannot by itself isolate recurrent-history relay from the added finite attention depth. Controlled tests verify the recurrent dependency path, while the task measures learning beyond the original backbone’s horizon. Reliable unbounded memory is not inferred; one seed has no between-seed standard deviation.

Temporary row volumes, autograd saved storage, peak process memory, retained ordinary/recurrent K/V, prefill and decoding are reported separately. Default zigzag execution does not retain Python lists of all U/R rows during inference; logical volumes recorded by diagnostic grid instrumentation are not simultaneous peak allocations. Full-sequence training autograd retains the tensors required for backward. Profiling saved checkpoints performs no optimizer updates or changes to model weights.

```bash
.venv/bin/python checks_zigzag.py
.venv/bin/python checks_two_sweep.py
.venv/bin/python run_zigzag.py --output-dir results-zigzag-reproduction
.venv/bin/python measure_two_sweep.py --checkpoint PATH_TO_ZIGZAG_CHECKPOINT
.venv/bin/python summarize_zigzag.py
.venv/bin/python plot_zigzag.py
```

Use a new output directory to retrain only zigzag. Saved comparison metrics are published without checkpoint binaries. Reports read the preserved default result directories. The checkpoint cost worker can profile the new checkpoint independently on a fresh clone.
