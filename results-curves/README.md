# V1 learning curves

Compare full-128 and sliding-32 on the six original V1 seeds. The twelve text models are rerun with read-only evaluation at updates 0, 20, ..., 200; all final tensors, training batches, recorded original training losses and final evaluation metrics must exactly match V1.

Each checkpoint scores the same 512 training-set sequences and 512 held-out sequences. Training-set curves are fixed-set evaluation NLL, not the loss of the current optimizer batch. Every optimizer-batch loss is also retained in curves.json. Both evaluation curves use next-character targets, with no independent chunks. The training evaluation set uses a separate generator with seed 20261012; held-out sampling remains V1 seed 2026. Evaluation does not change training sampling or weights.

The horizontal axis is processed training tokens (200 updates × 16 × 128 = 409,600 tokens), since randomly sampled sequences do not define ordinary dataset epochs. Lines average six seeds. Bands show one sample standard deviation across seeds, rather than confidence intervals. The difference panel subtracts full attention from sliding within each seed.

This figure measures early prediction performance under the V1 budget. It does not establish converged superiority, a pure generalization mechanism, or training-memory savings. The established KV reduction applies at inference.

Reproduce from the experiment workspace after V1 checkpoints exist:

```bash
.venv/bin/pip install matplotlib==3.11.2
.venv/bin/python learning_curves.py
.venv/bin/python plot_learning_curves.py
```

Outputs: curves.json, and ../figures/v1-learning-curves.png, .pdf, .svg. Rerun checkpoints remain local; their final tensors duplicate V1 checkpoints.
