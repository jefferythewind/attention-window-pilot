# CPU pilot results

Two paired seeds; all initialization, sequence-order and token-budget checks passed.

**Headline:** A fourfold attention-window reduction preserved simple controlled retrieval;
separately trained text models showed no held-out loss penalty after short matched training.

Each table row summarizes two independently trained models per seed: a text model and a
retrieval model. We have not tested retrieval on the text-trained checkpoints.

| Window | Text NLL, mean (seed range) | Recall d=4 | d=12 | d=24 | d=40 | KV KiB/sequence* |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2.5255 (2.5244–2.5267) | 100.0% | 100.0% | 100.0% | 100.0% | 94.5 |
| 16 | 2.4937 (2.4933–2.4941) | 100.0% | 100.0% | 100.0% | 100.0% | 22.5 |
| 8 | 2.4673 (2.4662–2.4684) | 100.0% | 100.0% | 29.3% | 13.1% | 10.5 |
| 4 | 2.3949 (2.3921–2.3977) | 73.6% | 19.1% | 11.7% | 13.3% | 4.5 |

*Steady-state float32 KV tensor storage, excluding metadata and transient allocations.

Text and recall columns refer to separately trained models. Text NLL is nats per character.

The text task predicts the next character on held-out Shakespeare text; it is not explicitly
a memorization test. The retrieval task learns one random stored value amid random filler,
with only the answer supervised. See archive/PILOT-HISTORY.md for actual data examples.

The W=16 probe matches full attention on this simple retrieval task, including distances
24 and 40 beyond its direct window. Its final-window recomputation has no source token:
accuracy at distance 40 is 12.5%, 11.3%.
Changing the stored source value yields the corresponding changed answer at distance 40:
100.0%, 100.0%.

W=8 can propagate information across a 28-position structural horizon, but did not learn
reliable retrieval at distance 24 in this budget. Structural reachability is not sufficient
for learned retrieval. Distance 40 is outside its horizon and is a negative control.

Smaller windows do not fail universally: W=8 reaches 100% accuracy at distances 4 and 12,
and W=4 averages 73.6% at distance 4. Degradation depends on source-to-query distance.

Smaller windows achieved better early character prediction in this short training run.
This could reflect optimization or regularization at limited training, and does not
establish a converged capability advantage or statistical equivalence.

Total timed training: 76.8 CPU wall seconds.
These timings use dense masked attention; they are not sparse-kernel efficiency claims.
All trained checkpoints are retained under results/. See archive/PILOT-HISTORY.md for protocol and limits.
