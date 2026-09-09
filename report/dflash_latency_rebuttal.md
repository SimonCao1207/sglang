# Latency breakdown: why a matched tree can lose to the chain at high batch

Per-step latency of a matched-budget best_first tree (16 nodes) versus the linear
chain (16 tokens), Qwen3-8B target, Math500, greedy. `verify_fwd` is the target
forward; `tree overhead` is the per-depth top-k over the vocabulary plus tree build
plus the dense tree mask (the work the chain replaces with a single argmax). Phase
timings are CUDA-synced, so absolute values are slightly inflated but the split and
the crossover are exact.

| batch | method | verify_fwd (ms) | tree overhead (ms) | step (ms) | accept len | norm. tput |
|---|---|---|---|---|---|---|
| 1  | Chain | 27.5 | 0.0  | 35.4  | 7.88 | 1.00 |
| 1  | Tree  | 26.8 | 2.5  | 35.1  | 8.38 | **1.07** |
| 16 | Chain | 73.7 | 0.0  | 89.1  | 7.82 | 1.00 |
| 16 | Tree  | 63.0 | 6.3  | 80.0  | 8.33 | **1.19** |
| 36 | Chain | 95.9 | 0.0  | 115.8 | 7.71 | 1.00 |
| 36 | Tree  | 110.3| 12.0 | 139.6 | 8.40 | **0.90** |

Normalized throughput is (accept length / step), tree relative to chain at the same
batch.

**Reading.** The target forward dominates the step (70 to 80%) and is nearly equal
for both methods at low batch, so the tree's higher accept length (about 8.4 vs 8.0)
turns directly into higher throughput (+7% at batch 1, +19% at batch 16). At high
batch the tree's step overtakes the chain by 24 ms: the dense tree-mask attention
adds 14 ms to `verify_fwd`, and full-vocabulary top-k adds 12 ms of overhead, while
the accept-length edge of 0.7 tokens cannot offset either. The tree therefore falls
about 10% behind at batch 36.

Both costs scale with the tree size as well as with batch (`verify_fwd` runs over N
nodes and top-k uses k = N candidates per depth), so a larger fixed tree crosses
over sooner and loses more, which is why a batch-adaptive budget is needed rather
than any single fixed tree size.
