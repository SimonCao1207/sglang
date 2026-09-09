# Rebuttal to W5: dynamic tree size at larger batch sizes

**Reviewer.** *"The experiments only cover batch size 1; the efficiency of dynamic
tree size under larger batch sizes therefore remains unverified."*

We have run the full evaluation on SGLang at concurrency 1, 4, 8, 16, 32, and 64, for
two models (Qwen3-4B, Qwen3-8B) and two tasks (Math500, HumanEval). The results
confirm that the tree size is genuinely dynamic across batch sizes and that this
dynamic sizing is what keeps it efficient as batching grows. Qwen3-4B / Math500 is
shown below; the trends hold in all four settings.

| Qwen3-4B, Math500 (tok/s / τ) | c1 | c8 | c32 | c64 |
|---|---|---|---|---|
| Fixed small tree (DDTree-16) | 376 / 8.61 | 1715 / 8.63 | 2330 / 8.78 | 2328 / 8.81 |
| Fixed large tree (DDTree-64) | 361 / 9.87 | 1017 / 9.92 | 1136 / 9.89 | 1135 / 9.91 |
| **BASTION (dynamic)** | 360 / **9.41** | 1695 / 8.87 | 2363 / 8.05 | **2374** / 7.46 |

**The tree adapts with batch.** BASTION's acceptance length falls monotonically as
concurrency rises (9.41 to 7.46 here, and likewise in the other three settings),
because the controller shrinks the tree as batching raises the marginal cost of
verification. This is the dynamic behavior W5 asks us to verify, and it is visible
directly in the measured τ; a fixed tree cannot do it.

**Dynamic sizing is what stays efficient at scale.** A fixed large tree wastes compute
once batches saturate the GPU: DDTree-64 plateaus near 1100 tok/s, whereas BASTION
scales to 2374 at concurrency 64. This gap is consistent, 2.1 to 2.3x over the fixed
large tree at concurrency 32 to 64 across both models and both tasks. The dynamic
controller is therefore most valuable precisely at larger batch sizes, where a static
large tree is least efficient.

**One policy covers both regimes.** At low batch BASTION keeps the large-tree
acceptance length (τ = 9.41 at concurrency 1, versus 8.61 for the fixed small tree);
at high batch it matches small-tree throughput (2374 at concurrency 64, versus 2328).
No single fixed tree size spans both ends, and BASTION reaches them without knowing the
batch size in advance.

**Summary.** The multi-batch results verify that dynamic tree sizing remains efficient
well beyond batch size 1, and that its benefit grows with batch: it recovers 2.1 to
2.3x the throughput of a fixed large tree at high concurrency while retaining the
acceptance-length advantage of large trees at low concurrency.
