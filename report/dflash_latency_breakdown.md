# DFlash Chain vs Tree: Latency Breakdown

Per-step latency analysis of why a fixed best_first tree helps at low batch but
loses its edge as batch grows, and why larger fixed trees collapse at scale.

## Setup

- Target Qwen3-8B, draft z-lab/Qwen3-8B-DFlash-b16, block_size 16, single A6000.
- Chain baseline: linear 16-token block (`run_dflash.sh`).
- Tree: best_first, 16 nodes (`run_dflash_best_first_tree.sh`, `BEST_FIRST_TOKENS=16`).
- Math500, greedy decoding, concurrency sweep 1/4/8/16/32/64.
- Phase timings come from `SGLANG_DFLASH_LATENCY_BREAKDOWN=1` (mean over 100-step
  windows). Each phase is CUDA-synced, so absolute ms are inflated (roughly 1.2 to
  1.3x); the relative split across phases is exact. Use the client throughput below
  for absolute numbers.

## Measured throughput (client, tok/s | accept length)

| concurrency | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| Chain (16) | 220 / 8.07 | 706 / 7.91 | 1177 / 8.02 | 1380 / 7.97 | 1572 / 8.03 | 1585 / 8.16 |
| Tree (16) | 236 / 8.53 | 765 / 8.64 | 1172 / 8.48 | 1451 / 8.52 | 1669 / 8.63 | 1656 / 8.68 |

The tree carries a persistent accept-length advantage (about 8.5 vs 8.0). At a
16-node budget that is enough to win or tie at every concurrency, because the step
latency stays close to the chain. The advantage narrows around concurrency 8 to 16,
which is where the per-step costs below start to diverge.

## Latency breakdown (ms/step)

Top-level phases are disjoint and sum to the step. `verify_fwd` is the target
forward pass; `prepare` is draft forward plus tree construction; `v_commit` is the
verify kernel plus KV commit.

### Chain (16, linear)

| batch | prepare | verify_fwd | v_commit | draft_upd | step | AL |
|---|---|---|---|---|---|---|
| 1 | 6.26 | 27.48 | 0.66 | 1.01 | 35.41 | 7.88 |
| 4 | 7.10 | 34.80 | 0.76 | 1.19 | 43.85 | 7.74 |
| 8 | 8.56 | 45.35 | 0.94 | 1.24 | 56.09 | 7.86 |
| 16 | 12.76 | 73.67 | 1.31 | 1.33 | 89.07 | 7.82 |
| ~36 | 16.94 | 95.90 | 1.60 | 1.39 | 115.84 | 7.71 |

### Tree best_first (16)

The last three columns are the tree-only extra work inside `prepare`.

| batch | prepare | verify_fwd | v_commit | draft_upd | step | AL | topk | build | mask |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.70 | 26.78 | 0.57 | 1.01 | 35.05 | 8.38 | 2.07 | 0.18 | 0.21 |
| 4 | 8.43 | 34.75 | 0.82 | 1.21 | 45.21 | 8.51 | 2.73 | 0.35 | 0.31 |
| 8 | 9.75 | 39.84 | 1.03 | 1.22 | 51.84 | 8.39 | 3.34 | 0.50 | 0.39 |
| 16 | 14.30 | 62.95 | 1.47 | 1.30 | 80.02 | 8.33 | 4.89 | 0.80 | 0.56 |
| ~36 | 25.29 | 110.33 | 2.47 | 1.48 | 139.57 | 8.40 | 9.53 | 1.53 | 0.94 |

## What the numbers show

**1. The target forward dominates the step (70 to 80%).** At batch 1 it is about
27 ms of a 35 ms step for both methods, and it is nearly identical (even slightly
lower for the tree). The tree-only CPU work (topk + build + mask, about 2.4 ms at
batch 1) is hidden because the chain spends a comparable amount on its own greedy
draft sampling. Same step latency, higher accept length, so the tree wins at low
batch.

**2. The crossover at high batch comes from two costs that scale worse for the
tree.** Decomposing the step gap at batch ~36:

| phase | chain | tree | tree - chain |
|---|---|---|---|
| prepare | 16.94 | 25.29 | +8.35 |
| verify_fwd | 95.90 | 110.33 | +14.43 |
| v_commit | 1.60 | 2.47 | +0.87 |
| draft_upd | 1.39 | 1.48 | +0.09 |
| step | 115.84 | 139.57 | +23.73 |
| accept length | 7.71 | 8.40 | +0.69 |

- `verify_fwd` grows faster for the tree (+14.4 ms). The tree needs a dense custom
  attention mask to encode ancestor-only access, which is more expensive than the
  chain's implicit causal mask, and the cost grows with batch.
- `prepare` grows faster (+8.4 ms), almost entirely from `topk`. The tree needs
  top-k over the full vocabulary at every depth to build the tree; the chain only
  needs an argmax. At batch 36 that is 9.5 ms versus about 2 ms.

The accept-length edge of 0.69 tokens cannot pay for a 23.7 ms step increase, so
the chain catches up as batch grows.

## Implication for larger trees and the adaptive controller

The two scaling costs both grow with the tree budget as well as with batch:
`verify_fwd` runs over N tree nodes, and `topk` uses k = N candidates per depth. A
Fixed(32) or Fixed(64) tree therefore pays a much larger verify and topk penalty at
high batch, which is why their throughput collapses at scale even though their
accept length is higher (Fixed(64) reaches accept length near 9.8 but drops to
about 740 tok/s at concurrency 64).

This is the motivation for the adaptive shared-budget controller: it shrinks the
per-step budget N* as batch grows, backing off the tree exactly when `verify_fwd`
and `topk` begin to dominate, so it keeps the large-tree accept length at low batch
and the small-tree throughput at high batch.
