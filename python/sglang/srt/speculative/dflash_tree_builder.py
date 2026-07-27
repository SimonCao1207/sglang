"""DFlash tree-draft builder.

Standard DFlash drafts a linear chain of `block_size` tokens per verify step.
This module supports *tree* drafts: each verify step expands the draft into a
tree of candidate tokens and the target verifies the tree in a single forward.

DFlash's draft model is block-diffusion: one forward emits `block_size - 1`
parallel logit distributions, conditioned on mask tokens + bonus. Each depth's
distribution is **parent-independent** — picking a different parent at depth
`d-1` does not change depth `d`. So per-depth top-k is sufficient to
materialize any tree shape; no extra draft forwards are needed.

Currently exposes one builder: `build_best_first_tree` (heap-driven, ported
from spec-dllm `model/tree_builder.py::build_best_first_tree_from_draft_logits`).
Additional builders (e.g. width_pruned) can be added alongside it.
"""

from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

# Additive bonus applied to a child's edge log-prob when pushing it onto the
# heap. It biases best-first toward extending the current path instead of
# collecting shallow siblings, trading breadth for depth. 0.0 reproduces plain
# path-probability ordering. Borrowed from the DDTree builder (sgl-project PR
# #27509), which uses 0.2; tune with SGLANG_DFLASH_TREE_DEPTH_BONUS.
DEFAULT_DEPTH_BONUS: float = float(
    os.environ.get("SGLANG_DFLASH_TREE_DEPTH_BONUS", "0.0")
)


# ---------------------------------------------------------------------------
# best_first dynamic tree builder
#
# Heap-driven search over a candidate space (depth, rank-within-depth). Each
# popped candidate becomes a tree node and pushes up to two new candidates:
#   - its first child  (depth+1, rank=0)  — uses popped node as parent
#   - its next sibling (same depth, rank+1) — uses popped node's parent
#
# Path log-probabilities are accumulated as the heap key (negated so heapq's
# min-heap becomes max-by-path-probability). The algorithm relies on DFlash's
# parent-independent draft distributions: at every depth the sorted top-k is
# fixed regardless of which parent the candidate hangs off.
#
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BestFirstTreeOut:
    """Per-request best_first tree.

    All tensors are on CPU. Move to the target device at the integration site.
    """

    tokens: torch.Tensor          # [tree_size] int64
    parent_indices: torch.Tensor  # [tree_size] int64
    depths: torch.Tensor          # [tree_size] int64
    first_child: torch.Tensor     # [tree_size] int64; -1 if leaf
    next_sibling: torch.Tensor    # [tree_size] int64; -1 if last child
    tree_size: int

    @property
    def max_depth(self) -> int:
        if self.tree_size == 0:
            return 0
        return int(self.depths.max().item())


def topk_logprobs_per_depth(
    logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-depth top-k log-probabilities and token ids.

    Args:
        logits: [..., depth, vocab]. log_softmax is applied along the last dim.
        k: number of candidates to retain per depth. Clamped to vocab size.

    Returns:
        (sorted_logprobs, sorted_token_ids), both [..., depth, k].
        sorted_logprobs is float32, sorted_token_ids is int64. Sorted descending.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    vocab = int(logits.shape[-1])
    k_eff = min(int(k), vocab)
    log_probs = torch.log_softmax(logits, dim=-1)
    sorted_vals, sorted_ids = torch.topk(log_probs, k=k_eff, dim=-1, sorted=True)
    return sorted_vals.float(), sorted_ids.long()


def build_first_child_next_sibling(
    parent_indices: List[int],
) -> Tuple[List[int], List[int]]:
    """Derive sibling-pointer form from a parent list.

    parent_indices[i] = index of parent (-1 for root). The list MUST be in an
    order where each parent appears before its children (best_first guarantees
    this by construction).

    Returns first_child[N], next_sibling[N], both with -1 sentinels.
    """
    n = len(parent_indices)
    first_child = [-1] * n
    next_sibling = [-1] * n
    last_child_of: dict[int, int] = {}
    for i, parent in enumerate(parent_indices):
        if parent < 0:
            continue
        if parent not in last_child_of:
            first_child[parent] = i
        else:
            next_sibling[last_child_of[parent]] = i
        last_child_of[parent] = i
    return first_child, next_sibling


def build_tree_mask_dense(parent_indices: List[int]) -> torch.Tensor:
    """Dense [N, N] bool tree mask. mask[i, j] = True iff j is ancestor of i or j == i.

    For attention backends that consume an explicit tree mask. Backends in the
    DFLASH "skip custom mask" set handle the tree shape natively from the
    sibling pointers and don't need this.

    Each row is accumulated as a Python int bitset and the [N, N] result is
    expanded in one vectorized step. Scattering the ancestor bits straight into
    a bool tensor instead costs one tensor op per ancestor edge, which dominates
    the whole tree build at the sizes used here.
    """
    n = len(parent_indices)
    if n > 63:
        # Bitsets no longer fit an int64 lane; fall back to direct scatter.
        mask = torch.eye(n, dtype=torch.bool)
        for i in range(n):
            p = parent_indices[i]
            while p >= 0:
                mask[i, p] = True
                p = parent_indices[p]
        return mask

    row_bits: List[int] = [0] * n
    for i in range(n):
        bits = 1 << i
        p = parent_indices[i]
        while p >= 0:
            bits |= 1 << p
            p = parent_indices[p]
        row_bits[i] = bits

    rows = torch.tensor(row_bits, dtype=torch.int64).unsqueeze(1)
    cols = torch.arange(n, dtype=torch.int64)
    return ((rows >> cols) & 1).bool()


def _grow_best_first_nodes(
    *,
    bonus_token_id: int,
    sorted_logprobs: List[List[float]],
    sorted_token_ids: List[List[int]],
    max_nodes: int,
    depth_bonus: float,
) -> Tuple[List[int], List[int], List[int], List[float]]:
    """Best-first heap walk, admitting up to `max_nodes` nodes.

    Returns (tokens, parent_indices, depths, path_probs) in admission order,
    where `path_probs[i]` is the *true* path probability (product of edge
    probabilities from root to node i, root == 1.0). The heap ordering key
    still folds in `depth_bonus` so the admission order — and hence any prefix
    of the returned lists — is identical to what the fixed-budget builder would
    produce; `path_probs` is tracked separately so the acceptance surrogate the
    controller consumes is the unbiased path probability regardless of the bonus.

    Stops early (returns fewer than `max_nodes`) only if the candidate heap is
    exhausted. Callers that need an exact size must check the length.
    """
    max_depth = len(sorted_logprobs)

    # Root-only.
    if max_nodes <= 1 or max_depth == 0:
        return [int(bonus_token_id)], [-1], [0], [1.0]

    k_static = len(sorted_logprobs[0])
    if k_static == 0:
        raise ValueError("sorted_logprobs[0] must have at least one entry.")

    # Candidate-space arrays. Index 0 is the root candidate (bonus token).
    cand_parent: List[int] = [-1]            # parent in candidate space
    cand_depth: List[int] = [0]
    cand_token: List[int] = [int(bonus_token_id)]
    cand_rank: List[int] = [-1]              # rank in per-depth top-k (-1 for root)
    cand_edge_logprob: List[float] = [0.0]   # edge from parent->this; root edge is 0
    cand_path_true: List[float] = [0.0]      # unbiased path log-prob (no depth_bonus)
    cand_tree_idx: List[int] = [-1]          # output-tree index; -1 until visited

    # Heap entries: (negated ordering key, candidate id). The key includes
    # depth_bonus; the unbiased path log-prob lives in cand_path_true.
    heap: List[Tuple[float, int]] = [(-0.0, 0)]

    tree_tokens: List[int] = []
    tree_parents: List[int] = []
    tree_depths: List[int] = []
    path_probs: List[float] = []

    while heap and len(tree_tokens) < max_nodes:
        neg_key, node_id = heapq.heappop(heap)
        cur_key = -neg_key

        parent_cand_id = cand_parent[node_id]
        tree_parent = (
            -1 if parent_cand_id < 0 else cand_tree_idx[parent_cand_id]
        )
        if parent_cand_id >= 0 and tree_parent < 0:
            # Best-first invariant: parents always pop before children.
            raise RuntimeError(
                "Parent must be visited before child in best-first tree search."
            )

        tree_tokens.append(cand_token[node_id])
        tree_parents.append(tree_parent)
        tree_depths.append(cand_depth[node_id])
        path_probs.append(math.exp(cand_path_true[node_id]))
        cand_tree_idx[node_id] = len(tree_tokens) - 1

        if len(tree_tokens) == max_nodes:
            break

        node_depth = cand_depth[node_id]
        node_path_true = cand_path_true[node_id]

        # Push first child at depth+1, rank 0.
        if node_depth < max_depth:
            child_depth = node_depth + 1
            child_lp = sorted_logprobs[child_depth - 1][0]
            child_tok = sorted_token_ids[child_depth - 1][0]
            child_id = len(cand_parent)
            cand_parent.append(node_id)
            cand_depth.append(child_depth)
            cand_token.append(child_tok)
            cand_rank.append(0)
            cand_edge_logprob.append(child_lp)
            cand_path_true.append(node_path_true + child_lp)
            cand_tree_idx.append(-1)
            heapq.heappush(
                heap, (-(cur_key + child_lp + depth_bonus), child_id)
            )

        # Push next sibling at same depth, rank+1, sharing the popped node's parent.
        if node_depth > 0:
            next_rank = cand_rank[node_id] + 1
            if next_rank < k_static:
                sib_lp = sorted_logprobs[node_depth - 1][next_rank]
                sib_tok = sorted_token_ids[node_depth - 1][next_rank]
                sib_id = len(cand_parent)
                cand_parent.append(cand_parent[node_id])
                cand_depth.append(node_depth)
                cand_token.append(sib_tok)
                cand_rank.append(next_rank)
                cand_edge_logprob.append(sib_lp)
                # Sibling shares the popped node's parent, so its unbiased path
                # log-prob is parent_path + sib_edge = (node_path - popped_edge) + sib_edge.
                cand_path_true.append(node_path_true - cand_edge_logprob[node_id] + sib_lp)
                cand_tree_idx.append(-1)
                # Ordering key: parent_key + sib_edge = (cur_key - popped_edge) + sib_edge.
                heapq.heappush(
                    heap,
                    (-(cur_key - cand_edge_logprob[node_id] + sib_lp), sib_id),
                )

    return tree_tokens, tree_parents, tree_depths, path_probs


def build_best_first_tree(
    *,
    bonus_token_id: int,
    sorted_logprobs: List[List[float]],   # [depth][k]
    sorted_token_ids: List[List[int]],    # [depth][k]
    verify_length: int,
    depth_bonus: float = None,
) -> BestFirstTreeOut:
    """Single-request best_first tree builder.

    Args:
        bonus_token_id: target's current token (root of the verify tree).
        sorted_logprobs: per-depth top-k log-probs, sorted descending.
            Shape [max_depth][k]. max_depth is `block_size - 1` for DFlash.
        sorted_token_ids: matching token ids at each (depth, rank).
        verify_length: total tree node budget (incl. root).
        depth_bonus: additive bonus on each child edge log-prob, biasing the
            search deeper. Defaults to DEFAULT_DEPTH_BONUS.

    Returns:
        BestFirstTreeOut with `tree_size == verify_length`.

    Raises:
        ValueError on malformed inputs OR when the heap exhausts before
        producing `verify_length` nodes (increase max_depth or per-depth k).
    """
    if verify_length < 1:
        raise ValueError(f"verify_length must be >= 1, got {verify_length}.")
    if depth_bonus is None:
        depth_bonus = DEFAULT_DEPTH_BONUS

    max_depth = len(sorted_logprobs)
    if max_depth > 0 and len(sorted_token_ids) != max_depth:
        raise ValueError(
            f"sorted_token_ids depth ({len(sorted_token_ids)}) does not match "
            f"sorted_logprobs depth ({max_depth})."
        )

    tokens, parents, depths, _ = _grow_best_first_nodes(
        bonus_token_id=bonus_token_id,
        sorted_logprobs=sorted_logprobs,
        sorted_token_ids=sorted_token_ids,
        max_nodes=verify_length,
        depth_bonus=depth_bonus,
    )

    if len(tokens) != verify_length:
        raise ValueError(
            f"Unable to build best-first tree with verify_length={verify_length}; "
            f"heap exhausted after {len(tokens)} nodes. Increase max_depth "
            f"(currently {max_depth}) or per-depth k."
        )

    return _finalize_tree(tokens=tokens, parent_indices=parents, depths=depths)


def _finalize_tree(
    *,
    tokens: List[int],
    parent_indices: List[int],
    depths: List[int],
) -> BestFirstTreeOut:
    """Assemble the BestFirstTreeOut from python lists. All output tensors CPU."""
    tree_size = len(tokens)
    first_child_list, next_sibling_list = build_first_child_next_sibling(parent_indices)
    return BestFirstTreeOut(
        tokens=torch.tensor(tokens, dtype=torch.int64),
        parent_indices=torch.tensor(parent_indices, dtype=torch.int64),
        depths=torch.tensor(depths, dtype=torch.int64),
        first_child=torch.tensor(first_child_list, dtype=torch.int64),
        next_sibling=torch.tensor(next_sibling_list, dtype=torch.int64),
        tree_size=tree_size,
    )


def build_best_first_trees_batched(
    *,
    bonus_tokens: torch.Tensor,         # [bs] int (any int dtype)
    sorted_logprobs: torch.Tensor,      # [bs, depth, K] float
    sorted_token_ids: torch.Tensor,     # [bs, depth, K] int
    verify_length: int,
    depth_bonus: float = None,
) -> List[BestFirstTreeOut]:
    """Build a best_first tree per request. Heap walk is CPU; inputs may be GPU.

    The function does ONE D2H transfer per input tensor (not per element).
    """
    if bonus_tokens.dim() != 1:
        raise ValueError(
            f"bonus_tokens must be 1D [bs], got shape {tuple(bonus_tokens.shape)}."
        )
    if sorted_logprobs.dim() != 3:
        raise ValueError(
            "sorted_logprobs must be 3D [bs, depth, K], got shape "
            f"{tuple(sorted_logprobs.shape)}."
        )
    if sorted_logprobs.shape != sorted_token_ids.shape:
        raise ValueError(
            "sorted_logprobs and sorted_token_ids must have the same shape; got "
            f"{tuple(sorted_logprobs.shape)} vs {tuple(sorted_token_ids.shape)}."
        )
    bs = int(bonus_tokens.shape[0])
    if int(sorted_logprobs.shape[0]) != bs:
        raise ValueError(
            f"sorted_logprobs batch dim {sorted_logprobs.shape[0]} != "
            f"bonus_tokens batch dim {bs}."
        )

    bonus_cpu = bonus_tokens.detach().cpu().tolist()
    lp_cpu = sorted_logprobs.detach().float().cpu().tolist()
    ids_cpu = sorted_token_ids.detach().long().cpu().tolist()

    trees: List[BestFirstTreeOut] = []
    for i in range(bs):
        trees.append(
            build_best_first_tree(
                bonus_token_id=int(bonus_cpu[i]),
                sorted_logprobs=lp_cpu[i],
                sorted_token_ids=ids_cpu[i],
                verify_length=verify_length,
                depth_bonus=depth_bonus,
            )
        )
    return trees


# ---------------------------------------------------------------------------
# beam_search (width-pruned) dynamic tree builder
#
# Ported from spec-dllm `model/tree_builder.py::build_width_pruned_tree_from_draft_logits`.
# Classic beam search: expand every frontier node by the per-depth top-k, keep the
# `width` highest cumulative-path-prob candidates as that depth's layer, and
# continue to the FULL block depth (block_size - 1). Unlike best_first there is no
# node budget cap — the tree always reaches max_depth, so paths can be up to
# max_depth long (acceptance is NOT capped at the width). Total nodes are a
# consequence of the width: 1 + width * max_depth (given width <= per-depth k). The
# fixed shape keeps batched verify rectangular.
# ---------------------------------------------------------------------------


def beam_tree_size(width: int, max_depth: int) -> int:
    """Node count (incl. root) of a full-depth width-`width` beam tree."""
    return 1 + max(1, int(width)) * max(0, int(max_depth))


def _grow_beam_search_nodes(
    *,
    bonus_token_id: int,
    sorted_logprobs: List[List[float]],
    sorted_token_ids: List[List[int]],
    width: int,
    max_depth: Optional[int] = None,
) -> Tuple[List[int], List[int], List[int], List[float]]:
    """Width-pruned (beam) expansion to the FULL `max_depth`.

    At each depth d in 1..max_depth, forms candidates by extending every frontier
    node with the per-depth top-k tokens, keeps the `width` highest by cumulative
    path log-prob, and advances. Returns (tokens, parent_indices, depths,
    path_probs) in level order (parents precede children). Total nodes =
    1 + width * max_depth when each depth has >= width candidates (true when the
    per-depth top-k has at least `width` entries). Matches spec-dllm width_pruned.
    """
    depth_avail = len(sorted_logprobs)
    if max_depth is None:
        max_depth = depth_avail
    max_depth = max(0, min(int(max_depth), depth_avail))
    width = max(1, int(width))

    if max_depth == 0:
        return [int(bonus_token_id)], [-1], [0], [1.0]

    tree_tokens: List[int] = [int(bonus_token_id)]
    tree_parents: List[int] = [-1]
    tree_depths: List[int] = [0]
    path_probs: List[float] = [1.0]

    # Frontier entries: (tree_index, cumulative path log-prob).
    frontier: List[Tuple[int, float]] = [(0, 0.0)]

    for depth in range(1, max_depth + 1):
        row_lp = sorted_logprobs[depth - 1]
        row_tok = sorted_token_ids[depth - 1]
        kk = len(row_lp)
        candidates: List[Tuple[float, int, int]] = []
        for pidx, base in frontier:
            for r in range(kk):
                candidates.append((base + row_lp[r], pidx, row_tok[r]))
        if not candidates:
            break
        n_sel = min(width, len(candidates))
        selected = heapq.nlargest(n_sel, candidates, key=lambda c: c[0])

        new_frontier: List[Tuple[int, float]] = []
        for score, pidx, tok in selected:
            idx = len(tree_tokens)
            tree_tokens.append(int(tok))
            tree_parents.append(int(pidx))
            tree_depths.append(depth)
            path_probs.append(math.exp(score))
            new_frontier.append((idx, score))
        frontier = new_frontier

    return tree_tokens, tree_parents, tree_depths, path_probs


def build_beam_search_tree(
    *,
    bonus_token_id: int,
    sorted_logprobs: List[List[float]],   # [depth][k]
    sorted_token_ids: List[List[int]],    # [depth][k]
    width: int,
    max_depth: Optional[int] = None,
) -> BestFirstTreeOut:
    """Single-request width-pruned (beam) tree builder.

    Builds a full-depth beam of width `width`; the resulting tree_size is
    ``1 + width * max_depth`` (max_depth defaults to the number of draft depths).
    Returns a ``BestFirstTreeOut``, so it is a drop-in for the verify path.
    """
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}.")

    depth_avail = len(sorted_logprobs)
    if depth_avail > 0 and len(sorted_token_ids) != depth_avail:
        raise ValueError(
            f"sorted_token_ids depth ({len(sorted_token_ids)}) does not match "
            f"sorted_logprobs depth ({depth_avail})."
        )

    tokens, parents, depths, _ = _grow_beam_search_nodes(
        bonus_token_id=bonus_token_id,
        sorted_logprobs=sorted_logprobs,
        sorted_token_ids=sorted_token_ids,
        width=width,
        max_depth=max_depth,
    )
    return _finalize_tree(tokens=tokens, parent_indices=parents, depths=depths)


def build_beam_search_trees_batched(
    *,
    bonus_tokens: torch.Tensor,         # [bs] int (any int dtype)
    sorted_logprobs: torch.Tensor,      # [bs, depth, K] float
    sorted_token_ids: torch.Tensor,     # [bs, depth, K] int
    width: int,
    max_depth: Optional[int] = None,
) -> List[BestFirstTreeOut]:
    """Build a width-pruned (beam) tree per request. Beam walk is CPU; inputs GPU.

    All trees have the same size (``1 + width * max_depth``), so the batched verify
    stays rectangular. Mirrors :func:`build_best_first_trees_batched`.
    """
    if bonus_tokens.dim() != 1:
        raise ValueError(
            f"bonus_tokens must be 1D [bs], got shape {tuple(bonus_tokens.shape)}."
        )
    if sorted_logprobs.dim() != 3:
        raise ValueError(
            "sorted_logprobs must be 3D [bs, depth, K], got shape "
            f"{tuple(sorted_logprobs.shape)}."
        )
    if sorted_logprobs.shape != sorted_token_ids.shape:
        raise ValueError(
            "sorted_logprobs and sorted_token_ids must have the same shape; got "
            f"{tuple(sorted_logprobs.shape)} vs {tuple(sorted_token_ids.shape)}."
        )
    bs = int(bonus_tokens.shape[0])
    if int(sorted_logprobs.shape[0]) != bs:
        raise ValueError(
            f"sorted_logprobs batch dim {sorted_logprobs.shape[0]} != "
            f"bonus_tokens batch dim {bs}."
        )

    bonus_cpu = bonus_tokens.detach().cpu().tolist()
    lp_cpu = sorted_logprobs.detach().float().cpu().tolist()
    ids_cpu = sorted_token_ids.detach().long().cpu().tolist()

    trees: List[BestFirstTreeOut] = []
    for i in range(bs):
        trees.append(
            build_beam_search_tree(
                bonus_token_id=int(bonus_cpu[i]),
                sorted_logprobs=lp_cpu[i],
                sorted_token_ids=ids_cpu[i],
                width=width,
                max_depth=max_depth,
            )
        )
    return trees


# ---------------------------------------------------------------------------
# Adaptive shared-budget controller
#
# Ports BASTION's per-request marginal cost/benefit stop rule
# (`bastion/tree_draft.py::build_adaptive_best_tree_from_draft_logits`) to
# synchronous batched decoding. BASTION grows a single request's tree until the
# marginal path-probability no longer pays for the marginal verify cost. Under
# batching every request shares ONE budget N (required by SGLang's rectangular
# `[bs, N]` verify), so the controller maximizes the batch objective
#
#     S_B(N) = (mean_b A_b(N)) * L_AR(B) / C_B(N),
#
# where A_b(N) is request b's accepted-length surrogate (sum of the true path
# probabilities of its first N best-first nodes; the root/bonus contributes 1.0)
# and C_B(N) = D_B + O_B + V_B(N). Each A_b is concave (best-first admits nodes
# in non-increasing path-probability order), so the batch average is concave; a
# convex C_B then makes S_B unimodal and the greedy marginal stop rule finds the
# argmax. Both sides of the stop test scale with any global latency factor, so
# N* depends only on the *relative* cost shape — see dflash_cost_model.py.
# ---------------------------------------------------------------------------


def select_shared_budget(
    *,
    accept_marginals: Sequence[Sequence[float]],
    batch_size: int,
    context_sum: int,
    cost_model,
    min_budget: int,
    max_budget: int,
) -> int:
    """Pick the shared tree budget N* for a batched speculative cycle.

    Args:
        accept_marginals: per-request path-probability sequences in best-first
            admission order (`accept_marginals[b][n-1]` = p_b(n), the unbiased
            path probability of request b's n-th node; index 0 is the root, 1.0).
            Sequences may be shorter than `max_budget`; missing entries count 0.
        batch_size: B.
        context_sum: sum_b of per-request context (prefix) lengths.
        cost_model: object exposing `fixed_overhead_s(B)`, `estimate_verify(B, N,
            context_sum)`. When it is None or not yet calibrated
            (`fixed_overhead_s` returns None), we fall back to `max_budget`.
        min_budget, max_budget: inclusive bounds on N*.

    Returns:
        N* in [min_budget, max_budget].
    """
    if max_budget < 1:
        raise ValueError(f"max_budget must be >= 1, got {max_budget}.")
    hi = int(max_budget)
    lo = max(1, min(int(min_budget), hi))
    if cost_model is None:
        return hi

    fixed = cost_model.fixed_overhead_s(batch_size)
    if fixed is None:
        return hi  # not calibrated yet: largest tree is the safe default
    if hi <= lo:
        return hi

    B = max(1, int(batch_size))
    inv_b = 1.0 / B

    # avg_marginal[n-1] = mean_b p_b(n), averaged over the batch (missing = 0).
    avg_marginal = [0.0] * hi
    for pp in accept_marginals:
        m = min(len(pp), hi)
        for n in range(m):
            avg_marginal[n] += pp[n]
    for n in range(hi):
        avg_marginal[n] *= inv_b

    # A_bar(N) = sum_{n<=N} avg_marginal[n]; start at N = lo.
    accept = sum(avg_marginal[:lo])
    verify_n = cost_model.estimate_verify(B, lo, context_sum)
    for n in range(lo, hi):
        marginal_next = avg_marginal[n]  # p(n+1) averaged over the batch
        verify_next = cost_model.estimate_verify(B, n + 1, context_sum)
        delta_verify = verify_next - verify_n
        cost_n = fixed + verify_n
        # Stop before growing to n+1 once the marginal ratio drops to/below the
        # average ratio: marginal_next / delta_verify <= accept / cost_n.
        if marginal_next * cost_n <= accept * delta_verify:
            return n
        accept += marginal_next
        verify_n = verify_next
    return hi


def build_adaptive_best_first_trees_batched(
    *,
    bonus_tokens: torch.Tensor,         # [bs] int (any int dtype)
    sorted_logprobs: torch.Tensor,      # [bs, depth, K] float
    sorted_token_ids: torch.Tensor,     # [bs, depth, K] int
    max_budget: int,
    min_budget: int,
    cost_model,
    context_sum: int,
    depth_bonus: Optional[float] = None,
) -> Tuple[List[BestFirstTreeOut], int, dict]:
    """Build one best_first tree per request under a controller-selected budget.

    Grows every request's tree to `max_budget` (recording per-node path
    probabilities), asks the controller for a single shared budget N*, then
    truncates each tree to N*. A truncated best-first tree stays valid because
    parents are always admitted before their children. Returns
    ``(trees, budget, stats)`` — all trees have ``tree_size == budget``.
    """
    if bonus_tokens.dim() != 1:
        raise ValueError(
            f"bonus_tokens must be 1D [bs], got shape {tuple(bonus_tokens.shape)}."
        )
    if sorted_logprobs.dim() != 3:
        raise ValueError(
            "sorted_logprobs must be 3D [bs, depth, K], got shape "
            f"{tuple(sorted_logprobs.shape)}."
        )
    if sorted_logprobs.shape != sorted_token_ids.shape:
        raise ValueError(
            "sorted_logprobs and sorted_token_ids must have the same shape; got "
            f"{tuple(sorted_logprobs.shape)} vs {tuple(sorted_token_ids.shape)}."
        )
    bs = int(bonus_tokens.shape[0])
    if int(sorted_logprobs.shape[0]) != bs:
        raise ValueError(
            f"sorted_logprobs batch dim {sorted_logprobs.shape[0]} != "
            f"bonus_tokens batch dim {bs}."
        )
    if max_budget < 1:
        raise ValueError(f"max_budget must be >= 1, got {max_budget}.")
    if depth_bonus is None:
        depth_bonus = DEFAULT_DEPTH_BONUS

    bonus_cpu = bonus_tokens.detach().cpu().tolist()
    lp_cpu = sorted_logprobs.detach().float().cpu().tolist()
    ids_cpu = sorted_token_ids.detach().long().cpu().tolist()

    grown: List[Tuple[List[int], List[int], List[int], List[float]]] = []
    reachable = int(max_budget)
    for i in range(bs):
        tokens, parents, depths, path_probs = _grow_best_first_nodes(
            bonus_token_id=int(bonus_cpu[i]),
            sorted_logprobs=lp_cpu[i],
            sorted_token_ids=ids_cpu[i],
            max_nodes=int(max_budget),
            depth_bonus=depth_bonus,
        )
        grown.append((tokens, parents, depths, path_probs))
        reachable = min(reachable, len(tokens))

    # Every request must be able to fill the chosen budget for rectangular
    # batching, so cap selection at the smallest reachable tree.
    hi = max(1, min(int(max_budget), reachable))
    lo = max(1, min(int(min_budget), hi))
    budget = select_shared_budget(
        accept_marginals=[g[3] for g in grown],
        batch_size=bs,
        context_sum=int(context_sum),
        cost_model=cost_model,
        min_budget=lo,
        max_budget=hi,
    )
    budget = max(1, min(int(budget), reachable))

    trees: List[BestFirstTreeOut] = []
    accept_sum = 0.0
    for tokens, parents, depths, path_probs in grown:
        trees.append(
            _finalize_tree(
                tokens=tokens[:budget],
                parent_indices=parents[:budget],
                depths=depths[:budget],
            )
        )
        accept_sum += float(sum(path_probs[:budget]))

    avg_accept = accept_sum / bs if bs > 0 else 0.0
    stats: dict = {
        "budget": int(budget),
        "max_budget": int(max_budget),
        "reachable": int(reachable),
        "avg_accept": avg_accept,
        "calibrated": bool(cost_model is not None and cost_model.is_ready(bs)),
    }
    if stats["calibrated"]:
        est_verify = cost_model.estimate_verify(bs, budget, int(context_sum))
        est_cycle = cost_model.fixed_overhead_s(bs) + est_verify
        ar_latency = cost_model.ar_latency_s(bs, int(context_sum))
        stats["est_cycle_s"] = est_cycle
        stats["est_speedup"] = (avg_accept * ar_latency / est_cycle) if est_cycle > 0 else 0.0

    return trees, int(budget), stats
