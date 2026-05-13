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
from dataclasses import dataclass
from typing import List, Tuple

import torch


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
    """
    n = len(parent_indices)
    mask = torch.eye(n, dtype=torch.bool)
    for i in range(n):
        p = parent_indices[i]
        while p >= 0:
            mask[i, p] = True
            p = parent_indices[p]
    return mask


def build_best_first_tree(
    *,
    bonus_token_id: int,
    sorted_logprobs: List[List[float]],   # [depth][k]
    sorted_token_ids: List[List[int]],    # [depth][k]
    verify_length: int,
) -> BestFirstTreeOut:
    """Single-request best_first tree builder.

    Args:
        bonus_token_id: target's current token (root of the verify tree).
        sorted_logprobs: per-depth top-k log-probs, sorted descending.
            Shape [max_depth][k]. max_depth is `block_size - 1` for DFlash.
        sorted_token_ids: matching token ids at each (depth, rank).
        verify_length: total tree node budget (incl. root).

    Returns:
        BestFirstTreeOut with `tree_size == verify_length`.

    Raises:
        ValueError on malformed inputs OR when the heap exhausts before
        producing `verify_length` nodes (increase max_depth or per-depth k).
    """
    if verify_length < 1:
        raise ValueError(f"verify_length must be >= 1, got {verify_length}.")

    max_depth = len(sorted_logprobs)
    if max_depth > 0 and len(sorted_token_ids) != max_depth:
        raise ValueError(
            f"sorted_token_ids depth ({len(sorted_token_ids)}) does not match "
            f"sorted_logprobs depth ({max_depth})."
        )

    # Trivial root-only tree.
    if verify_length == 1 or max_depth == 0:
        return _finalize_tree(
            tokens=[int(bonus_token_id)],
            parent_indices=[-1],
            depths=[0],
        )

    k_static = len(sorted_logprobs[0])
    if k_static == 0:
        raise ValueError("sorted_logprobs[0] must have at least one entry.")

    # Candidate-space arrays. Index 0 is the root candidate (bonus token).
    cand_parent: List[int] = [-1]            # parent in candidate space
    cand_depth: List[int] = [0]
    cand_token: List[int] = [int(bonus_token_id)]
    cand_rank: List[int] = [-1]              # rank in per-depth top-k (-1 for root)
    cand_edge_logprob: List[float] = [0.0]   # edge from parent->this; root edge is 0
    cand_tree_idx: List[int] = [-1]          # output-tree index; -1 until visited

    # Heap entries: (negated path log-prob, candidate id).
    heap: List[Tuple[float, int]] = [(-0.0, 0)]

    tree_tokens: List[int] = []
    tree_parents: List[int] = []
    tree_depths: List[int] = []

    while heap and len(tree_tokens) < verify_length:
        neg_path_lp, node_id = heapq.heappop(heap)
        cur_path_lp = -neg_path_lp

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
        cand_tree_idx[node_id] = len(tree_tokens) - 1

        if len(tree_tokens) == verify_length:
            break

        node_depth = cand_depth[node_id]

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
            cand_tree_idx.append(-1)
            heapq.heappush(heap, (-(cur_path_lp + child_lp), child_id))

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
                cand_tree_idx.append(-1)
                # Sibling path = parent_path + sib_edge
                #              = (cur_path - popped_edge) + sib_edge
                heapq.heappush(
                    heap,
                    (-(cur_path_lp - cand_edge_logprob[node_id] + sib_lp), sib_id),
                )

    if len(tree_tokens) != verify_length:
        raise ValueError(
            f"Unable to build best-first tree with verify_length={verify_length}; "
            f"heap exhausted after {len(tree_tokens)} nodes. Increase max_depth "
            f"(currently {max_depth}) or per-depth k (currently {k_static})."
        )

    return _finalize_tree(
        tokens=tree_tokens, parent_indices=tree_parents, depths=tree_depths
    )


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
            )
        )
    return trees
