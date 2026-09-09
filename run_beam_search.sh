#!/usr/bin/env bash
# Launch DFLASH with the beam_search (width-pruned) tree builder, the A/B
# counterpart of run_dflash_best_first_tree.sh.
#
# Ported from spec-dllm build_width_pruned_tree_from_draft_logits: classic beam
# search that keeps BEAM_WIDTH candidates per depth and expands to BEAM_DEPTH
# levels (default: the FULL block depth, block_size - 1). The tree size is a
# consequence of width and depth:
#     nodes = 1 + BEAM_WIDTH * BEAM_DEPTH
# Full depth (BEAM_DEPTH unset): W=1 -> 16 (chain), W=2 -> 31, W=4 -> 61, W=8 -> 121.
# Capped depth trades depth for width at a fixed budget, e.g. BEAM_WIDTH=2
# BEAM_DEPTH=8 -> 1 + 2*8 = 17 nodes.  NOTE: acceptance length is CAPPED at
# BEAM_DEPTH, so a shallow beam cannot accept more than BEAM_DEPTH tokens/step.
#
# BEAM_WIDTH / BEAM_DEPTH are the knobs (not a total-node budget). To compare
# against best_first at the SAME node budget, set that script's BEST_FIRST_TOKENS
# to the node count printed below.
#
# Target backend must be triton, fa3, or fa4; CUDA graph is auto-disabled.

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

ATTN_BACKEND="${ATTN_BACKEND:-triton}"
DRAFT_BACKEND="${DRAFT_BACKEND:-fa3}"
PORT="${PORT:-30000}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"          # = --speculative-num-draft-tokens
BEAM_WIDTH="${BEAM_WIDTH:-4}"           # nodes kept per depth (the knob)
# Beam depth (max path length). Empty = full depth (block_size - 1).
BEAM_DEPTH="${BEAM_DEPTH:-}"

FULL_DEPTH=$(( BLOCK_SIZE - 1 ))
DEPTH="${BEAM_DEPTH:-$FULL_DEPTH}"
if [ "${DEPTH}" -gt "${FULL_DEPTH}" ]; then DEPTH="${FULL_DEPTH}"; fi

# Beam node count (incl. root): 1 + width * depth.
NODES=$(( 1 + BEAM_WIDTH * DEPTH ))

echo "DFLASH beam_search: MODEL=$MODEL width=$BEAM_WIDTH depth=$DEPTH -> nodes=$NODES PORT=$PORT"
echo "  (to match best_first at this budget: BEST_FIRST_TOKENS=$NODES ./run_dflash_best_first_tree.sh)"

DEPTH_ARG=()
if [ -n "${BEAM_DEPTH}" ]; then
    DEPTH_ARG=(--speculative-dflash-beam-max-depth "${DEPTH}")
fi

python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "${DRAFT_MODEL}" \
    --speculative-num-draft-tokens "${BLOCK_SIZE}" \
    --speculative-dflash-best-first-tokens "${NODES}" \
    --speculative-dflash-tree-method beam_search \
    --speculative-dflash-beam-width "${BEAM_WIDTH}" \
    "${DEPTH_ARG[@]}" \
    --tp-size 1 \
    --attention-backend "${ATTN_BACKEND}" \
    --speculative-draft-attention-backend "${DRAFT_BACKEND}" \
    --mem-fraction-static 0.75 \
    --port "${PORT}" \
    --trust-remote-code

# Startup logs "DFLASH beam_search: width=W, depth=D -> verify_length=N nodes".
#
# Sampling: tree verify now supports temperature > 0 (distribution-preserving,
# one exact target sample per node). For a genuine T=1 run make sure top-k is
# DISABLED, otherwise top_k=1 silently forces greedy:
#   greedy : temperature=0                 (top_k value irrelevant)
#   T=1    : temperature=1, top_k=-1, top_p=1
# e.g. with run_bastion.sh:  TEMPERATURE=1.0 TOP_K=-1 TOP_P=1.0 ./run_bastion.sh
