#!/usr/bin/env bash
# Launch DFLASH with the beam_search (width-pruned) tree builder, the A/B
# counterpart of run_dflash_best_first_tree.sh.
#
# Ported from spec-dllm build_width_pruned_tree_from_draft_logits: classic beam
# search that keeps BEAM_WIDTH candidates per depth and expands to the FULL block
# depth (block_size - 1). So the deepest path is block_size - 1 (acceptance is NOT
# capped at the width), and the tree size is a consequence of the width:
#     nodes = 1 + BEAM_WIDTH * (block_size - 1)
# e.g. block_size=16 (depth 15): W=1 -> 16 (chain), W=2 -> 31, W=4 -> 61, W=8 -> 121.
#
# BEAM_WIDTH is the knob (not a total-node budget). To compare against best_first
# at the SAME node budget, set that script's BEST_FIRST_TOKENS to the node count
# printed below.
#
# Target backend must be triton, fa3, or fa4; CUDA graph is auto-disabled. Verify
# is greedy-only.

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

ATTN_BACKEND="${ATTN_BACKEND:-triton}"
DRAFT_BACKEND="${DRAFT_BACKEND:-fa3}"
PORT="${PORT:-30000}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"          # = --speculative-num-draft-tokens
BEAM_WIDTH="${BEAM_WIDTH:-4}"           # nodes kept per depth (the knob)

# Full-depth beam node count (incl. root): 1 + width * (block_size - 1).
NODES=$(( 1 + BEAM_WIDTH * (BLOCK_SIZE - 1) ))

echo "DFLASH beam_search: MODEL=$MODEL width=$BEAM_WIDTH depth=$((BLOCK_SIZE-1)) -> nodes=$NODES PORT=$PORT"
echo "  (to match best_first at this budget: BEST_FIRST_TOKENS=$NODES ./run_dflash_best_first_tree.sh)"

python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "${DRAFT_MODEL}" \
    --speculative-num-draft-tokens "${BLOCK_SIZE}" \
    --speculative-dflash-best-first-tokens "${NODES}" \
    --speculative-dflash-tree-method beam_search \
    --speculative-dflash-beam-width "${BEAM_WIDTH}" \
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
