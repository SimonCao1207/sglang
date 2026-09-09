#!/usr/bin/env bash
# Launch DFLASH with the best_first dynamic tree builder.
#
# Tree size (total nodes incl. root) is --speculative-dflash-best-first-tokens;
# --speculative-num-draft-tokens stays the block size (max tree depth + 1).
#
# Target backend must be triton, fa3, or fa4; triton consumes the tree mask in
# one kernel and wins once the budget exceeds 16. Keep the draft on fa3 (its
# forward has no tree mask). CUDA graph is auto-disabled since the tree shape
# varies per step. Verify is greedy-only.

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

ATTN_BACKEND="${ATTN_BACKEND:-triton}"
DRAFT_BACKEND="${DRAFT_BACKEND:-fa3}"
PORT="${PORT:-30000}"
BEST_FIRST_TOKENS="${BEST_FIRST_TOKENS:-16}"

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
    --speculative-num-draft-tokens 16 \
    --speculative-dflash-best-first-tokens "${BEST_FIRST_TOKENS}" \
    --tp-size 1 \
    --attention-backend "${ATTN_BACKEND}" \
    --speculative-draft-attention-backend "${DRAFT_BACKEND}" \
    --mem-fraction-static 0.75 \
    --port "${PORT}" \
    --trust-remote-code

# Bench with the same greedy body as run_dflash.sh, or tree verify raises:
#   --extra-request-body '{"temperature": 0, "top_k": 1}'
