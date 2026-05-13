#!/usr/bin/env bash
# Launch DFLASH with the best_first dynamic tree builder.
#
# Tree size (= total nodes incl. root) is set by
# --speculative-dflash-best-first-tokens; block size (= max tree depth + 1)
# stays at --speculative-num-draft-tokens.
#
# Supported verify backends: triton, fa3, fa4. Flashinfer's custom-mask path
# for trees is not wired yet and will be rejected at startup.
# CUDA graph capture is auto-disabled (tree shape varies per step).
#
# Override backends with env vars, e.g. ATTN_BACKEND=triton DRAFT_BACKEND=flashinfer.

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

ATTN_BACKEND="${ATTN_BACKEND:-fa3}"
DRAFT_BACKEND="${DRAFT_BACKEND:-fa3}"

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
    --speculative-num-draft-tokens 16 \
    --speculative-dflash-best-first-tokens 128 \
    --tp-size 1 \
    --attention-backend "${ATTN_BACKEND}" \
    --speculative-draft-attention-backend "${DRAFT_BACKEND}" \
    --mem-fraction-static 0.75 \
    --trust-remote-code
