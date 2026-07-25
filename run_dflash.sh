#!/usr/bin/env bash
# Launch DFLASH in chain mode — the A/B baseline for run_dflash_best_first_tree.sh.
#
# Backends and --disable-cuda-graph are pinned to match that script so the runs
# differ only in tree shape; both handicap chain mode. For chain's deployable
# number use ATTN_BACKEND=fa3 and drop --disable-cuda-graph.
#
# To run alongside the tree script, give each its own PORT and its own GPU:
#   CUDA_VISIBLE_DEVICES=0 PORT=30000 ./run_dflash.sh
#   CUDA_VISIBLE_DEVICES=1 PORT=30001 ./run_dflash_best_first_tree.sh

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Optional: enable schedule overlapping (experimental, may not be stable)
# export SGLANG_ENABLE_SPEC_V2=1
# export SGLANG_ENABLE_DFLASH_SPEC_V2=1
# export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

ATTN_BACKEND="${ATTN_BACKEND:-triton}"
DRAFT_BACKEND="${DRAFT_BACKEND:-fa3}"
PORT="${PORT:-30000}"

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
    --speculative-num-draft-tokens 16 \
    --tp-size 1 \
    --attention-backend "${ATTN_BACKEND}" \
    --speculative-draft-attention-backend "${DRAFT_BACKEND}" \
    --disable-cuda-graph \
    --mem-fraction-static 0.75 \
    --port "${PORT}" \
    --trust-remote-code

# Bench with the same greedy body as run_dflash_best_first_tree.sh:
#   python -m sglang.bench_serving --backend sglang \
#       --dataset-name random --random-input-len 1024 --random-output-len 512 \
#       --num-prompts 200 --max-concurrency 1 \
#       --extra-request-body '{"temperature": 0, "top_k": 1}'
