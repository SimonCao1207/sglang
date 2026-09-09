#!/usr/bin/env bash
# Launch DFLASH with the adaptive shared-budget tree controller.
#
# --speculative-dflash-best-first-tokens is the budget UPPER BOUND N_max;
# --speculative-dflash-adaptive-tree turns on the controller, which picks a
# shared per-cycle budget N* in [min-tokens, N_max] that maximizes the batch
# speedup surrogate S_B(N) = mean_b A_b(N) * L_AR(B) / C_B(N). The cost model
# C_B(N) is calibrated ONCE at startup: a synthetic sweep over the real target-
# verify kernels is least-squares fit to a per-batch-size affine roofline
# (BASTION-style), then frozen -- so N* is deterministic with no per-step timing.
# The startup sweep adds a short one-time cost (~tens of seconds); tune it with
# SGLANG_DFLASH_CALIB_{MAX_BATCH,CONTEXT,REPEATS,WARMUP} (see the offline script).
#
# Target backend must be triton, fa3, or fa4; CUDA graph is auto-disabled (the
# tree size varies per step). Verify is greedy-only.

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

ATTN_BACKEND="${ATTN_BACKEND:-triton}"
DRAFT_BACKEND="${DRAFT_BACKEND:-fa3}"
PORT="${PORT:-30000}"
# N_max (budget upper bound) and the per-cycle floor.
MAX_TOKENS="${MAX_TOKENS:-64}"
MIN_TOKENS="${MIN_TOKENS:-1}"

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
    --speculative-num-draft-tokens 16 \
    --speculative-dflash-best-first-tokens "${MAX_TOKENS}" \
    --speculative-dflash-adaptive-tree \
    --speculative-dflash-tree-min-tokens "${MIN_TOKENS}" \
    --tp-size 1 \
    --attention-backend "${ATTN_BACKEND}" \
    --speculative-draft-attention-backend "${DRAFT_BACKEND}" \
    --mem-fraction-static 0.75 \
    --port "${PORT}" \
    --trust-remote-code

# Bench with the same greedy body as run_dflash.sh, or tree verify raises:
#   --extra-request-body '{"temperature": 0, "top_k": 1}'
#
# The verify cost model is calibrated automatically at startup (no flags needed).
# To also dump the fitted calibration for inspection, add:
#   --speculative-dflash-cost-calibration-path ./dflash_calib.json
# (see run_dflash_best_first_tree_offline.sh, which sets this by default).
