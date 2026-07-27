#!/usr/bin/env bash
# Launch DFLASH with the adaptive best-first tree controller whose verify cost
# model is calibrated OFFLINE — i.e. once at startup, before serving — by a
# synthetic sweep over the real target-verify kernels. This is the BASTION-style
# baseline: a per-batch-size affine roofline (max(alpha_c*compute+beta_c,
# alpha_m*memory+beta_m)) is least-squares fit to the measured sweep, then frozen
# so the chosen budget N* is deterministic with zero per-step timing overhead.
#
#   run_dflash.sh                          -> chain (no tree)
#   run_dflash_best_first_tree.sh          -> fixed best-first tree (constant budget)
#   run_dflash_adaptive_tree.sh            -> adaptive tree (same startup-fit path)
#   run_dflash_best_first_tree_offline.sh  -> adaptive tree + dump the fitted calibration
#
# The startup sweep runs the real prefill->draft->verify path over a coarse
# (batch, tree_size) grid and fits one affine roofline per batch-size bucket, so
# the calibration reflects how batching amortizes weight loads. It adds a short
# one-time cost to startup (~tens of seconds). Tune the sweep via env:
#   SGLANG_DFLASH_CALIB_MAX_BATCH (default 32)  highest batch size to calibrate
#   SGLANG_DFLASH_CALIB_CONTEXT   (default 2048) representative context length
#   SGLANG_DFLASH_CALIB_REPEATS   (default 2)    timed repeats per grid point
#   SGLANG_DFLASH_CALIB_WARMUP    (default 1)    discarded warmup repeats
#
# Keep --model/--backends/--speculative-num-draft-tokens/N_max identical to the
# other launchers so the only variable across baselines is the tree strategy.
# Target backend must be triton, fa3, or fa4; CUDA graph is auto-disabled; verify
# is greedy-only (bench with temperature 0).

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

ATTN_BACKEND="${ATTN_BACKEND:-triton}"
DRAFT_BACKEND="${DRAFT_BACKEND:-fa3}"
PORT="${PORT:-30000}"
# Target + DFlash draft model. Override both together to switch models, e.g.:
#   MODEL=Qwen/Qwen3-4B DRAFT_MODEL=z-lab/Qwen3-4B-DFlash-b16 ./run_dflash_best_first_tree_offline.sh
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}"
# N_max (budget upper bound) and the per-cycle floor. Match run_dflash_adaptive_tree.sh.
MAX_TOKENS="${MAX_TOKENS:-64}"
MIN_TOKENS="${MIN_TOKENS:-1}"
# Dump the fitted startup calibration to a per-model file so switching models
# never clobbers another model's calibration (e.g. dflash_calib_qwen3-8b.json).
MODEL_SLUG="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
CALIB="${CALIB:-$(dirname "$0")/dflash_calib_${MODEL_SLUG}.json}"

echo "DFLASH offline adaptive tree: MODEL=$MODEL N_max=$MAX_TOKENS CALIB=$CALIB PORT=$PORT"

python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "${DRAFT_MODEL}" \
    --speculative-num-draft-tokens 16 \
    --speculative-dflash-best-first-tokens "${MAX_TOKENS}" \
    --speculative-dflash-adaptive-tree \
    --speculative-dflash-tree-min-tokens "${MIN_TOKENS}" \
    --speculative-dflash-cost-calibration-path "${CALIB}" \
    --tp-size 1 \
    --attention-backend "${ATTN_BACKEND}" \
    --speculative-draft-attention-backend "${DRAFT_BACKEND}" \
    --mem-fraction-static 0.75 \
    --port "${PORT}" \
    --trust-remote-code

# Watch startup for:
#   "DFLASH calibration fit complete in ...s from N samples ... buckets={1:'affine',...}"
#   "DFLASH dumped fitted calibration to $CALIB."
# then, per cycle: "DFLASH adaptive cycle #k: N*=.../64 ... est_speedup=...".
#
# Bench with the same greedy body as the other baselines (tree verify raises on
# non-greedy):
#   --extra-request-body '{"temperature": 0, "top_k": 1}'
# or via the driver: (in ~/dflash) ./scripts/run_bastion.sh 1 4 8 16
#
# To run all baselines side by side, give each its own GPU and PORT, e.g.:
#   CUDA_VISIBLE_DEVICES=0 PORT=30000 ./run_dflash.sh
#   CUDA_VISIBLE_DEVICES=1 PORT=30001 ./run_dflash_best_first_tree.sh
#   CUDA_VISIBLE_DEVICES=2 PORT=30002 ./run_dflash_best_first_tree_offline.sh
