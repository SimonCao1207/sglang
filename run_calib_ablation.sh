#!/usr/bin/env bash
# Calibration ablation for the DFLASH adaptive tree controller.
#
# Runs the SAME adaptive controller with two cost models, so the only variable is
# whether the verify cost model is calibrated:
#
#   CALIB_MODE=fit  (default) -> calibrated roofline: startup sweep fits a per-
#                                batch-size affine roofline on the real kernels.
#   CALIB_MODE=raw            -> uncalibrated: skip the sweep, select N* from the
#                                raw analytical roofline with zero fixed overhead.
#
# Everything else (model, backends, N_max, min-tokens) is held identical, so a
# throughput / acceptance-length gap between the two runs is attributable to the
# calibration alone. Give each condition its own GPU + PORT and benchmark both
# with the same client (greedy body). See calib_ablation.md for the protocol.
#
# Optional non-adaptive references for the same plot: run_dflash_best_first_tree.sh
# (fixed tree budget) and run_dflash.sh (chain).

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

CALIB_MODE="${CALIB_MODE:-fit}"          # fit | raw
export SGLANG_DFLASH_CALIB_MODE="${CALIB_MODE}"

ATTN_BACKEND="${ATTN_BACKEND:-triton}"
DRAFT_BACKEND="${DRAFT_BACKEND:-fa3}"
PORT="${PORT:-30000}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}"
MAX_TOKENS="${MAX_TOKENS:-64}"           # N_max (budget upper bound)
MIN_TOKENS="${MIN_TOKENS:-1}"

if [ "$CALIB_MODE" != "fit" ] && [ "$CALIB_MODE" != "raw" ]; then
    echo "ERROR: CALIB_MODE must be 'fit' or 'raw', got '$CALIB_MODE'." >&2
    exit 1
fi

# Dump the fitted calibration only in fit mode (raw does no fitting).
MODEL_SLUG="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
CALIB_ARGS=()
if [ "$CALIB_MODE" = "fit" ]; then
    CALIB="${CALIB:-$(dirname "$0")/dflash_calib_${MODEL_SLUG}.json}"
    CALIB_ARGS=(--speculative-dflash-cost-calibration-path "$CALIB")
fi

echo "DFLASH calibration ablation: CALIB_MODE=$CALIB_MODE MODEL=$MODEL N_max=$MAX_TOKENS PORT=$PORT"

python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "${DRAFT_MODEL}" \
    --speculative-num-draft-tokens 16 \
    --speculative-dflash-best-first-tokens "${MAX_TOKENS}" \
    --speculative-dflash-adaptive-tree \
    --speculative-dflash-tree-min-tokens "${MIN_TOKENS}" \
    "${CALIB_ARGS[@]}" \
    --tp-size 1 \
    --attention-backend "${ATTN_BACKEND}" \
    --speculative-draft-attention-backend "${DRAFT_BACKEND}" \
    --mem-fraction-static 0.75 \
    --port "${PORT}" \
    --trust-remote-code

# Startup confirms the condition:
#   fit -> "calibration=FIT ... calibration fit complete in ...ms ... buckets={...}"
#   raw -> "calibration=RAW (uncalibrated). Skipping the startup sweep ..."
# Then bench both servers identically, e.g. (in ~/dflash):
#   ./scripts/run_bastion.sh 1 4 8 16 32 64
