export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Optional: enable schedule overlapping (experimental, may not be stable)
# export SGLANG_ENABLE_SPEC_V2=1
# export SGLANG_ENABLE_DFLASH_SPEC_V2=1
# export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
    --speculative-num-draft-tokens 16 \
    --tp-size 1 \
    --attention-backend triton \
    --speculative-draft-attention-backend flashinfer \
    --mem-fraction-static 0.75 \
    --trust-remote-code