export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp-size 1 \
    --attention-backend flashinfer \
    --speculative-draft-attention-backend flashinfer \
    --mem-fraction-static 0.75 \
    --trust-remote-code