#!/usr/bin/env bash
# T18: text baselines for the 1.5B reader that the soft compressor (P1) is built on, so soft rows
# are compared with the same reader. Compressed contexts come from the on-disk cache filled by
# run_all_compression.sh, so only LLM calls are made. Start the server first:
#   VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
#     --served-model-name qwen2.5-1.5b-instruct --max-model-len 4096 --port 8000 --seed 0
set -u
export LLM_MODEL=qwen2.5-1.5b-instruct
P3="--positions 0 9 19"
CONFIGS=(
  "closedbook|{}|--positions 0"
  "oracle|{}|--positions 0"
  "full|{}|$P3"
  "rerank_topk|{\"k\": 5}|$P3"
  "provence|{\"threshold\": 0.1}|$P3"
)
mkdir -p data/logs
for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r method params extra <<< "$spec"
  log="data/logs/$(date +%H%M%S)_1p5b_${method}.log"
  echo "=== $method $params $extra"
  # shellcheck disable=SC2086
  uv run python scripts/run_compression.py --method "$method" --params "$params" $extra "$@" \
    > "$log" 2>&1 || echo "FAILED $method $params (see $log)"
  grep -E "^gold@[0-9]+: acc" "$log" | sed 's/ -> .*//'
done
