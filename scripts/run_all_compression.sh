#!/usr/bin/env bash
# Track A experiment matrix, trimmed to what the conclusions need (budget-aware design):
#  - closed-book / oracle do not depend on the gold position -> 1 position only;
#  - document-level / sentence-level / LLMLingua-2 compressors are position-invariant, so the
#    compression is computed once and reused for gold positions 1/10/20 (only the LLM call differs);
#  - truncate-2x and BM25/dense top-k are reported with LLM-free context recall plus the published
#    GPT-3.5 numbers instead of extra LLM runs;
#  - LongLLMLingua (the most expensive compressor) runs on 200 questions at the middle position.
# Pass --dry-run to only compress (fills the cache). Extra args are forwarded to run_compression.py.
# One config at a time; GPU for the compressors, vLLM for generation.
set -u
OP="hotchpotch/open-provence-reranker-v1-gte-modernbert-base"
P3="--positions 0 9 19"
CONFIGS=(
  "closedbook|{}|--positions 0"
  "oracle|{}|--positions 0"
  "full|{}|$P3"
  "truncate|{\"budget\": 730}|$P3"
  "bm25_topk|{\"k\": 5}|--positions 9"
  "dense_topk|{\"k\": 5}|--positions 9"
  "rerank_topk|{\"k\": 10}|$P3"
  "rerank_topk|{\"k\": 5}|$P3"
  "llmlingua2|{\"rate\": 0.5}|$P3"
  "llmlingua2|{\"rate\": 0.25}|$P3"
  "chain|{\"stages\": [{\"method\": \"rerank_topk\", \"k\": 10}, {\"method\": \"llmlingua2\", \"rate\": 0.5}]}|$P3"
  "provence|{\"threshold\": 0.01}|$P3"
  "provence|{\"threshold\": 0.1}|$P3"
  "provence|{\"threshold\": 0.1, \"model\": \"$OP\"}|$P3"
  "longllmlingua|{\"target_token\": 730}|--positions 9 --n 200"
)
mkdir -p data/logs
for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r method params extra <<< "$spec"
  log="data/logs/$(date +%H%M%S)_${method}.log"
  echo "=== $method $params $extra"
  # shellcheck disable=SC2086
  uv run python scripts/run_compression.py --method "$method" --params "$params" $extra "$@" \
    > "$log" 2>&1 || echo "FAILED $method $params (see $log)"
  grep -E "^gold@[0-9]+: acc" "$log" | sed 's/ -> .*//'
done
echo "ALL DONE"
