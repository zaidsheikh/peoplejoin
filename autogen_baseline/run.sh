#!/bin/bash

cd $(dirname $0)

export LLM_BASE_URL=${LLM_BASE_URL:-"https://api.fireworks.ai/inference/v1"}
[ -z "$LLM_API_KEY" ] && echo "WARNING: LLM_API_KEY is not set"

mkdir -p output/

cat ../data/peoplejoin-qa/dev.jsonl | jq -r '[.question, .datum_id, .tenant_id] | @tsv' | \
while IFS=$'\t' read -r question datum_id tenant_id; do
    [ -f output/${datum_id}_main.log ] && echo "Skipping ${datum_id} as output file already exists" && continue
    python main.py \
        --groupchat_type graph \
        --output_dir output/ \
        --question "${question}" \
        --datum_id "${datum_id}" \
        --tenant_id "${tenant_id}" \
        --data_dir ../data/peoplejoin-qa/ \
        --primary_llm_model "accounts/fireworks/models/qwen3-235b-a22b-thinking-2507" \
        --default_llm_model "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507" \
        |& tee output/${datum_id}_main.log
    python utils/pretty_print_llm_traces.py < output/${datum_id}_llm_calls.jsonl > output/${datum_id}_llm_calls.yaml
done
