#!/bin/bash

input_dir=$(readlink -ve $1) || { echo "Usage: $0 <input_dir containing *_autogen_messages.jsonl files>"; exit 1; }  

nfiles=$(ls $input_dir/*_autogen_messages.jsonl | wc -l)
[ $nfiles -eq 0 ] && {
    echo "Input directory contains no *_autogen_messages.jsonl files"
    exit 1
}

if [ ! -d geval ]; then
    git clone https://github.com/nlpyang/geval
    (cd geval; openai migrate;)
fi

[ -z "$OPENAI_BASE_URL" ] && \
    echo "WARNING: OPENAI_BASE_URL is not set, using default OpenAI endpoint" || \
    echo "Using OPENAI_BASE_URL: $OPENAI_BASE_URL"

[ -z "$OPENAI_API_KEY" ] && \
    echo "WARNING: OPENAI_API_KEY is not set, GPT-4.1 calls will fail" || \
    echo "Using provided OPENAI_API_KEY for GPT-4.1 calls"

cd $(dirname $0)

set -x
python prepare_data_for_geval.py $input_dir ${input_dir}/geval_data.json

for mode in coh con rel; do
    python geval/gpt4_eval.py \
        --model azure/gpt-4.1 \
        --prompt_fp geval/prompts/summeval/${mode}_detailed.txt \
        --summeval_fp ${input_dir}/geval_data.json \
        --save_fp ${input_dir}/geval_${mode}_results.json
done

python summarize_geval_results.py ${input_dir}/geval_{coh|con|rel}_results.json > ${input_dir}/geval_summary.csv