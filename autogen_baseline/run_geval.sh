#!/bin/bash

usage="Usage: $0 <input dir containing *_autogen_messages.jsonl files or jsonl file containing generated messages from the single agent baseline> [output_dir]"
[ $# -lt 1 ] && { echo "$usage"; exit 1; }
input_dir_or_file=$(readlink -ve $1) || { echo "$usage"; exit 1; }  
output_dir=$2

[ -z "$output_dir" ] && output_dir=$(dirname "$input_dir_or_file")
mkdir -p "$output_dir"

echo "Input data: $input_dir_or_file"
echo "Using output directory: $output_dir"

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


fprefix=$(basename "$input_dir_or_file" .jsonl)
set -x
python prepare_data_for_geval.py $input_dir_or_file ${output_dir}/${fprefix}_geval_data.json

for mode in coh con rel; do
    python geval/gpt4_eval.py \
        --model azure/gpt-4.1 \
        --prompt_fp geval/prompts/summeval/${mode}_detailed.txt \
        --summeval_fp ${output_dir}/${fprefix}_geval_data.json \
        --save_fp ${output_dir}/${fprefix}_geval_${mode}_results.json
done

python summarize_geval_scores.py ${output_dir}/${fprefix}_geval_{coh,con,rel}_results.json > ${output_dir}/${fprefix}_geval_summary.csv