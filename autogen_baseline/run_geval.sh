#!/bin/bash -x

# directory containing *_autogen_messages.jsonl
input_dir=$(readlink -ve $1) || { echo "Invalid input directory"; exit 1; }  

cd $(dirname $0)
python prepare_data_for_geval.py $input_dir ${input_dir}/geval_data.json

git clone https://github.com/nlpyang/geval
(cd geval; openai migrate;)

for mode in coh con rel; do
    python geval/gpt4_eval.py \
        --model azure/gpt-4.1 \
        --prompt_fp geval/prompts/summeval/${mode}_detailed.txt \
        --summeval_fp ${input_dir}/geval_data.json \
        --save_fp ${input_dir}/geval_${mode}_results.json
done