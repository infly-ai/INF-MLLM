#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:$PWD
pip install shortuuid openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple


model_path="./InfMLLM_7B_Chat"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate.infmllm_chat.model_vqa_science \
        --model-path ${model_path} \
        --question-file datasets/ScienceQA/data/llava_test_CQM-A.json \
        --image-folder datasets/ScienceQA/data/test \
        --answers-file ${model_path}/eval/scienceqa/answers/${CHUNKS}_${IDX}.jsonl \
        --single-pred-prompt \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait 


output_file=${model_path}/eval/scienceqa/answers/merged.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${model_path}/eval/scienceqa/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done



python evaluate/infmllm_chat/eval_science_qa.py \
    --base-dir datasets/ScienceQA/data \
    --result-file ${model_path}/eval/scienceqa/answers/merged.jsonl \
    --output-file ${model_path}/eval/scienceqa/answers/merged_output.jsonl \
    --output-result ${model_path}/eval/scienceqa/answers/merged_result.json

echo "model_path: ${model_path}"
