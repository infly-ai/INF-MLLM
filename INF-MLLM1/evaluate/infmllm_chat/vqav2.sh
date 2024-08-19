#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:$PWD
pip install shortuuid openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple


model_path="./InfMLLM_7B_Chat"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate.infmllm_chat.model_vqa_loader \
        --model-path ${model_path} \
        --question-file datasets/VQAv2/llava_1_5_vqav2_testdev.jsonl \
        --image-folder datasets/VQAv2/ \
        --answers-file ${model_path}/eval/vqav2/answers/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait


output_file=${model_path}/eval/vqav2/answers/merge.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${model_path}/eval/vqav2/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python evaluate/infmllm_chat/convert_vqav2_for_submission.py \
    --src ${output_file} \
    --test datasets/VQAv2/llava_1_5_vqav2_testdev.jsonl \
    --dst ${model_path}/eval/vqav2/answers/upload.json


echo "model_path: ${model_path}"
echo "submit to : https://eval.ai/web/challenges/challenge-page/830/my-submission"