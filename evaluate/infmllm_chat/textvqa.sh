#!/bin/bash
pip install shortuuid -i https://pypi.tuna.tsinghua.edu.cn/simple
export PYTHONPATH=${PYTHONPATH}:/home/ma-user/work/projects/infmllm

model_path="./InfMLLM_7B_Chat"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate.infmllm_chat.model_vqa_loader \
        --model-path ${model_path} \
        --question-file datasets/TextVQA/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder datasets/TextVQA/train_images \
        --answers-file ${model_path}/eval/textvqa/answers/${CHUNKS}_${IDX}.jsonl \
        --temperature 0 \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 &
done
wait


output_file=${model_path}/eval/textvqa/answers/prediction.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${model_path}/eval/textvqa/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python evaluate/infmllm_chat/eval_textvqa.py \
    --annotation-file datasets/TextVQA/TextVQA_0.5.1_val.json \
    --result-file ${output_file}

echo "model_path: ${model_path}"

