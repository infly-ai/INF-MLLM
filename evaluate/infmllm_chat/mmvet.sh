#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:$PWD
pip install shortuuid openpyxl scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple


model_path="./InfMLLM_7B_Chat"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

mkdir -p ${model_path}/eval/mm-vet/answers

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate.infmllm_chat.model_vqa \
        --model-path ${model_path} \
        --question-file datasets/mm-vet/llava_1_5_mmvet.jsonl \
        --image-folder datasets/mm-vet/images \
        --answers-file ${model_path}/eval/mm-vet/answers/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait


output_file=${model_path}/eval/mm-vet/answers/merged.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${model_path}/eval/mm-vet/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


mkdir -p ${model_path}/eval/mm-vet/answers_submit
python evaluate/infmllm_chat/convert_mmvet_for_eval.py \
    --src ${output_file} \
    --dst ${model_path}/eval/mm-vet/answers_submit/result.json

echo "model_path: ${model_path}"
echo "submit to https://huggingface.co/spaces/whyu/MM-Vet_Evaluator"