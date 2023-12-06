#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:$PWD
pip install shortuuid openpyxl scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple


model_path="./InfMLLM_7B_Chat"

question_file="datasets/MME_Benchmark/mme_llava_v1_5.json"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate.infmllm_chat.model_vqa_loader \
        --model-path ${model_path} \
        --question-file ${question_file} \
        --image-folder datasets/MME_Benchmark/ \
        --answers-file ${model_path}/eval/MME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait

output_file="${model_path}/eval/MME/mme_results.jsonl"
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${model_path}/eval/MME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mme_results_dir="${model_path}/eval/MME/mme_results"
python evaluate/infmllm_chat/convert_answer_to_mme.py --answer_file ${output_file} --question_file ${question_file} --out_path ${mme_results_dir}
python evaluate/infmllm_chat/calculation_mme.py --results_dir ${mme_results_dir}

echo "model_path: ${model_path}"
