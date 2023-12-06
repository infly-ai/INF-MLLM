#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:$PWD
pip install shortuuid openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple


model_path="./InfMLLM_7B_Chat"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


SPLIT="mmbench_dev_cn_20231003"
question_file=datasets/mmbench/$SPLIT.tsv
answer_dir="${model_path}/eval//mmbench/answers/$SPLIT"
upload_dir="${model_path}/eval/mmbench/answers_upload/$SPLIT"

mkdir -p ${answer_dir}
mkdir -p ${upload_dir}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate.infmllm_chat.model_vqa_mmbench \
        --model-path ${model_path} \
        --question-file ${question_file} \
        --answers-file ${answer_dir}/${CHUNKS}_${IDX}.jsonl \
        --lang cn \
        --single-pred-prompt \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait

output_file=${answer_dir}/vicuna_v1.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${answer_dir}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python evaluate/infmllm_chat/convert_mmbench_for_submission.py \
    --annotation-file ${question_file} \
    --result-dir ${answer_dir} \
    --upload-dir ${upload_dir} \
    --experiment vicuna_v1

echo "SPLIT: ${SPLIT}"
echo "model_path: ${model_path}"
echo "submit the results to the evaluation server: https://opencompass.org.cn/"
