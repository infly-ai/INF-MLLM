# 系统默认环境变量，不建议修改
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_PORT="6060"
JOB_ID="1234"
NNODES="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
NGPUS_PER_NODE="$MA_NUM_GPUS"

export PYTHONPATH=${PYTHONPATH}:$PWD


model_path="./InfMLLM_7B"
dataset="okvqa_val,gqa_testdev,textvqa_val,ocrvqa_test,vqav2_testdev"

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NGPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --use_env \
    evaluate/infmllm/evaluate_vqa.py \
    \
    --model_path ${model_path} \
    --length_penalty=0 \
    --num_beams=5 \
    --min_len=1 \
    --prompt='<image><ImageHere>Question:{} Short answer:' \
    --dataset=${dataset} \
    --batch_size 2

echo "Done !!!"
echo "model_path: ${model_path}"
echo "dataset: ${dataset}"