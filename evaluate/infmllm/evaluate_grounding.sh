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
dataset="refcoco_testA,refcoco_testB,refcoco+_testA,refcoco+_testB,refcocog_test"


python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NGPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --use_env \
    evaluate/infmllm/evaluate_grounding.py \
    \
    --model_path ${model_path} \
    --prompt='<image><ImageHere><ref>{}</ref>' \
    --dataset=${dataset} \
    --batch_size 2

echo "Done !!!"
echo "model_path: ${model_path}"
echo "dataset: ${dataset}"