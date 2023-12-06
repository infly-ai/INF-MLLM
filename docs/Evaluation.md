**Dependencies**
```
pip install pycocoevalcap tqdm spacy shortuuid openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1. InfMLLM (Stage-2 multitask finetuning)

#### Preparation

Prior to conducting evaluations, obtain the Vicuna-7B model and the InfMLLM-7B model from Hugging Face. Once downloaded, these should be placed in the ```pretrained_models``` directory.


To access comprehensive guidance on preparing evaluation datasets such as okvqa, vqav2, and others, it is advised to consult the [Qwen-VL](https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/EVALUATION.md) repository. 

Once prepared, the directory should have the following structure.

```
|-- rootdir
    |-- pretrained_models
        |-- lmsys/vicuna-7b-v1.5/
        |-- infmllm/InfMLLM-7B

    |-- datasets
        |-- okvqa
        |-- vqav2
        |-- TextVQA
        |-- gqa
        |-- ocr-vqa
        |-- refcoco
        |-- refcoco+
        |-- refcocog
```

#### Evaluation

To evaluate VQA benchmarks, execute the scripts provided by ```evaluate/infmllm/evaluate_vqa.sh```. The evaluated performance is expected to be as follows with InfMLLM-7B:

```
okvqa: 61.23
textvqa: 67.90
gqa: 63.06
ocr-vqa: 73.51
vqav2-testdev: 
```


The ```vqav2-testdev``` needs to be submitted to [eval.ai](https://eval.ai/web/challenges/challenge-page/830/my-submission) for evaluation through their online platform.


To evaluate visual grounding benchmarks, execute the scripts provided by ```evaluate/infmllm/evaluate_grounding.sh```. The evaluated performance is expected to be as follows with InfMLLM-7B:
```
refcoco_testA: 94.59
refcoco_testB: 89.24
refcoco+_testA: 92.33
refcoco+_testB: 81.61
refcocog_test: 89.78
```


### 2. InfmLLM-Chat (Stage-3 instruction tuning)

#### Preparation


Prior to conducting evaluations, obtain the InfMLLM-7B-Chat model from Hugging Face.

To access comprehensive guidance on preparing evaluation datasets such as MME, MMBench, and others, it is advised to consult the [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) repository.


Once prepared, the directory should have the following structure.

```
|-- rootdir
    |-- pretrained_models
        |-- infmllm/InfMLLM-7B-Chat

    |-- datasets
        |-- MME_Benchmark
        |-- mmbench
        |-- SEED-Bench
        |-- POPE
        |-- mm-vet
        |-- ScienceQA
        |-- TextVQA
        |-- gqa
        |-- VQAv2

```

#### Evauation

You can find all the scripts for evaluation in the ```evaluate/infmllm_chat/``` directory. For example, use the ```evaluate/infmllm_chat/seed.sh``` script to carry out the evaluation on the SEED benchmark.

The evaluated performance is expected to be as follows with InfMLLM-7B-Chat:
```
MME: 1498.87
MMBench: 
MMBench-CN: 
SEED: 61.70
POPE-f1: 86.56
MM-Vet: 32.9
ScienceQA-Image: 68.07
TextVQA: 63.91
GQA: 64.97
vqav2-testdev: 82.25
```