## INF-MLLM2: High-Resolution Image and Document Understanding

<p align="center">
<img src="docs/model.png" alt="" width="100%"/>
</p>

[Technical Report](docs/tech_report.pdf)

### Install

```bash
conda create -n infmllm2 python=3.9
conda activate infmllm2
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.1.2

pip install transformers==4.40.2 timm==0.5.4 pillow==10.4.0 sentencepiece==0.1.99
pip install bigmodelvis peft einops spacy
```

### Model Zoo
We have released the INF-MLLM2-7B model on Hugging Face.
- [INF-MLLM2-7B](https://huggingface.co/QianYEee/InfMLLM2_7B_chat)


### Visualization

<p align="center">
<img src="docs/demo1.png" alt="" width="90%"/>
</p>

<p align="center">
<img src="docs/demo2.png" alt="" width="90%"/>
</p>

<p align="center">
<img src="docs/demo3.png" alt="" width="90%"/>
</p>

### Usage

The inference process for INF-MLLM2 is straightforward. We also provide a simple [demo.py](demo.py) script as a reference.

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py --model_path /path/to/InfMLLM2_7B_chat
```