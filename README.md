<div align="center">
<a><h2><img src="Infinity-Parser/assets/logo.png" height="24" width="24" style="display: inline"> INF-MLLM: Multimodal Large Language Models from INF Tech </h2></a>
</div>

## Introduction

INF-MLLM is a series of open-source multimodal large language models developed by INF Tech. This repository contains the code, models, and documentation for our projects, which aim to advance the state-of-the-art in visual-language understanding and document intelligence. We are committed to open research and have released our models and datasets to the community to foster collaboration and innovation.

## Updates

- [2025/06/30] The [Infinity-Doc-55K dataset](https://huggingface.co/datasets/infly/Infinity-Doc-55K) and [Infinity-Parser web demo](https://huggingface.co/spaces/infly/Infinity-Parser-Demo) are now available.
- [2025/05/27] We have added an introduction to our latest model, **Infinity-Parser**.
- [2025/04/22] VL-Rethinker models (7B & 72B) are released! They achieve new state-of-the-art results on MathVista, MathVerse, and MathVision benchmarks.
- [2024/08/19] We have released **INF-MLLM2**, with the [INF-MLLM2-7B model](https://huggingface.co/QianYEee/InfMLLM2_7B_chat) and evaluation code now available.
- [2023/12/06] The models and evaluation code for **INF-MLLM1** are now available.
- [2023/11/06] We have released **INF-MLLM1** and uploaded the initial version of the manuscript to [arXiv](https://arxiv.org/abs/2311.06791).

## Models

Here is a brief overview of the models available in this repository. For more details, please refer to the respective project directories.

### [Infinity-Parser](Infinity-Parser)

**Infinity-Parser** is an end-to-end scanned document parsing model trained with reinforcement learning. It is designed to maintain the original document's structure and content with high fidelity by incorporating verifiable rewards based on layout and content. Infinity-Parser demonstrates state-of-the-art performance on various benchmarks for text recognition, table and formula extraction, and reading-order detection.

- **Key Features:** Layout-aware, reinforcement learning, high-fidelity document parsing.
- **Paper:** [Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing](https://arxiv.org/abs/2506.03197)
- **Dataset:** [Infinity-Doc-55K](https://huggingface.co/datasets/infly/Infinity-Doc-55K)
- **Web Demo:** [Infinity-Parser-Demo](https://huggingface.co/spaces/infly/Infinity-Parser-Demo)

### [VL-Rethinker](https://github.com/TIGER-AI-Lab/VL-Rethinker)

**VL-Rethinker** is a project designed to incentivize the self-reflection capabilities of Vision-Language Models (VLMs) through Reinforcement Learning. The research introduces a novel technique called Selective Sample Replay (SSR) to enhance the GRPO algorithm, addressing the "vanishing advantages" problem. It also employs "Forced Rethinking" to explicitly guide the model through a self-reflection reasoning step. By combining these methods, VL-Rethinker significantly advances the state-of-the-art performance on multiple vision-language benchmarks, including MathVista, MathVerse, and MathVision.

- **Key Features:** Advanced RL techniques, fine-grained multimodal dataset, fully open-sourced.
- **Paper:** [VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning](https://arxiv.org/abs/2504.08837)
- **Dataset:** [ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K)
- **Models:** [VL-Rethinker-7B](https://huggingface.co/TIGER-Lab/VL-Rethinker-7B), [VL-Rethinker-72B](https://huggingface.co/TIGER-Lab/VL-Rethinker-72B)
- **Web Demo:** [VL-Rethinker-Demo](https://huggingface.co/spaces/TIGER-Lab/VL-Rethinker)

### [INF-MLLM2](INF-MLLM2)

**INF-MLLM2** is an advanced multimodal model with significant improvements in high-resolution image processing and document understanding. It supports dynamic image resolutions up to 1344x1344 pixels and features enhanced OCR capabilities for robust document parsing, table and formula recognition, and key information extraction.

- **Key Features:** High-resolution image support, advanced OCR, progressive multi-stage training.
- **Paper:** [Technical Report](INF-MLLM2/docs/tech_report.pdf)
- **Model:** [INF-MLLM2-7B](https://huggingface.co/QianYEee/InfMLLM2_7B_chat)

### [INF-MLLM1](INF-MLLM1)

**INF-MLLM1** is a unified model for a wide range of visual-language tasks. It is designed to handle both multitask and instruction-tuning scenarios, demonstrating strong performance on various VQA and visual grounding datasets.

- **Key Features:** Unified framework, multitask learning, instruction tuning.
- **Paper:** [InfMLLM: A Unified Framework for Visual-Language Tasks](https://arxiv.org/abs/2311.06791)
- **Models:** [InfMLLM-7B](https://huggingface.co/mightyzau/InfMLLM_7B), [InfMLLM-7B-Chat](https://huggingface.co/mightyzau/InfMLLM_7B_Chat), [InfMLLM-13B-Chat](https://huggingface.co/mightyzau/inf-mllm-13b-chat)




