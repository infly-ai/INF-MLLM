# INF-MLLM

<p align="center">
    <img src="Infinity-Parser/assets/logo.png" width="400"/>
<p>

## Introduction

INF-MLLM is a series of open-source multimodal large language models developed by INF Tech. This repository contains the code, models, and documentation for our projects, which aim to advance the state-of-the-art in visual-language understanding and document intelligence. We are committed to open research and have released our models and datasets to the community to foster collaboration and innovation.

## Updates

- [2026-05-11] Released flagship document parsing models: [Infinity-Parser2-Pro](https://huggingface.co/infly/Infinity-Parser2-Pro), [Infinity-Parser2-Flash](https://huggingface.co/infly/Infinity-Parser2-Flash), and the dataset [Infinity-Doc2-5M](https://huggingface.co/datasets/infly/Infinity-Doc2-5M). [Infinity-Parser2](Infinity-Parser2) achieves SOTA results on olmOCR-bench and ParseBench.
- [2026/04/08] Infinity-Parser has been accepted as a Findings paper at ACL 2026. 👏
- [2025/11/03] [Infinity-Parser](Infinity-Parser) models released: [Infinity-Parser-7B](https://huggingface.co/infly/Infinity-Parser-7B), [Infinity-Doc-400K dataset](https://huggingface.co/datasets/infly/Infinity-Doc-400K), synthetic data [generation code](https://github.com/infly-ai/INF-MLLM/tree/main/Infinity-Parser/Infinity-Synth), and [web demo](https://huggingface.co/spaces/infly/Infinity-Parser-Demo).
- [2025/09/19] VL-Rethinker has been accepted as a Spotlight paper at NeurIPS 2025. 👏👏
- [2025/04/22] [VL-Rethinker-7B](https://huggingface.co/TIGER-Lab/VL-Rethinker-7B) and [VL-Rethinker-72B](https://huggingface.co/TIGER-Lab/VL-Rethinker-72B) are released! They achieve new state-of-the-art results on MathVista, MathVerse, and MathVision benchmarks.
- [2024/08/19] We have released **INF-MLLM2**, with the [INF-MLLM2-7B model](https://huggingface.co/QianYEee/InfMLLM2_7B_chat) and evaluation code now available.
- [2023/12/06] The models and evaluation code for **INF-MLLM1** are now available. The initial manuscript is on [arXiv](https://arxiv.org/abs/2311.06791).

## Models

Here is a brief overview of the models available in this repository. For more details, please refer to the respective project directories.

### [Infinity-Parser2](Infinity-Parser2)

**Infinity-Parser2** is our latest flagship document parsing model, offering two distinct variants: Infinity-Parser2-Pro optimized for maximum accuracy, and Infinity-Parser2-Flash engineered for high-speed inference (3.68x faster than Infinity-Parser-7B). Built on an upgraded data engine supporting nearly 5 million diverse document samples and a novel multi-task reinforcement learning framework with joint verification rewards, Infinity-Parser2 achieves state-of-the-art results on olmOCR-Bench (87.6%) and ParseBench (74.3%), surpassing frontier models including DeepSeek-OCR-2, PaddleOCR-VL-1.5, and MinerU-2.5.

- **Key Features:** Upgraded data engine, multi-task RL, dual variants (Pro/Flash).
- **Models:** [Infinity-Parser2-Pro](https://huggingface.co/infly/Infinity-Parser2-Pro), [Infinity-Parser2-Flash](https://huggingface.co/infly/Infinity-Parser2-Flash)
- **Dataset:** [Infinity-Doc2-5M](https://huggingface.co/datasets/infly/Infinity-Doc2-5M)
- **Web Demo:** [Infinity-Parser2-Demo](https://huggingface.co/spaces/infly/Infinity-Parser2-Demo)

### [Infinity-Parser](Infinity-Parser)

**Infinity-Parser** is an end-to-end scanned document parsing model trained with reinforcement learning. It is designed to maintain the original document's structure and content with high fidelity by incorporating verifiable rewards based on layout and content. Infinity-Parser demonstrates state-of-the-art performance on various benchmarks for text recognition, table and formula extraction, and reading-order detection.

- **Key Features:** Layout-aware, reinforcement learning, high-fidelity document parsing.
- **Paper:** [Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing](https://arxiv.org/abs/2506.03197)
- **Dataset:** [Infinity-Doc-55K](https://huggingface.co/datasets/infly/Infinity-Doc-55K), [Infinity-Doc-400K](https://huggingface.co/datasets/infly/Infinity-Doc-400K)
- **Model:** [Infinity-Parser-7B](https://huggingface.co/infly/Infinity-Parser-7B)
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
