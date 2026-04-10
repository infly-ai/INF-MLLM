# Infinity-Parser2

A document parsing Python package that converts PDFs and images into structured text content.

## Features

- **Multi-format Support**: Parse PDF files and images (PNG, JPG, etc.)
- **Batch Processing**: Handle single files, file lists, or entire directories
- **Multiple Inference Backends**: Support for transformers, vLLM Engine, and vLLM Server
- **Flexible Deployment**: Offline batch inference and online API service
- **Extensible Backend System**: Easy to add new inference backends

## Model Information

- **Model Name**: Infinity-Parser2-Pro
- **Model Hub**: [infly/Infinity-Parser2-Pro](https://huggingface.co/infly/Infinity-Parser2-Pro)
- **Architecture**: Qwen3.5 MoE, 35B parameters
- **Benchmark Performance**: 86.7% on olmOCR-Bench

## Quick Start

### Installation

#### Install via pip (Recommended)

```bash
pip install infinity-parser2
```

#### Install from Source

```bash
git clone https://github.com/infly-ai/INF-MLLM.git
cd INF-MLLM/Infinity-Parser2
pip install -e .
```

### Model Inference

#### 1. vLLM Engine (Offline Batch Inference)

```python
from infinity_parser2 import InfinityParser2

# Initialize parser (default backend)
parser = InfinityParser2(model_name="infly/Infinity-Parser2-Pro")

# Parse a single file
result = parser.parse("document.pdf")
print(result)

# Parse multiple files
results = parser.parse(["doc1.pdf", "doc2.png", "image.jpg"])

# Parse an entire directory
results = parser.parse("/path/to/documents/")
```

#### 2. vLLM Server (Online HTTP API)

**Start the vLLM server:**

```bash
vllm serve infly/Infinity-Parser2-Pro \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2
```

**Send inference requests:**

```python
from infinity_parser2 import InfinityParser2

parser = InfinityParser2(
    backend="vllm-server",
    api_url="http://localhost:8000/v1/chat/completions"
)

result = parser.parse("document.pdf")
print(result)
```

#### 3. Transformers Backend (Local Inference)

```python
from infinity_parser2 import InfinityParser2

parser = InfinityParser2(
    backend="transformers",
    device="cuda"
)

result = parser.parse("document.pdf")
print(result)
```

## Requirements

- Python >= 3.12
- torch >= 2.10.0
- transformers >= 5.3.0
- vllm >= 0.17.1
- qwen-vl-utils
- Pillow >= 9.0.0
- pypdf >= 3.0.0

See `requirements.txt` for full dependency list.

## Benchmark Results

Evaluation results on olmOCR-Bench:

| Task | Score |
|------|-------|
| Overall | 86.7% |
| Arxiv Math | 88.3% |
| Old Scans Math | 90.8% |
| Table Tests | 89.4% |
| Long Tiny Text | 91.6% |
| Headers Footers | 93.9% |

## License

This project is licensed under the Apache-2.0 License.

## Citation

If you use Infinity-Parser2-Pro in your research, please cite:

```bibtex
Coming soon...
```
