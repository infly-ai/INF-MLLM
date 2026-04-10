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

#### 3. Transformers Inference

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

# Load model and processor
model_name = "infly/Infinity-Parser2-Pro"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

# Load image
image = Image.open("document.pdf")  # or Image.open("image.jpg")

# Build conversation
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Please parse this document and extract all text content."}
        ]
    }
]

# Process inputs
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)
inputs = inputs.to("cuda")

# Run inference
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(output_text)
```

## API Reference

### InfinityParser2

Main document parser class.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"infly/Infinity-Parser2-Pro"` | Model name or local path |
| `backend` | `str` | `"vllm-engine"` | Inference backend, `"transformers"`, `"vllm-engine"`, or `"vllm-server"` |
| `tensor_parallel_size` | `int` | `1` | Tensor parallel size for vllm-engine |
| `device` | `str` | `"cuda"` | Device type |

#### Methods

##### `parse(input_data, **kwargs)`

Parse documents and extract text content.

**Parameters:**
- `input_data`: Can be one of:
  - `str`: Single file path or directory path
  - `List[str]`: List of file paths
  - `PIL.Image.Image`: Image object
  - `bytes`: Image or PDF bytes

**Returns:**
- Parsed text content. Returns a list if multiple inputs are provided.

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
