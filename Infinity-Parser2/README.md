# Infinity-Parser2

<p align="center">
    <img src="https://raw.githubusercontent.com/infly-ai/INF-MLLM/main/Infinity-Parser2/assets/logo.png" width="400"/>
<p>

<p align="center">
🤗 <a href="https://huggingface.co/infly/Infinity-Parser2-Pro">Model</a> |
📊 <a>Dataset (coming soon...)</a> |
📄 <a>Paper (coming soon...)</a> |
🚀 <a>Demo (coming soon...)</a>
</p>

## Introduction

We are excited to release Infinity-Parser2-Pro, our latest flagship document understanding model that achieves a new state-of-the-art on olmOCR-Bench with a score of 86.7%, surpassing frontier models such as DeepSeek-OCR-2, PaddleOCR-VL, and dots.mocr. Building on our previous model Infinity-Parser-7B, we have significantly enhanced our data engine and multi-task reinforcement learning approach. This enables the model to consolidate robust multi-modal parsing capabilities into a unified architecture, delivering brand-new zero-shot capabilities for diverse real-world business scenarios.

### Key Features

- **Upgraded Data Engine**: We have comprehensively enhanced our synthetic data engine to support both fixed-layout and flexible-layout document formats. By generating over 1 million diverse full-text samples covering a wide range of document layouts, combined with a dynamic adaptive sampling strategy, we ensure highly balanced and robust multi-task learning across various document types.
- **Multi-Task Reinforcement Learning**: We designed a novel verifiable reward system to support Joint Reinforcement Learning (RL), enabling seamless and simultaneous co-optimization of multiple complex tasks, including doc2json and doc2markdown.
- **Breakthrough Parsing Performance**: It substantially outperforms our previous 7B model, achieving 86.7% on olmOCR-Bench, surpassing frontier models such as DeepSeek-OCR-2, PaddleOCR-VL, and dots.mocr.
- **Inference Acceleration**: By adopting the highly efficient MoE architecture, our inference throughput has increased by 21% (from 441 to 534 tokens/sec), reducing deployment latency and costs.

## Performance

<p align="left">
    <img src="https://raw.githubusercontent.com/infly-ai/INF-MLLM/main/Infinity-Parser2/assets/document_parsing_performance_evaluation.png" width="1200"/>
<p>

## Quick Start

### Installation

#### Pre-requisites

```bash
# Create a Conda environment (Optional)
conda create -n infinity_parser2 python=3.12
conda activate infinity_parser2

# Install PyTorch (CUDA). Find the proper version at https://pytorch.org/get-started/previous-versions based on your CUDA version.
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# Install FlashAttention (required for NVIDIA GPUs).
# This command builds flash-attn from source, which can take 10 to 30 minutes.
pip install flash-attn==2.8.3 --no-build-isolation
# For Hopper GPUs (e.g. H100, H800), we recommend FlashAttention-3 instead. See the official guide at https://github.com/Dao-AILab/flash-attention.

# Install vLLM
# NOTE: you may need to run the command below to resolve triton and numpy conflicts before installing vllm.
# pip uninstall -y pytorch-triton opencv-python opencv-python-headless numpy && rm -rf "$(python -c 'import site; print(site.getsitepackages()[0])')/cv2"
pip install vllm==0.17.1
```

#### Install infinity_parser2

```bash
# From PyPI
pip install infinity_parser2

# From source
git clone https://github.com/infly-ai/INF-MLLM.git
cd INF-MLLM/Infinity-Parser2
pip install -e .
```

### Usage

#### Command Line

The `parser` command is the fastest way to get started.

```bash
# Parse a PDF (outputs Markdown by default)
parser demo_data/demo.pdf

# Parse an image
parser demo_data/demo.png

# Batch parse multiple files
parser demo_data/demo.pdf demo_data/demo.png -o ./output

# Parse an entire directory
parser demo_data -o ./output

# Output raw JSON with layout bboxes
parser demo_data/demo.pdf --output-format json

# Convert to Markdown directly
parser demo_data/demo.png --task doc2md
```

```bash
# View all options
parser --help
```

#### Python API

```python
from infinity_parser2 import InfinityParser2

parser = InfinityParser2()

# Parse a single file (returns Markdown)
result = parser.parse("demo_data/demo.pdf")
print(result)

# Parse multiple files (returns list)
results = parser.parse(["demo_data/demo.pdf", "demo_data/demo.png"])

# Parse a directory (returns dict)
results = parser.parse("demo_data")
```

**Output formats:**

| task_type   | Description                                          | Default Output |
|-------------|------------------------------------------------------|----------------|
| `doc2json`  | Extract layout elements with bboxes (default)        | Markdown       |
| `doc2md`    | Directly convert to Markdown                         | Markdown       |
| `custom`    | Use your own prompt                                 | Raw model output |

```python
# doc2json: get raw JSON with bbox coordinates
result = parser.parse("demo_data/demo.pdf", output_format="json")

# doc2md: direct Markdown conversion
result = parser.parse("demo_data/demo.pdf", task_type="doc2md")

# Custom prompt
result = parser.parse("demo_data/demo.pdf", task_type="custom",
                      custom_prompt="Extract the title and authors only.")

# Batch processing with custom batch size
result = parser.parse("demo_data", batch_size=8)

# Save results to directory
parser.parse("demo_data/demo.pdf", output_dir="./output")
```

**Backends:**

Infinity-Parser2 supports three inference backends. By default it uses the **vLLM Engine** (offline batch inference).

```python
# vLLM Engine (default) — offline batch inference
parser = InfinityParser2(
    model_name="infly/Infinity-Parser2-Pro",
    backend="vllm-engine",        # default
    tensor_parallel_size=2,
)

# Transformers — local single-GPU inference
parser = InfinityParser2(
    model_name="infly/Infinity-Parser2-Pro",
    backend="transformers",
    device="cuda",
    torch_dtype="bfloat16",       # "float16" or "bfloat16"
)

# vLLM Server — online HTTP API (start server first)
parser = InfinityParser2(
    model_name="infly/Infinity-Parser2-Pro",
    backend="vllm-server",
    api_url="http://localhost:8000/v1/chat/completions",
    api_key="EMPTY",
)
```

To start a vLLM server:

```bash
vllm serve infly/Infinity-Parser2-Pro \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 65536 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --enable-prefix-caching
```

## API Reference

### InfinityParser2

```python
parser = InfinityParser2(model_name="infly/Infinity-Parser2-Pro")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"infly/Infinity-Parser2-Pro"` | HuggingFace model name or local path |
| `backend` | `str` | `"vllm-engine"` | Inference backend: `"transformers"`, `"vllm-engine"`, or `"vllm-server"` |
| `tensor_parallel_size` | `int` | `None` | GPU count by default. Tensor parallel size for vLLM Engine |
| `device` | `str` | `"cuda"` | Only `"cuda"` is supported |
| `api_url` | `str` | `"http://localhost:8000/v1/chat/completions"` | API URL for vLLM Server backend |
| `api_key` | `str` | `"EMPTY"` | API key for vLLM Server backend |
| `min_pixels` | `int` | `2048` | Minimum pixel count for image input (transformers backend only) |
| `max_pixels` | `int` | `16777216` | Maximum pixel count (~4096x4096), transformers backend only |
| `model_cache_dir` | `str` | `None` | Model cache directory (defaults to `~/.cache/infinity_parser2/`) |

### parse()

```python
result = parser.parse("demo_data/demo.pdf")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_data` | `str \| List[str] \| PIL.Image` | **Required** | File path(s), directory path, or PIL Image object |
| `task_type` | `str` | `"doc2json"` | `"doc2json"` (layout to JSON) \| `"doc2md"` (direct Markdown) \| `"custom"` |
| `custom_prompt` | `str` | `None` | Custom prompt; required when `task_type="custom"` |
| `batch_size` | `int` | `4` | Number of images to process per batch |
| `output_dir` | `str` | `None` | If set, saves results to this directory instead of returning them |
| `output_format` | `str` | `"md"` | `"md"` \| `"json"`. Only `"md"` is supported for `doc2md` / `custom` tasks |
| `**kwargs` | — | — | Additional args passed to the model (e.g., `max_new_tokens`, `temperature`) |

### Return Values

| Input           | output_dir=None                  | output_dir set |
|-----------------|----------------------------------|---------------|
| Single file     | `str`                            | `None`        |
| List of files   | `List[str]`                      | `None`        |
| Directory       | `Dict[str, str]` (path→content) | `None`        |

When `output_dir` is set, results are saved to `output_dir/{filename}/result.md` (or `result.json`).

## Advanced Usage

### Model Caching

Models are downloaded automatically on first use and cached at `~/.cache/infinity_parser2/`. You can customize the cache location:

```python
parser = InfinityParser2(
    model_name="infly/Infinity-Parser2-Pro",
    model_cache_dir="/path/to/cache"
)
```

### Generation Parameters

```python
result = parser.parse(
    "demo_data/demo.pdf",
    max_new_tokens=16384,
    temperature=0.01,
    top_p=0.95,
)
```

### Utility Functions

```python
from infinity_parser2 import (
    convert_pdf_to_images,
    convert_json_to_markdown,
    extract_json_content,
    get_files_from_directory,
    is_supported_file,
    SUPPORTED_TASK_TYPES,
    ModelCache,
    get_model_cache,
)

# Convert PDF pages to PIL Images
images = convert_pdf_to_images("demo_data/demo.pdf", dpi=300)

# Convert layout JSON to Markdown
markdown = convert_json_to_markdown(json_string)

# Check model cache
cache = get_model_cache()
print(cache.resolve_model_path("infly/Infinity-Parser2-Pro"))
```

## Requirements

- Python 3.12+
- CUDA-compatible GPU
- See `setup.py` for full dependency list.

## Acknowledgments

We would like to thank [Qwen3.5](https://github.com/QwenLM/Qwen3.5), [ms-swift](https://github.com/modelscope/ms-swift), [VeRL](https://github.com/verl-project/verl), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), [olmocr](https://huggingface.co/datasets/allenai/olmOCR-bench), [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR), [MinerU](https://github.com/opendatalab/MinerU), [dots.ocr](https://github.com/rednote-hilab/dots.ocr), [Chandra-OCR-2](https://github.com/datalab-to/chandra) for providing dataset, code and models.
