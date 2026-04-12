## Quick Start

### Installation

#### Install via pip (Recommended)

```bash
pip install infinity_parser2
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

# Parse a single PDF file
result = parser.parse("demo_data/demo.pdf")
print(result)

# Parse a single image file
result = parser.parse("demo_data/demo.png")
print(result)

# Parse multiple files
results = parser.parse([
    "demo_data/demo.pdf",
    "demo_data/demo.png"
])

# Parse an entire directory
results = parser.parse("demo_data")
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

result = parser.parse("demo_data/demo.pdf")
print(result)
```

#### 3. Transformers Backend (Local Inference)

```python
from infinity_parser2 import InfinityParser2

parser = InfinityParser2(
    model_name="infly/Infinity-Parser2-Pro",
    backend="transformers",
    device="cuda"
)

result = parser.parse("demo_data/demo.png")
print(result)
```

## Requirements

See `requirements.txt` for full dependency list.

## API Reference

### InfinityParser2 Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"infly/Infinity-Parser2-Pro"` | Model name on HuggingFace Hub or local path |
| `model_cache_dir` | `Optional[str]` | `None` | Custom cache directory for downloaded models (defaults to `~/.cache/infinity_parser2/`) |
| `backend` | `str` | `"vllm-engine"` | Inference backend: `"transformers"`, `"vllm-engine"`, or `"vllm-server"` |
| `tensor_parallel_size` | `Optional[int]` | `None` | Tensor parallel size for vLLM Engine (defaults to GPU count) |
| `device` | `str` | `"cuda"` | Device type, currently only `"cuda"` is supported |
| `api_url` | `str` | `"http://localhost:8000/v1/chat/completions"` | API URL for vLLM Server (used only with vllm-server backend) |
| `api_key` | `str` | `"EMPTY"` | API key for vLLM Server (used only with vllm-server backend) |
| `min_pixels` | `int` | `2048` | Minimum pixel count for input images (transformers backend only) |
| `max_pixels` | `int` | `16777216` | Maximum pixel count for input images (~4096x4096, transformers backend only) |
| `**kwargs` | - | - | Additional arguments passed to the backend |

### parse() Method Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_data` | `str \| List[str] \| PIL.Image.Image` | **Required** | File path(s), directory path, or PIL Image object |
| `prompt` | `str` | `"Please parse this document and extract all text content."` | Prompt text sent to the model |
| `batch_size` | `int` | `4` | Number of images to process per batch |
| `output_dir` | `Optional[str]` | `None` | If provided, results are saved to this directory |
| `**kwargs` | - | - | Additional arguments passed to the model |

### Automatic Model Download

Infinity-Parser2 features automatic model downloading and caching. When you first initialize the parser:

1. **First Use**: If the model is not found locally, it will automatically download from HuggingFace Hub and cache it at `~/.cache/infinity_parser2/`.

2. **Subsequent Uses**: The cached model will be detected and loaded directly without re-downloading.

```python
from infinity_parser2 import InfinityParser2

# First time: downloads model automatically if not cached
parser = InfinityParser2(model_name="infly/Infinity-Parser2-Pro")
# Output: [Infinity-Parser2] Model 'infly/Infinity-Parser2-Pro' not found locally.
#         Starting download to: ~/.cache/infinity_parser2/infly_Infinity-Parser2-Pro
#         ...

# Second time: uses cached model
parser = InfinityParser2(model_name="infly/Infinity-Parser2-Pro")
# Output: [Infinity-Parser2] Found cached model at: ~/.cache/infinity_parser2/infly_Infinity-Parser2-Pro
```

You can also customize the cache directory:

```python
parser = InfinityParser2(
    model_name="infly/Infinity-Parser2-Pro",
    model_cache_dir="/path/to/your/cache"
)
```

### Advanced Usage Examples

```python
from infinity_parser2 import InfinityParser2

# Use transformers backend (local inference)
parser = InfinityParser2(
    model_name="infly/Infinity-Parser2-Pro",
    backend="transformers",
    device="cuda",
    min_pixels=2048,       # Minimum pixel count
    max_pixels=16777216,   # Maximum pixel count (~4096x4096)
)

# Use custom prompt
result = parser.parse(
    "document.pdf",
    prompt="Please extract all table content from this document in Markdown format."
)

# Batch process multiple files
results = parser.parse(
    ["doc1.pdf", "doc2.png", "doc3.jpg"],
    batch_size=4  # Process 4 images per batch
)

# Save results to specified directory
saved_paths = parser.parse(
    "documents_folder",
    batch_size=8,
    output_dir="./parsed_output"
)

# Use vLLM Server backend (remote inference)
parser = InfinityParser2(
    backend="vllm-server",
    api_url="http://your-server:8000/v1/chat/completions",
    api_key="your-api-key"
)
```
