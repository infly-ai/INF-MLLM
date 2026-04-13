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

**Send inference requests:**

```python
from infinity_parser2 import InfinityParser2

parser = InfinityParser2(
    model_name="infly/Infinity-Parser2-Pro",
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
| `min_pixels` | `int` | `2048` | Minimum pixel count for input images |
| `max_pixels` | `int` | `16777216` | Maximum pixel count for input images (~4096x4096) |
| `torch_dtype` | `str` | `"bfloat16"` | Data type for model weights in transformers backend (`"float16"` or `"bfloat16"`) |
| `timeout` | `int` | `300` | Request timeout in seconds for vLLM Server backend |
| `**kwargs` | - | - | Additional arguments passed to the backend |

### parse() Method Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_data` | `str \| List[str] \| PIL.Image.Image` | **Required** | File path(s), directory path, or PIL Image object |
| `custom_prompt` | `Optional[str]` | `None` | Custom prompt text. If provided, uses this prompt directly and skips result post-processing |
| `task_type` | `str` | `"doc2json"` | Parsing mode: `"doc2json"` (layout to JSON) or `"doc2md"` (direct Markdown output). Set to `"custom"` when using `custom_prompt`. |
| `batch_size` | `int` | `4` | Number of images to process per batch |
| `output_dir` | `Optional[str]` | `None` | If provided, results are saved to this directory |
| `output_format` | `str` | `"md"` | Output format for DOC2JSON tasks: `"md"` (markdown) or `"json"` (raw JSON). Only `"md"` is supported for DOC2MD tasks or when custom prompt is provided. |
| `**kwargs` | - | - | Additional arguments passed to the model (e.g., `max_new_tokens`, `temperature`) |

### task_type Options

```python
from infinity_parser2 import SUPPORTED_TASK_TYPES

# Available task types
"doc2json"  # Extract layout to JSON with bbox coordinates
"doc2md"    # Directly convert to Markdown format
"custom"    # Use custom_prompt for parsing
```

| task_type | Default Output | Description |
|-----------|----------------|-------------|
| `doc2json` | Markdown string | Extracts layout elements with bounding box coordinates, category, and text content. By default (`output_format="md"`), converts JSON to Markdown. Set `output_format="json"` to return raw JSON. When `output_dir` is set, saves either `result.md` or `result.json` based on `output_format`. |
| `doc2md` | Markdown string | Directly converts document content to Markdown format. Returns plain Markdown text. |
| `custom` | Varies | Uses `custom_prompt` for parsing. Returns raw model output (post-processing is skipped). Only `output_format="md"` is supported. |

### Return Value

**Without output_dir (returns results directly):**
- **Single file**: Returns `str` — parsed result for the file.
- **Multiple files** (List): Returns `List[str]` — parsed results for all files.
- **Directory**: Returns `Dict[str, str]` — mapping from file path to parsed result.

**With output_dir (saves results to disk):**
- Returns `None` directly.
- Creates subdirectories for each input file (named by filename or UUID for PIL Images).
- For `doc2json`:
  - `output_format="md"`: each subdirectory contains `result.md` (markdown).
  - `output_format="json"`: each subdirectory contains `result.json` (raw JSON).
- For `doc2md` or custom_prompt: each subdirectory contains `result.md`.

### Automatic Model Download

Infinity-Parser2 features automatic model downloading and caching. When you first initialize the parser:

1. **First Use**: If the model is not found locally, it will automatically download from HuggingFace Hub and cache it at `~/.cache/infinity_parser2/`.

2. **Subsequent Uses**: The cached model will be detected and loaded directly without re-downloading.

3. **Local Path**: If `model_name` is a local path, it will be used directly without caching.

4. **Endpoint Fallback**: Automatically detects connectivity and falls back to HuggingFace mirror (`hf-mirror.com`) if needed.

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

# Default parsing (doc2json mode - returns Markdown by default)
parser = InfinityParser2(model_name="infly/Infinity-Parser2-Pro")
result = parser.parse("demo_data/demo.pdf")
# Returns Markdown (JSON is converted to Markdown via convert_json_to_markdown)

# doc2json mode with raw JSON output
result = parser.parse("demo_data/demo.pdf", output_format="json")
# Returns JSON string with layout elements: [{"bbox": [x1,y1,x2,y2], "category": "...", "text": "..."}]

# doc2md mode (direct Markdown output)
result = parser.parse("demo_data/demo.pdf", task_type="doc2md")
# Returns Markdown string directly

# Custom prompt (skips result post-processing)
result = parser.parse(
    "demo_data/demo.pdf",
    custom_prompt="Please transform the document's contents into Markdown format."
)

# Batch process multiple files
results = parser.parse(
    ["demo_data/demo.pdf", "demo_data/demo.png", "demo_data/demo.png"],
    batch_size=4  # Process 4 images per batch
)

# Save results to specified directory
# doc2json mode: saves result.md by default
parser.parse(
    "demo_data",
    task_type="doc2json",
    batch_size=8,
    output_dir="./parsed_output"
)
# Returns: None (results saved to ./parsed_output/{filename}/result.md)

# doc2json mode with JSON output: saves result.json
parser.parse(
    "demo_data",
    task_type="doc2json",
    batch_size=8,
    output_dir="./parsed_output",
    output_format="json"
)
# Returns: None (results saved to ./parsed_output/{filename}/result.json)

# doc2md mode: saves result.md for each file
parser.parse(
    "demo_data",
    task_type="doc2md",
    batch_size=8,
    output_dir="./parsed_output"
)
# Returns: None (results saved to ./parsed_output/{filename}/)

# Use transformers backend with custom dtype
parser = InfinityParser2(
    model_name="infly/Infinity-Parser2-Pro",
    backend="transformers",
    device="cuda",
    torch_dtype="float16",  # or "bfloat16"
    min_pixels=2048,
    max_pixels=16777216,
)

# Use vLLM Server backend (remote inference)
parser = InfinityParser2(
    model_name="infly/Infinity-Parser2-Pro",
    backend="vllm-server",
    api_url="http://your-server:8000/v1/chat/completions",
    api_key="your-api-key",
    timeout=300,
)

# Custom generation parameters
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
    convert_pdf_to_images,      # Convert PDF pages to PIL Images
    convert_json_to_markdown,    # Convert layout JSON to Markdown
    extract_json_content,        # Extract JSON from LLM response
    restore_abs_bbox_coordinates, # Convert normalized bboxes to pixel coordinates
    postprocess_doc2json_result,  # Full DOC2JSON post-processing
    postprocess_doc2md_result,    # Post-process DOC2MD result (remove code fences)
    get_files_from_directory,    # Get supported files from directory
    is_supported_file,           # Check if file type is supported
    save_results,               # Save parsing results to directory
    SUPPORTED_TASK_TYPES,       # List of supported task types
    ModelCache,                 # Model cache management class
    get_model_cache,            # Get global model cache instance
)

# Convert PDF to images
images = convert_pdf_to_images("document.pdf", dpi=300)
for page_img in images:
    print(page_img.size)

# Convert JSON to Markdown
markdown = convert_json_to_markdown(json_string)

# Restore absolute bbox coordinates
json_with_coords = restore_abs_bbox_coordinates(json_string, height, width)
```
