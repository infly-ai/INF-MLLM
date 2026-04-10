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

result = parser.parse("demo_data/demo.pdf")
print(result)
```

#### 3. Transformers Backend (Local Inference)

```python
from infinity_parser2 import InfinityParser2

parser = InfinityParser2(
    backend="transformers",
    device="cuda"
)

result = parser.parse("demo_data/demo.png")
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
