# Infinity-Parser2

<p align="center">
    <img src="https://raw.githubusercontent.com/infly-ai/INF-MLLM/main/Infinity-Parser2/assets/logo.png" width="400"/>
<p>

<p align="center">
🤗 <a href="https://huggingface.co/infly/Infinity-Parser2-Pro">Infinity-Parser2-Pro</a> |
🤗 <a href="https://huggingface.co/infly/Infinity-Parser2-Flash">Infinity-Parser2-Flash</a> |
📊 <a href="https://huggingface.co/datasets/infly/Infinity-Doc2-5M">Dataset</a> |
📄 <a>Paper (coming soon...)</a> |
🚀 <a href="https://huggingface.co/spaces/infly/Infinity-Parser2-Demo">Demo</a>
</p>

## Introduction

We are excited to release Infinity-Parser2, our latest flagship document understanding model. We offer two distinct variants to address diverse deployment constraints: Infinity-Parser2-Pro, optimized for maximum accuracy in precision-critical tasks, achieves state-of-the-art results on olmOCR-Bench (87.6%) and ParseBench (74.3%), surpassing frontier models including DeepSeek-OCR-2, PaddleOCR-VL-1.5, and MinerU-2.5. Infinity-Parser2-Flash, engineered for low-latency inference, delivers a 3.68x speedup over our previous Infinity-Parser-7B model. With significant upgrades to both our data engine and multi-task reinforcement learning approach, the model consolidates robust multi-modal parsing capabilities into a unified architecture, unlocking brand-new zero-shot capabilities across a wide range of real-world business scenarios.

<p align="center">
    <img src="https://raw.githubusercontent.com/infly-ai/INF-MLLM/main/Infinity-Parser2/assets/newspaper_1.png" width="1200"/>
</p>

### Key Features

- **Upgraded Data Engine**: We have comprehensively enhanced our synthetic data engine to support both fixed-layout and flexible-layout document formats. By curating nearly 5 million diverse document parsing samples across a wide range of layouts, combined with a dynamic adaptive sampling strategy, we ensure highly balanced and robust multi-task learning across various document types.
- **Multi-Task Reinforcement Learning**: We designed a novel verifiable reward system to support Joint Reinforcement Learning (RL), enabling seamless and simultaneous co-optimization of multiple complex tasks, including document parsing, element parsing, chart parsing, chemical formula parsing, document vqa, and general multimodal understanding.
- **Breakthrough Parsing Performance**: Infinity-Parser2-Pro substantially outperforms our previous 7B model, achieving 87.6% on olmOCR-Bench and 74.3% on ParseBench, surpassing frontier models such as DeepSeek-OCR-2, PaddleOCR-VL, and MinerU-2.5.
- **Inference Acceleration**: Infinity-Parser2-Flash delivers significantly higher efficiency than Infinity-Parser-7B, with inference throughput increased by 3.68x (from 441 to 1,624 tokens/sec), reducing both deployment latency and costs.

## Visual Parsing Examples

<table align="center">
  <thead>
    <tr>
      <th align="center">Category</th>
      <th align="center">Visualization</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>A-Share Financial Report</b></td>
      <td align="center"><img src="https://raw.githubusercontent.com/infly-ai/INF-MLLM/main/Infinity-Parser2/assets/a_stock.png" width="800"/></td>
    </tr>
    <tr>
      <td align="center"><b>Multi-Column Layout</b></td>
      <td align="center"><img src="https://raw.githubusercontent.com/infly-ai/INF-MLLM/main/Infinity-Parser2/assets/muti_column.png" width="800"/></td>
    </tr>
    <tr>
      <td align="center"><b>Historical Newspaper</b></td>
      <td align="center"><img src="https://raw.githubusercontent.com/infly-ai/INF-MLLM/main/Infinity-Parser2/assets/newspaper_2.png" width="800"/></td>
    </tr>
    <tr>
      <td align="center"><b>US Stock Financial Report</b></td>
      <td align="center"><img src="https://raw.githubusercontent.com/infly-ai/INF-MLLM/main/Infinity-Parser2/assets/us_stock.png" width="800"/></td>
    </tr>
    <tr>
      <td align="center"><b>Academic Paper (arXiv)</b></td>
      <td align="center"><img src="https://raw.githubusercontent.com/infly-ai/INF-MLLM/main/Infinity-Parser2/assets/arxiv.png" width="800"/></td>
    </tr>
    <tr>
      <td align="center"><b>Magazine Page</b></td>
      <td align="center"><img src="https://raw.githubusercontent.com/infly-ai/INF-MLLM/main/Infinity-Parser2/assets/magazine.png" width="800"/></td>
    </tr>
    <tr>
      <td align="center"><b>Scanned Mathematics</b></td>
      <td align="center"><img src="https://raw.githubusercontent.com/infly-ai/INF-MLLM/main/Infinity-Parser2/assets/old_scan_math.png" width="800"/></td>
    </tr>
  </tbody>
</table>

## Performance

<p align="left">
    <img src="https://raw.githubusercontent.com/infly-ai/INF-MLLM/main/Infinity-Parser2/assets/olmocr_bench_perf.png" width="1200"/>
<p>

<p align="left">
    <img src="https://raw.githubusercontent.com/infly-ai/INF-MLLM/main/Infinity-Parser2/assets/parsebench_perf.png" width="1200"/>
<p>

<table align="center">
  <thead>
    <tr>
      <th>Task</th>
      <th>Infinity-Parser2-Pro</th>
      <th>Infinity-Parser2-Flash</th>
      <th>PaddleOCR-VL-1.5</th>
      <th>DeepSeek-OCR-2</th>
      <th>MinerU-2.5</th>
      <th>Gemini-3-Pro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan=7><b>Document Parsing</b></td>
    </tr>
    <tr>
      <td>olmOCR-bench</td>
      <td><b>87.6</b></td>
      <td>86.0</td>
      <td>80.0†</td>
      <td>76.3</td>
      <td>75.2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ParseBench</td>
      <td><b>74.3</b></td>
      <td>72.2</td>
      <td>40.9†</td>
      <td>41.2</td>
      <td>45.9</td>
      <td>69.1‡</td>
    </tr>
    <tr>
      <td>OmniDocBench-v1.6</td>
      <td>93.95</td>
      <td>91.98</td>
      <td><b>94.87</b></td>
      <td>90.17</td>
      <td>92.98</td>
      <td>92.85</td>
    </tr>
    <tr>
      <td colspan=7><b>Layout Analysis (mIoU)</b></td>
    </tr>
    <tr>
      <td>DocLayNet</td>
      <td>64.93*</td>
      <td>64.97*</td>
      <td><b>71.05*</b></td>
      <td>45.62*</td>
      <td>67.74*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>D4LA</td>
      <td><b>52.41*</b></td>
      <td>46.05*</td>
      <td>50.21*</td>
      <td>33.03*</td>
      <td>51.62*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>OmniDocBench-v1.5-Layout</td>
      <td>74.56*</td>
      <td>73.07*</td>
      <td>74.80*</td>
      <td>55.28*</td>
      <td><b>76.28*</b></td>
      <td>-</td>
    </tr>
    <tr>
      <td colspan=7><b>Element Parsing</b></td>
    </tr>
    <tr>
      <td>OmniDocBench-v1.5-TextBlock</td>
      <td>93.66</td>
      <td>93.53</td>
      <td><b>94.97*</b></td>
      <td>84.13*</td>
      <td>86.00</td>
      <td>-</td>
    </tr>
    <tr>
      <td>PubTabNet (val)</td>
      <td><b>94.76</b></td>
      <td>92.41</td>
      <td>84.60</td>
      <td>89.53*</td>
      <td>89.07</td>
      <td>91.40</td>
    </tr>
    <tr>
      <td>UniMERNet</td>
      <td><b>97.7</b></td>
      <td>96.5</td>
      <td>95.8*</td>
      <td>79.8*</td>
      <td>96.5</td>
      <td>96.4</td>
    </tr>
    <tr>
      <td colspan=7><b>Chart Parsing</b></td>
    </tr>
    <tr>
      <td>Chart2Table</td>
      <td>80.45</td>
      <td>80.49</td>
      <td><b>86.2*</b></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Chart2Json</td>
      <td><b>73.69</b></td>
      <td>67.66</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td colspan=7><b>Chemical Formula Parsing</b></td>
    </tr>
    <tr>
      <td>CoSyn_Chemical</td>
      <td><b>71.48</b></td>
      <td>62.08</td>
      <td>-</td>
      <td>52.16*</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td colspan=7><b>Document VQA</b></td>
    </tr>
    <tr>
      <td>DocVQA (val)</td>
      <td><b>96.43</b></td>
      <td>93.16</td>
      <td>-</td>
      <td>43.42*</td>
      <td>-</td>
      <td>93.68*</td>
    </tr>
    <tr>
      <td>InfoVQA (val)</td>
      <td><b>86.26</b></td>
      <td>75.94</td>
      <td>-</td>
      <td>22.07*</td>
      <td>-</td>
      <td>85.24*</td>
    </tr>
    <tr>
      <td colspan=7><b>General Multimodal Understanding</b></td>
    </tr>
    <tr>
      <td>AI2D</td>
      <td>88.89</td>
      <td>79.53</td>
      <td>-</td>
      <td>37.66*</td>
      <td>-</td>
      <td><b>91.87*</b></td>
    </tr>
    <tr>
      <td>MathVista (testmini)</td>
      <td>71.4</td>
      <td>59.5</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td><b>81.8*</b></td>
    </tr>
    <tr>
      <td>MMBench-EN (dev)</td>
      <td>87.54</td>
      <td>77.92</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td><b>90.29*</b></td>
    </tr>
    <tr>
      <td>MMBench-CN (dev)</td>
      <td>86.43</td>
      <td>75.77</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td><b>90.98*</b></td>
    </tr>
    <tr>
      <td>MMMU (val)</td>
      <td><b>61.89</b></td>
      <td>45.89</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>56.00*</td>
    </tr>
    <tr>
      <td>MMStar</td>
      <td>69.66</td>
      <td>57.13</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td><b>83.78*</b></td>
    </tr>
    <tr>
      <td>OCRBench</td>
      <td>86.20</td>
      <td>81.60</td>
      <td>-</td>
      <td>47.20*</td>
      <td>-</td>
      <td><b>89.30*</b></td>
    </tr>
  </tbody>
</table>

Note: '*' denotes results evaluated using our internal evaluation tools. Results marked with '†' are from PaddleOCR-VL. '‡' denotes results from the Gemini-3.1-Pro.

## Quick Start

### 1. Minimal "Hello World" (Native Transformers)

If you are looking for a minimal script to parse a single image to Markdown using the native `transformers` library, here is a simple snippet:

```python
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load the model and processor
model = AutoModelForImageTextToText.from_pretrained(
    "infly/Infinity-Parser2-Pro",
    torch_dtype="float16",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("infly/Infinity-Parser2-Pro")

# Build the messages for the model
pil_image = Image.open("demo_data/demo.png").convert("RGB")
min_pixels = 2048  # 32 * 64
max_pixels = 16777216  # 4096 * 4096
prompt = """
- Extract layout information from the provided PDF image.
- For each layout element, output its bbox, category, and the text content within the bbox.
- Bbox format: [x1, y1, x2, y2].
- Allowed layout categories: ['header', 'title', 'text', 'figure', 'table', 'formula', 'figure_caption', 'table_caption', 'formula_caption', 'figure_footnote', 'table_footnote', 'page_footnote', 'footer'].
- Text extraction and formatting:
  1) For 'figure', the text field must be an empty string.
  2) For 'formula', format text as LaTeX.
  3) For 'table', format text as HTML.
  4) For all other categories (e.g., text, title), format text as Markdown.
- The output text must be exactly the original text from the image, with no translation or rewriting.
- Sort all layout elements in human reading order.
- Final output must be a single JSON object.
"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": pil_image,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
            {"type": "text", "text": prompt},
        ],
    }
]

chat_template_kwargs = {"enable_thinking": False}

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
)
image_inputs, _ = process_vision_info(messages, image_patch_size=16)

inputs = processor(
    text=text,
    images=image_inputs,
    do_resize=False,
    padding=True,
    return_tensors="pt",
)

# Move all tensors to the same device as the model
inputs = {
    k: v.to(model.device) if isinstance(v, torch.Tensor) else v
    for k, v in inputs.items()
}

# Generate the response
generated_ids = model.generate(
    **inputs,
    max_new_tokens=32768,
    temperature=0.0,
    top_p=1.0,
)

# Strip input tokens, keeping only the newly generated response
generated_ids_trimmed = [
    out_ids[len(in_ids) :]
    for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

### 2. Advanced Pipeline (infinity_parser2)

For bulk processing, advanced features, or an end-to-end PDF parsing pipeline, we recommend using our infinity_parser2 wrapper.

#### Pre-requisites

```bash
# Create a Conda environment (Optional)
conda create -n infinity_parser2 python=3.12
conda activate infinity_parser2

# Install PyTorch (CUDA). Find the proper version at https://pytorch.org/get-started/previous-versions based on your CUDA version.
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# Install FlashAttention (FlashAttention-2 is recommended by default)
# Standard install (compiles from source, ~10-30 min):
pip install flash-attn==2.8.3 --no-build-isolation
# Faster install: download wheel from https://github.com/Dao-AILab/flash-attention/releases. Then run: pip install /path/to/<wheel_filename>.whl
# For Hopper GPUs (e.g. H100, H800), we recommend FlashAttention-3 instead. See: https://github.com/Dao-AILab/flash-attention
# NOTE: The code will prioritize detecting FlashAttention-3. If not found, it falls back to FlashAttention-2.

# Install vLLM
# NOTE: you may need to run the command below to resolve triton and numpy conflicts before installing vllm.
# pip uninstall -y pytorch-triton opencv-python opencv-python-headless numpy && rm -rf "$(python -c 'import site; print(site.getsitepackages()[0])')/cv2"
pip install vllm==0.17.1
```

#### Install infinity_parser2

Install from PyPI

```bash
pip install infinity_parser2
```

Install from source code

```bash
git clone https://github.com/infly-ai/INF-MLLM.git
cd INF-MLLM/Infinity-Parser2
pip install -e .
```

#### Usage

##### Command Line

The `parser` command is the fastest way to get started.

```bash
# NOTE: The Infinity-Parser2 model will be automatically downloaded on the first run.

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

##### Python API

```python
# NOTE: The Infinity-Parser2 model will be automatically downloaded on the first run.

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
                      custom_prompt="Please transform the document's contents into Markdown format.")

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

## Limitations

Infinity-Parser2 has several known limitations to consider. It primarily supports English and Chinese documents, and performance degrades when processing multilingual content. Accuracy may also be reduced when parsing charts with complex layouts, as well as documents containing multi-oriented elements such as table rotated at varying angles. Additionally, the model does not capture fine-grained text formatting (e.g., bold, italic, strikethrough) and exhibits suboptimal multimodal instruction-following capability, meaning it may not always reliably follow complex multi-step visual instructions.

## Acknowledgments

We would like to thank [Qwen3.5](https://github.com/QwenLM/Qwen3.5), [ms-swift](https://github.com/modelscope/ms-swift), [VeRL](https://github.com/verl-project/verl), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), [olmocr](https://huggingface.co/datasets/allenai/olmOCR-bench), [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR), [MinerU](https://github.com/opendatalab/MinerU), [dots.ocr](https://github.com/rednote-hilab/dots.ocr), [Chandra-OCR-2](https://github.com/datalab-to/chandra) for providing dataset, code and models.
