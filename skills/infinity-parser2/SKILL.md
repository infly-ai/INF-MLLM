---
name: infinity-parser2
description: Parse PDFs, scanned documents, and document images (screenshots, invoices, paper documents, whiteboard photos) into structured layout JSON (labeled regions with bounding boxes) and Markdown via Infinity-Parser2 through an existing vLLM server. Use when the user wants to extract text, tables, formulas, or structured data from visual documents; mentions OCR, text recognition, or document parsing; or asks to parse, digitize, or extract content from a document. Requires a vLLM OpenAI-compatible API URL and API key.
---

# Infinity-Parser2 Skill

> **Prerequisite**: `pip install infinity_parser2`

Use only the `vllm-server` backend. The vLLM server must already be running.
The CLI reads the server URL and API key directly from the `INFINITY_PARSER2_*`
process environment variables.

## CLI

The `parser` command is available after installation. Set the environment
variables first:

```bash
export INFINITY_PARSER2_API_URL=<full /v1/chat/completions URL>
export INFINITY_PARSER2_API_KEY=<api-key>

parser /path/to/document.pdf \
  --backend vllm-server \
  --output-dir /path/to/output \
  --output-format md,json \
  --pages 5-7 \
  --model-name infinity-parser2-flash
```

Key flags:

- `-o/--output-dir <dir>` — write results to disk (`<dir>/<basename>/result.md|json`).
  Omit it and results print to stdout, which floods the model's context — prefer writing to disk.
- `--output-format md|json|md,json` — `md,json` keeps both files; default `md`.
- `--pages 1-3,5` — 1-based physical PDF pages (same syntax as the Python API).
- `--backend vllm-server` — required; the default backend is `vllm-engine`.
- Overrides if the env vars are not set:
  `--api-url <full /v1/chat/completions URL> --api-key <key>`.

## Python API

```python
import os

from infinity_parser2 import InfinityParser2

parser = InfinityParser2(
    model_name="infinity-parser2-flash",
    backend="vllm-server",
    api_url=os.environ["INFINITY_PARSER2_API_URL"],
    api_key=os.environ["INFINITY_PARSER2_API_KEY"],
)
parser.parse(
    "/path/to/document.pdf",
    output_dir="/path/to/output",
    output_format="md,json",
)
print("saved to /path/to/output")
```

## Tasks

- Default (Markdown + layout JSON): `parser.parse(path, output_dir="/path/to/output", output_format="md,json")`.
- Write only Markdown: `parser.parse(path, output_dir="/path/to/output", output_format="md")`.
- Write only layout JSON: `parser.parse(path, output_dir="/path/to/output", output_format="json")`.
- Convert directly to Markdown: `parser.parse(path, task_type="doc2md", output_dir="/path/to/output")`.
- Parse several paths: `parser.parse([path1, path2], output_dir="/path/to/output")`.

`output_format="json"` or `"md,json"` is only valid when `task_type="doc2json"`
(the default). `doc2md` and `custom` tasks only support `output_format="md"`;
passing `json`/`md,json` with them raises a `ValueError`.

Accept PDF files and supported document images. For a directory, pass its path directly to `parser.parse()`.When both `result.md` and `result.json` are needed, explicitly set `output_format="md,json"`. Each input gets its own subdirectory: `<output_dir>/<input-basename>/result.md` and `result.json`.

## PDF Pages

Pass `pages` to select physical PDF pages with 1-based numbering:

```python
parser.parse("/path/to/document.pdf", pages="1-3,5,8-10", output_dir="/path/to/output")
parser.parse("/path/to/document.pdf", pages=[1, 3, 5], output_dir="/path/to/output")
```

Omit `pages` to parse every page. The selection applies to each PDF in a multi-file call and is ignored for images.

Before running, verify that `INFINITY_PARSER2_API_URL` is exported in the
process environment. If the endpoint requires authentication, also verify
`INFINITY_PARSER2_API_KEY`. Never request, print, or expose API keys.
