# olmOCR-Bench Evaluation

Evaluate **Infinity-Parser2** on [olmOCR-Bench](https://github.com/allenai/olmocr) — a
document-parsing benchmark that scores OCR/parsing output against per-document unit
tests across categories (arxiv math, tables, multi-column, old scans, etc.).

The pipeline has two stages:

1. **Inference** (`infer.py`) — run Infinity-Parser2 over the benchmark PDFs and write
   one Markdown file per document, laid out the way the benchmark harness expects.
2. **Scoring** (`olmocr.bench.benchmark`) — feed those Markdown files to the official
   olmOCR-Bench harness and get category-level scores.

---

## Layout

```
olmocr-bench/
├── infer.py                     # Stage 1: run inference, write per-doc Markdown
├── utils.py                     # Category-aware post-processing helpers
└── README.md
```

`utils.py` bundles the post-processing applied to raw model output:

| Function | Applied to categories | Purpose |
|---|---|---|
| `convert_latex_in_markdown` | `multi_column`, `tables` | Convert LaTeX to Unicode symbols |
| `apply_synonym_map` | `multi_column`, `tables` | Normalize common symbol variants |
| `latex_formula_normalization` | `arxiv_math`, `old_scans_math` | Merge/split adjacent LaTeX formulas |

---

## Environment and Data Preparation

- Infinity-Parser2 installed (`pip install -e .` from the repo root, or `pip install infinity-parser2`).
- A running **vLLM server** exposing the model (the inference script uses the
  `vllm-server` backend at `http://localhost:8000`).
- `hf` for downloading the benchmark dataset.

### Get the benchmark harness and data

```bash
# Clone the olmOCR repo and pin the evaluated commit
git clone https://github.com/allenai/olmocr.git
cd olmocr
git checkout f7cfe4c22098b154c76b6ec950d1c0a464eecf8d

# Download the olmOCR-Bench dataset (PDFs + ground-truth unit tests)
hf download --repo-type dataset allenai/olmOCR-bench --local-dir ./olmOCR-bench
```

The PDFs live under `olmOCR-bench/bench_data`. Stage 1 runs inference over these PDFs.
By default its Markdown output goes to `./Infinity-Parser2-results` next to the script,
which you then copy into `bench_data` before scoring; alternatively pass `OUTPUT_DIR`
pointing straight at `olmOCR-bench/bench_data/Infinity-Parser2-results` to skip the copy.

---

## Stage 1 — Inference

### Start the model server

```bash
vllm serve infly/Infinity-Parser2-Flash \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 65536 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --enable-prefix-caching \
    --served-model-name inf-mllm
```

`infer.py` is driven entirely by command-line arguments:

```bash
python infer.py PDF_DIR [OUTPUT_DIR] [BATCH_SIZE] [--model_name ...] [--api_url ...]
```

| Argument | Meaning |
|---|---|
| `PDF_DIR` | **(required, positional)** Directory of benchmark PDFs, searched recursively. **Sub-folder names are used as category labels**, so keep the benchmark's directory structure. |
| `OUTPUT_DIR` | (optional, positional) Where Markdown + `inference.jsonl` are written (default: `./Infinity-Parser2-results` next to the script). Leave unset and copy into `bench_data` afterwards, or pass `olmOCR-bench/bench_data/Infinity-Parser2-results` to write there directly. |
| `BATCH_SIZE` | (optional, positional) PDFs handed to the model per batch (default: `4`). |
| `--model_name` | Served model name; must match `--served-model-name` on the vLLM server (default: `inf-mllm`). |
| `--api_url` | vLLM chat-completions endpoint; must match your running server (default: `http://localhost:8000/v1/chat/completions`). |

Then run:

```bash
cd INF-MLLM/Infinity-Parser2/evaluation/olmocr-bench
python infer.py olmOCR-bench/bench_data
```

**Output**

- `Infinity-Parser2-results/inference.jsonl` — raw results, one JSON line per PDF
  (`{"pdf": ..., "markdown": ...}`). Used as a **resumable checkpoint**: rerunning
  `infer.py` skips PDFs already present here.
- `Infinity-Parser2-results/<category>/<doc>_pg1_repeat1.md` — the post-processed
  Markdown the benchmark harness scores. The `_pg1_repeat1` suffix is the filename
  convention olmOCR-Bench expects.

The run is crash-safe: each result is flushed immediately, and batches that fail fall
back to one-by-one inference so a single bad PDF cannot take down the whole batch.

---

## Stage 2 — Scoring with olmOCR-Bench

The harness expects each candidate's Markdown under
`bench_data/<candidate_name>/<category>/...`. If you ran Stage 1 with the default
`OUTPUT_DIR`, copy the results into `bench_data` first (skip this if you already pointed
`OUTPUT_DIR` at `bench_data`):

```bash
cp -r ./Infinity-Parser2-results /path/to/olmOCR-bench/bench_data/
```

Then score:

```bash
python -m olmocr.bench.benchmark \
    --dir ./olmOCR-bench/bench_data \
    --candidate Infinity-Parser2-results
```

`--candidate` is the results folder name inside `bench_data` (here
`Infinity-Parser2-results`). The harness prints per-category and overall scores.

---



## Notes

- **Resuming:** delete `Infinity-Parser2-results/inference.jsonl` to force a full
  re-inference; otherwise completed PDFs are skipped.
- **Categories** are derived from the PDF's parent folder name. Post-processing in
  `utils.py` is keyed on these names, so renaming category folders will silently
  disable the corresponding post-processing.
- **Pinned commit `f7cfe4c22098b154c76b6ec950d1c0a464eecf8d`** keeps scoring reproducible; newer olmOCR revisions may
  change tests or metrics.
