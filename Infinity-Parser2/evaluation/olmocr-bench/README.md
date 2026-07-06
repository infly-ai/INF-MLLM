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
├── Infinity-Parser2-results/    # Output: <category>/<doc>_pg1_repeat1.md  (+ inference.jsonl)
└── README.md
```

`utils.py` bundles the post-processing applied to raw model output:

| Function | Applied to categories | Purpose |
|---|---|---|
| `convert_latex_in_markdown` | `multi_column`, `tables` | Convert LaTeX to Unicode symbols |
| `apply_synonym_map` | `multi_column`, `tables` | Normalize common symbol variants |
| `latex_formula_normalization` | `arxiv_math`, `old_scans_math` | Merge/split adjacent LaTeX formulas |

---

## Prerequisites

- Infinity-Parser2 installed (`pip install -e .` from the repo root, or `pip install infinity-parser2`).
- A running **vLLM server** exposing the model (the inference script uses the
  `vllm-server` backend at `http://localhost:8000`).
- `huggingface-cli` for downloading the benchmark dataset.

### Start the model server

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
    --enable-prefix-caching \
    --served-model-name inf-mllm
```

---

## Stage 1 — Inference

Before running, open `infer.py` and check the config block at the top:

| Constant | Meaning |
|---|---|
| `PDF_DIR` | Directory of benchmark PDFs, searched recursively. **Sub-folder names are used as category labels**, so keep the benchmark's directory structure. |
| `OUTPUT_DIR` | Where Markdown + `inference.jsonl` are written (default: `./Infinity-Parser2-results`). |
| `BATCH_SIZE` | PDFs handed to the model per batch. |
| `api_url` / `model_name` | Must match your running vLLM server. |

Then run:

```bash
cd INF-MLLM/Infinity-Parser2/evaluation/olmocr-bench
python infer.py
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

### 1. Get the benchmark harness and data

```bash
# Clone the olmOCR repo and pin the evaluated commit
git clone https://github.com/allenai/olmocr.git
cd olmocr
git checkout 1e139a5

# Download the olmOCR-Bench dataset (PDFs + ground-truth unit tests)
huggingface-cli download --repo-type dataset --resume-download \
    allenai/olmOCR-bench --local-dir ./olmOCR-bench
```

> The PDFs under `olmOCR-bench/bench_data` are the same ones `infer.py` runs over —
> point `PDF_DIR` in Stage 1 at them (or at your local copy) so categories line up.

### 2. Drop the inference results into `bench_data`

The harness expects each candidate's Markdown to live under
`bench_data/<candidate_name>/<category>/...`. Copy the Stage-1 output in:

```bash
cp -r \
  /home/ma-user/work/renkexuan/07_codes/INF-MLLM/Infinity-Parser2/evaluation/olmocr-bench/Infinity-Parser2-results \
  ./olmOCR-bench/bench_data/
```

### 3. Run the benchmark

```bash
python -m olmocr.bench.benchmark \
    --dir ./olmOCR-bench/bench_data \
    --candidate Infinity-Parser2-results
```

`--candidate` is the folder name you copied into `bench_data` (here
`Infinity-Parser2-results`). The harness prints per-category and overall scores.

---

## End-to-end quick reference

```bash
# 0. Serve the model (separate terminal) — see "Start the model server" above
vllm serve infly/Infinity-Parser2-Pro --trust-remote-code --reasoning-parser qwen3 \
    --host 0.0.0.0 --port 8000 --tensor-parallel-size 2 ...

# 1. Inference
cd INF-MLLM/Infinity-Parser2/evaluation/olmocr-bench
python infer.py

# 2. Benchmark harness + data
git clone https://github.com/allenai/olmocr.git
cd olmocr && git checkout 1e139a5
huggingface-cli download --repo-type dataset --resume-download \
    allenai/olmOCR-bench --local-dir ./olmOCR-bench

# 3. Copy results in and score
cp -r ../Infinity-Parser2-results ./olmOCR-bench/bench_data/
python -m olmocr.bench.benchmark \
    --dir ./olmOCR-bench/bench_data \
    --candidate Infinity-Parser2-results
```

---

## Notes

- **Resuming:** delete `Infinity-Parser2-results/inference.jsonl` to force a full
  re-inference; otherwise completed PDFs are skipped.
- **Categories** are derived from the PDF's parent folder name. Post-processing in
  `utils.py` is keyed on these names, so renaming category folders will silently
  disable the corresponding post-processing.
- **Pinned commit `1e139a5`** keeps scoring reproducible; newer olmOCR revisions may
  change tests or metrics.
