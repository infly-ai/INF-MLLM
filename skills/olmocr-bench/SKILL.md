---
name: olmocr-bench
description: Reproduce Infinity Parser scores on olmOCR-Bench. Parses the benchmark PDFs through the Infinity Parser API, applies the category-specific post-processing, and scores the output with the official olmOCR-Bench harness at a pinned commit.
---

# olmOCR-Bench Reproduction Skill

Reproduce [olmOCR-Bench](https://github.com/allenai/olmocr/tree/main/olmocr/bench) scores for Infinity Parser in three steps: set up, parse, score. Requires Python 3.11+, git, and an Infinity Parser API key. Run every command from this skill directory.

## 1. Set up

Copy `.env.example` to `.env`, then set `INF_API_URL` and `INF_API_KEY`. Environment variables with the same names override the file.

Install the official harness and download the dataset, both pinned to the evaluated revisions:

```bash
git clone https://github.com/allenai/olmocr.git
git -C olmocr checkout f7cfe4c22098b154c76b6ec950d1c0a464eecf8d
pip install -e './olmocr[bench]'
pip install -U huggingface_hub  # provides the hf CLI
pip install requests  # used by scripts/infer.py
playwright install chromium
hf download allenai/olmOCR-bench --repo-type dataset \
  --revision 54a96a6fb6a2bd3b297e59869491db4d3625b711 \
  --local-dir olmocr/olmOCR-bench
```

If the download times out, prefix the `hf download` command with `HF_ENDPOINT=https://hf-mirror.com` to use the mirror.

## 2. Parse

```bash
python3 scripts/infer.py \
  --pdf-dir olmocr/olmOCR-bench/bench_data/pdfs \
  --output-dir olmocr/olmOCR-bench/bench_data/infinity_parser \
  --tier flash \
  --workers 8
```

The script sends each PDF to `/v1/chat/completions` as a Base64 `file` content part with model `infinity-parser-<tier>` and `parser_options` set to page 1, `keep_header_footer: false`, and `parse_chart: false`. It saves each raw response under `raw/`, and writes `<category>/<name>_pg1_repeat1.md` in the layout the harness expects. Category-specific post-processing in `scripts/postprocess.py` is applied to the Markdown:

| Category | Post-processing |
| --- | --- |
| `multi_column`, `tables` | Convert LaTeX to Unicode and normalize symbol variants |
| `arxiv_math`, `old_scans_math` | Merge adjacent formulas; split `aligned` blocks in `old_scans_math` |

Re-running the command skips PDFs that already have a raw response and rebuilds all Markdown. If some PDFs fail, the script exits non-zero; run it again before scoring, because the harness gives a candidate a score of 0 if any file is missing.

`--tier` defaults to `flash`. To compare tiers, write each tier to a different `--output-dir` folder inside `bench_data`.

## 3. Score

```bash
cd olmocr
python -m olmocr.bench.benchmark \
  --dir ./olmOCR-bench/bench_data \
  --candidate infinity_parser
```

`--candidate` is the name of the output folder inside `bench_data`. The harness prints per-category scores and the overall score with a 95% confidence interval.

## Notes

- Do not rename the category folders under `pdfs/`; the post-processing and the harness both use the folder names.
- The pinned harness commit and dataset revision keep scores reproducible. Newer revisions may change tests or metrics.

## Reference scores

Scores from running the commands above with `--tier flash`. A successful reproduction lands within 1 point of each value.

| Category | Score (%) | Passed / Tests |
| --- | --- | --- |
| `arxiv_math` | 88.1 | 2579 / 2927 |
| `baseline` | 99.9 | 1392 / 1394 |
| `headers_footers` | 91.8 | 698 / 760 |
| `long_tiny_text` | 89.8 | 397 / 442 |
| `multi_column` | 83.4 | 737 / 884 |
| `old_scans` | 52.1 | 274 / 526 |
| `old_scans_math` | 89.5 | 410 / 458 |
| `table_tests` | 87.1 | 890 / 1022 |
| **Overall** | **85.2** (95% CI: 84.3–86.1) | 8413 tests |

The overall score is the average of the eight category scores, not the pooled pass rate.
