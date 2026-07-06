import json
import os
import traceback
from pathlib import Path

from infinity_parser2 import InfinityParser2

from utils import (
    convert_latex_in_markdown,
    latex_formula_normalization,
    apply_synonym_map,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDF_DIR = "olm_ocr/pdfs" # Write real pdfpath
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "Infinity-Parser2-results")
# Raw inference results, one JSON line per PDF: {"pdf": ..., "markdown": ...}
JSONL_PATH = os.path.join(OUTPUT_DIR, "inference.jsonl")
BATCH_SIZE = 4
parser = InfinityParser2(
    model_name="inf-mllm",
    backend="vllm-server",
    api_url="http://localhost:8000/v1/chat/completions",
    api_key="EMPTY",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Post-processing + md writing (category-aware)
# ---------------------------------------------------------------------------
def postprocess(text, parent_name):
    """Apply category-specific post-processing to the parsed markdown."""
    # Post-processing for the multi_column and tables categories
    if parent_name in ("multi_column", "tables"):
        text = convert_latex_in_markdown(text)  # convert LaTeX to Unicode
        text = apply_synonym_map(text)  # apply synonym map
    # Post-processing for the arxiv_math and old_scans_math categories
    if parent_name in ("arxiv_math", "old_scans_math"):
        text = latex_formula_normalization(text, parent_name)  # normalize LaTeX formulas
    return text


def write_str_to_md(output_path, src_path, content):
    """Write `content` into output_path/{parent_folder}/{src_filename}_pg1_repeat1.md."""
    parent_name = os.path.basename(os.path.dirname(src_path))
    file_name = os.path.basename(src_path).rsplit(".", 1)[0] + "_pg1_repeat1.md"
    out_dir = os.path.join(output_path, parent_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


# ---------------------------------------------------------------------------
# Collect all PDFs (recursively, preserving sub-folder / category structure)
# ---------------------------------------------------------------------------
pdf_paths = sorted(Path(PDF_DIR).rglob("*.pdf"))
print(f"[infer] found {len(pdf_paths)} PDFs under {PDF_DIR}")

# ---------------------------------------------------------------------------
# Checkpoint: skip PDFs already present in the jsonl
# ---------------------------------------------------------------------------
done_pdfs = set()
if os.path.exists(JSONL_PATH):
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done_pdfs.add(json.loads(line)["pdf"])
            except Exception:
                # tolerate a partially-written trailing line from a crash
                pass
print(f"[infer] {len(done_pdfs)} already done, resuming")

done, failed = 0, 0

# Filter out already-completed PDFs, only run batch inference on the rest
pending = [p for p in pdf_paths if str(p) not in done_pdfs]
skipped = len(pdf_paths) - len(pending)
print(f"[infer] {skipped} skipped, {len(pending)} to parse (batch_size={BATCH_SIZE})")


def handle_result(out, src_path, text):
    """Write one jsonl line + post-process into a md file."""
    text = text if text is not None else ""
    record = {"pdf": src_path, "markdown": text}
    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    out.flush()  # persist each line -> crash-safe checkpoint
    parent_name = os.path.basename(os.path.dirname(src_path))
    write_str_to_md(OUTPUT_DIR, src_path, postprocess(text, parent_name))


# Append mode so previous results are preserved across resumes.
with open(JSONL_PATH, "a", encoding="utf-8") as out:
    for start in range(0, len(pending), BATCH_SIZE):
        batch = [str(p) for p in pending[start : start + BATCH_SIZE]]
        print(
            f"[infer] batch {start // BATCH_SIZE + 1}: "
            f"{start + 1}-{start + len(batch)}/{len(pending)}"
        )
        try:
            # Hand the whole batch to the model at once; batch_size lets the
            # backend pool inference across PDFs
            results = parser.parse(
                batch,
                task_type="doc2json",
                output_format="md",
                batch_size=BATCH_SIZE,
            )
            if isinstance(results, str):  # parse returns str for a single-element list
                results = [results]
            for src_path, text in zip(batch, results):
                handle_result(out, src_path, text)
                done += 1
        except Exception:
            # If the whole batch fails, fall back to one-by-one so a single
            # bad PDF doesn't take down the entire batch
            print(f"[infer] batch failed, retry one-by-one:\n{traceback.format_exc()}")
            for src_path in batch:
                try:
                    text = parser.parse(
                        src_path,
                        task_type="doc2json",
                        output_format="md",
                        batch_size=BATCH_SIZE,
                    )
                    handle_result(out, src_path, text)
                    done += 1
                except Exception:
                    failed += 1
                    print(f"[infer] FAILED on {src_path}:\n{traceback.format_exc()}")

print(f"[infer] done. parsed={done} skipped={skipped} failed={failed} total={len(pdf_paths)}")
print(f"[infer] jsonl -> {JSONL_PATH}")
print(f"[infer] md    -> {OUTPUT_DIR}/<category>/*.md")
