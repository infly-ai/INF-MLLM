import json
import re
from pathlib import Path
from typing import Union

from PIL import Image


# ---------------------------------------------------------------------------
# JSON extraction & cleanup
# ---------------------------------------------------------------------------

def extract_json_content(text: str) -> str:
    """Extract the JSON block from a markdown-wrapped LLM response."""
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    partial = re.search(r"```json\n(.*)", text, re.DOTALL)
    if partial:
        return partial.group(1).strip()
    return text


def truncate_last_incomplete_element(text: str) -> tuple[str, bool]:
    """
    Truncate the response at the last complete dict entry so the JSON is always parseable.
    Returns (cleaned_text, was_truncated).
    """
    needs_truncation = len(text) > 50_000 or not text.rstrip().endswith("]")

    if not needs_truncation:
        return text, False

    if text.count('{"bbox":') <= 1:
        return text, False

    last_bbox_pos = text.rfind('{"bbox":')
    truncated = text[:last_bbox_pos].rstrip()
    if truncated.endswith(","):
        truncated = truncated[:-1] + "]"
    return truncated, True


# ---------------------------------------------------------------------------
# Coordinate normalisation
# ---------------------------------------------------------------------------

def obtain_origin_hw(image: Union[str, Path, Image.Image]) -> tuple[int, int]:
    """
    Return (height, width) of the image.
    Accepts a file path (str/Path) or a PIL Image object.
    """
    if isinstance(image, Image.Image):
        w, h = image.size
        return h, w  # (height, width)
    try:
        img = Image.open(image).convert("RGB")
        w, h = img.size
        return h, w  # (height, width)
    except Exception:
        return 1000, 1000


def restore_abs_bbox_coordinates(ans: str, origin_h: float, origin_w: float) -> str:
    """Convert normalised [0-1000] bboxes back to pixel coordinates."""
    try:
        data = json.loads(ans)
    except json.JSONDecodeError:
        return ans

    valid = True
    for item in data:
        for key in item:
            if "bbox" not in key:
                continue
            bbox = item[key]
            if len(bbox) == 4 and all(isinstance(c, (int, float)) for c in bbox):
                x1, y1, x2, y2 = bbox
                item[key] = [
                    int(x1 / 1000.0 * origin_w),
                    int(y1 / 1000.0 * origin_h),
                    int(x2 / 1000.0 * origin_w),
                    int(y2 / 1000.0 * origin_h),
                ]
            else:
                valid = False

    return json.dumps(data, ensure_ascii=False) if valid else ans


# ---------------------------------------------------------------------------
# JSON → Markdown
# ---------------------------------------------------------------------------

def convert_json_to_markdown(ans: str, keep_header_footer: bool = False) -> str:
    """Convert the layout JSON list into a markdown string."""
    try:
        items = json.loads(ans)
        if not isinstance(items, list):
            return ans
        lines = []
        for sub in items:
            if "text" not in sub or not sub["text"]:
                continue
            if keep_header_footer:
                lines.append(sub["text"])
            else:
                if sub.get("category") not in ("header", "footer", "page_footnote"):
                    lines.append(sub["text"])
        return "\n\n".join(lines) if lines else ans
    except Exception:
        return ans


# ---------------------------------------------------------------------------
# DOC2JSON postprocess
# ---------------------------------------------------------------------------

def postprocess_doc2json_result(
    raw_text: str,
    image: Union[str, Path, Image.Image],
) -> str:
    """
    Postprocess raw LLM output for DOC2JSON mode:
      1. Extract JSON block from markdown-wrapped response
      2. Truncate last incomplete element for parseable JSON
      3. Restore normalised [0-1000] bboxes to pixel coordinates
    """
    text = extract_json_content(raw_text)
    text, _ = truncate_last_incomplete_element(text)
    origin_h, origin_w = obtain_origin_hw(image)
    return restore_abs_bbox_coordinates(text, origin_h, origin_w)


def postprocess_doc2json_batch(
    batch_results: list[str],
    batch_entries: list[tuple[int, Union[str, Path, Image.Image]]],
) -> list[str]:
    """Postprocess a batch of raw LLM outputs for DOC2JSON mode."""
    return [postprocess_doc2json_result(batch_results[i], batch_entries[i][1]) for i in range(len(batch_entries))]
