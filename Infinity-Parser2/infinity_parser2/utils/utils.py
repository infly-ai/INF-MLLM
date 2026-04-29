import json
import re
from pathlib import Path
from typing import Union

from PIL import Image, ImageDraw, ImageFont

CATEGORY_COLORS = {
    "header": (255, 107, 107),
    "title": (78, 205, 196),
    "text": (69, 183, 209),
    "figure": (150, 206, 180),
    "table": (255, 234, 167),
    "formula": (162, 155, 254),
    "figure_caption": (253, 203, 110),
    "table_caption": (0, 184, 148),
    "formula_caption": (108, 92, 231),
    "figure_footnote": (214, 48, 49),
    "table_footnote": (9, 132, 227),
    "page_footnote": (253, 121, 168),
    "footer": (116, 185, 255),
}
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
    needs_truncation = len(text) > 65, 536 or not text.rstrip().endswith("]")

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
    output_format: str = "json",
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
    text = restore_abs_bbox_coordinates(text, origin_h, origin_w)
    if output_format == "md":
        text = convert_json_to_markdown(text)
    return text


# ---------------------------------------------------------------------------
# Markdown cleanup
# ---------------------------------------------------------------------------


def postprocess_doc2md_result(text: str) -> str:
    """Remove markdown code block fences from text.

    Removes ```markdown\n and ``` (or similar) fences from the beginning
    and end of text if present.

    Args:
        text: Input text that may contain markdown code block fences.

    Returns:
        Text with code block fences removed.
    """
    text = text.strip()
    text = re.sub(r"^```markdown\s*\n?", "", text)
    text = re.sub(r"^```\s*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _get_font(size: int = 14):
    """Try to load a decent font, fall back to default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_bboxes_on_image(
    image_path: Union[str, Path],
    json_text: str,
) -> Image.Image | None:
    """Draw category-colored bounding boxes on a copy of the image.

    json_text is expected to be already post-processed (single-page flat list)
    with pixel-coordinate bboxes.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except (IOError, OSError):
        return None

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list) or len(data) == 0:
        return img

    draw = ImageDraw.Draw(img)
    font = _get_font(16)

    for item in data:
        bbox = item.get("bbox", [])
        category = item.get("category", "unknown")
        if len(bbox) != 4:
            continue

        color = CATEGORY_COLORS.get(category, (200, 200, 200))
        x1, y1, x2, y2 = bbox

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = category
        try:
            tb = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(
                [tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2],
                fill=color,
            )
        except AttributeError:
            pass  # Compatible with low version Pillow.

        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)

    # Display the image size in the upper right corner.
    size_text = f"Size: {img.width}x{img.height}"
    try:
        tb = draw.textbbox((0, 0), size_text, font=font)
        text_width = tb[2] - tb[0]
        text_height = tb[3] - tb[1]

        margin = 10
        x = img.width - text_width - margin * 2
        y = margin

        draw.rectangle(
            [x, y, x + text_width + margin * 2, y + text_height + margin * 2],
            fill=(0, 0, 0, 180),
        )
        draw.text((x + margin, y + margin), size_text, fill=(255, 255, 255), font=font)
    except AttributeError:
        pass

    return img
