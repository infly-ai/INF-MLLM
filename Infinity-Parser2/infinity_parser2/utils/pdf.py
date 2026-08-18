"""PDF to image conversion utility."""

import io
from typing import List, Optional, Union

from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "PyMuPDF is required for PDF rendering. Install it with: pip install pymupdf"
    )


def parse_pages_spec(pages: Union[str, List[int]], page_count: int) -> List[int]:
    """Parse a physical page selection into sorted 0-based page indices.

    Args:
        pages: 1-based physical page number selection. Accepts:
            - str: human-friendly spec, e.g. "1-3,5,8-10"
            - List[int]: explicit 1-based page numbers, e.g. [1, 3, 5]
        page_count: Total number of pages in the document. Used to validate
            and clip the selection.

    Returns:
        Sorted list of unique 0-based page indices, each in [0, page_count).

    Raises:
        ValueError: If the spec is malformed, or selects no page within
            [1, page_count].
    """
    selected: set = set()

    if isinstance(pages, str):
        for part in pages.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                try:
                    start, end = int(start_s), int(end_s)
                except ValueError:
                    raise ValueError(f"Invalid page range: '{part}'")
                if start > end:
                    raise ValueError(f"Invalid page range: '{part}'")
                selected.update(range(start - 1, end))
            else:
                try:
                    selected.add(int(part) - 1)
                except ValueError:
                    raise ValueError(f"Invalid page number: '{part}'")
    else:
        selected.update(int(p) - 1 for p in pages)

    valid = sorted(i for i in selected if 0 <= i < page_count)
    if not valid:
        raise ValueError(
            f"No valid pages selected from pages={pages!r} for a document "
            f"with {page_count} page(s)."
        )
    return valid


def convert_pdf_to_images(
    pdf_path: Union[str, bytes],
    dpi: int = 300,
    pages: Optional[Union[str, List[int]]] = None,
) -> List[Image.Image]:
    """Convert a PDF file to a list of PIL Images (one per selected page).

    Args:
        pdf_path: Path to the PDF file or PDF bytes.
        dpi: Resolution for rendering. Higher values give better quality
             but use more memory. Defaults to 300.
        pages: Optional physical page selection, 1-based. Accepts a
            human-friendly spec string such as "1-3,5,8-10" or an explicit
            list of 1-based page numbers, e.g. [1, 3, 5]. Defaults to None,
            which renders every page in the document.

    Returns:
        List of PIL Images, one per selected PDF page, in ascending page
        order.
    """
    Image.MAX_IMAGE_PIXELS = (
        None  # Disable decompression bomb check for large PDF pages
    )

    if isinstance(pdf_path, bytes):
        doc = fitz.open(stream=pdf_path, filetype="pdf")
    else:
        doc = fitz.open(pdf_path)

    page_indices = (
        parse_pages_spec(pages, len(doc)) if pages is not None else range(len(doc))
    )

    images = []
    for page_num in page_indices:
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        images.append(Image.open(io.BytesIO(img_data)).convert("RGB"))

    doc.close()
    return images
