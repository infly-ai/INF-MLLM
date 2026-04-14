"""PDF to image conversion utility."""

import io
from typing import List, Union

from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "PyMuPDF is required for PDF rendering. Install it with: pip install pymupdf"
    )


def convert_pdf_to_images(
    pdf_path: Union[str, bytes],
    dpi: int = 300,
) -> List[Image.Image]:
    """Convert a PDF file to a list of PIL Images (one per page).

    Args:
        pdf_path: Path to the PDF file or PDF bytes.
        dpi: Resolution for rendering. Higher values give better quality
             but use more memory. Defaults to 300.

    Returns:
        List of PIL Images, one per PDF page.
    """
    Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check for large PDF pages

    if isinstance(pdf_path, bytes):
        doc = fitz.open(stream=pdf_path, filetype="pdf")
    else:
        doc = fitz.open(pdf_path)

    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        images.append(Image.open(io.BytesIO(img_data)).convert("RGB"))

    doc.close()
    return images
