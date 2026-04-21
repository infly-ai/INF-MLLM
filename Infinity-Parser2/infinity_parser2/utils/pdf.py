"""PDF to image conversion utility."""

import io
from pathlib import Path
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
    Image.MAX_IMAGE_PIXELS = (
        None  # Disable decompression bomb check for large PDF pages
    )

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


def images_to_pdf(
    img_paths: Union[str, Path, list],
    pdf_path: Union[str, Path],
) -> Path:
    """Convert one or more image files into a single PDF.

    Args:
        img_paths: A single image path or a list of image paths.
            Images are combined in order (first image → first page, etc.).
        pdf_path: Destination path for the generated PDF file.
            Parent directories are created automatically.

    Returns:
        The resolved Path of the written PDF file.

    Raises:
        ValueError: If ``img_paths`` is empty.
        FileNotFoundError: If any image path does not exist.
    """
    if isinstance(img_paths, (str, Path)):
        img_paths = [img_paths]

    if not img_paths:
        raise ValueError("img_paths is empty")

    images = []
    for p in img_paths:
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        img = Image.open(p)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        images.append(img)

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(pdf_path, save_all=True, append_images=images[1:], resolution=300.0)
    return pdf_path
