"""Image encoding and loading utilities."""

import base64
import warnings
from io import BytesIO
from pathlib import Path
from typing import Tuple, Union, Optional, Callable

from PIL import Image
from importlib import metadata

# smart_resize is optional - only needed for encode_file_to_base64 with resizing
smart_resize: Optional[Callable] = None
try:
    from qwen_vl_utils.vision_process import smart_resize
    _qwen_vl_utils_version = metadata.version("qwen-vl-utils")
    if _qwen_vl_utils_version < "0.0.14":
        warnings.warn(
            f"qwen-vl-utils version {_qwen_vl_utils_version} is installed, "
            f"but version 0.0.14 or higher is recommended. "
            f"Some features may not work correctly. "
            f"Upgrade with: pip install qwen-vl-utils>=0.0.14",
            UserWarning
        )
except metadata.PackageNotFoundError:
    warnings.warn(
        "qwen-vl-utils is not installed. The encode_file_to_base64 function with "
        "smart resizing will not be available. Install it with: pip install qwen-vl-utils",
        UserWarning
    )
except ImportError as e:
    warnings.warn(
        f"Failed to import qwen-vl-utils: {e}. The encode_file_to_base64 function with "
        "smart resizing will not be available. Install it with: pip install qwen-vl-utils",
        UserWarning
    )


# MIME type mapping for common image formats
IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


def load_image(
    input_data: Union[str, Image.Image],
) -> Image.Image:
    """Load image from file path or PIL Image and convert to RGB.

    Args:
        input_data: File path or PIL Image.

    Returns:
        PIL Image in RGB mode.

    Raises:
        TypeError: If input_data is an unsupported type.
    """
    if isinstance(input_data, str):
        return Image.open(input_data).convert("RGB")
    elif isinstance(input_data, Image.Image):
        return input_data.convert("RGB")
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")


def encode_file_to_base64(
    image_obj: Union[Image.Image, str],
    min_pixels: int = 2048,
    max_pixels: int = 16777216,
) -> Tuple[str, str]:
    """Encode image to base64 string and determine its MIME type.

    Args:
        image_obj: File path or PIL Image.
        min_pixels: Minimum number of pixels for resizing.
        max_pixels: Maximum number of pixels for resizing.

    Returns:
        Tuple of (base64 string, MIME type string).

    Raises:
        ImportError: If qwen-vl-utils is not installed.
    """
    if smart_resize is None:
        raise ImportError(
            "encode_file_to_base64 requires qwen-vl-utils. "
            "Install it with: pip install qwen-vl-utils"
        )

    if isinstance(image_obj, str):
        image = Image.open(image_obj)
        ext = Path(image_obj).suffix.lower()
        mime_type = IMAGE_MIME_TYPES.get(ext, "image/jpeg")
    else:
        # Note: image.copy() loses the format attribute, so get it before copying
        original_format = image_obj.format
        image = image_obj.copy()
        # Try to get format from original PIL Image, default to jpeg
        mime_type = (
            IMAGE_MIME_TYPES.get(f".{original_format}".lower(), "image/jpeg")
            if original_format
            else "image/jpeg"
        )

    resized_height, resized_width = smart_resize(
        height=image.size[1],
        width=image.size[0],
        factor=32,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    output_buffer = BytesIO()
    image.save(output_buffer, format="PNG")
    byte_data = output_buffer.getvalue()

    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str, mime_type


def encode_image(image_path: Union[str, Path]) -> str:
    """Read an image file and return its raw base64-encoded string (no resize).

    Unlike encode_file_to_base64, this function does NOT resize the image and
    does NOT require qwen_vl_utils. It is used by build_message to embed images
    directly into OpenAI-compatible API requests.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64-encoded string of the raw file bytes.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def images_to_b64(
    file_path: Union[str, Path],
    pdf_dpi: int = 200,
) -> list[str]:
    """Convert a single image or PDF file to a list of base64 data-URI strings.

    Each element is a data-URI suitable for embedding in an ``<img>`` tag or
    passing to a Gradio ``gr.HTML`` component. PDF files are rasterised page
    by page; image files produce a single-element list.

    Note: This function only accepts a *file path*. If you have raw bytes from
    a Gradio ``gr.File`` object (``file.data``), convert them yourself before
    calling this function.

    Args:
        file_path: Path to a PDF or image file (PNG / JPEG / WEBP …).
        pdf_dpi: Resolution used when rasterising PDF pages. Defaults to 200.

    Returns:
        List of data-URI strings, one per page (PDFs) or one for the image.
    """
    from io import BytesIO
    from pdf2image import convert_from_bytes, convert_from_path

    out: list[str] = []
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages = convert_from_path(str(path), dpi=pdf_dpi)
        for page in pages:
            buf = BytesIO()
            page.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            out.append(f"data:image/png;base64,{b64}")
    else:
        raw_bytes = path.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode()
        mime = IMAGE_MIME_TYPES.get(suffix, f"image/{suffix.lstrip('.')}")
        out.append(f"data:{mime};base64,{b64}")

    return out
