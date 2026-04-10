"""Image encoding and loading utilities."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Tuple, Union

from PIL import Image

from qwen_vl_utils.vision_process import smart_resize

try:
    from importlib import metadata
    _qwen_vl_utils_version = metadata.version("qwen-vl-utils")
    if _qwen_vl_utils_version < "0.0.14":
        raise ImportError("qwen-vl-utils version 0.0.14 or higher is required")
except metadata.PackageNotFoundError:
    raise ImportError("qwen-vl-utils is not installed. Install it with: pip install qwen-vl-utils")


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
    """
    if isinstance(image_obj, str):
        image = Image.open(image_obj)
        ext = Path(image_obj).suffix.lower()
        mime_type = IMAGE_MIME_TYPES.get(ext, "image/jpeg")
    else:
        image = image_obj.copy()
        # Try to get format from PIL Image, default to jpeg
        mime_type = IMAGE_MIME_TYPES.get(f".{image.format}".lower(), "image/jpeg") if image.format else "image/jpeg"

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


def encode_file_to_data_url(
    image_obj: Union[Image.Image, str],
    min_pixels: int = 2048,
    max_pixels: int = 16777216,
) -> str:
    """Encode image to a data URL string.

    Args:
        image_obj: File path or PIL Image.
        min_pixels: Minimum number of pixels for resizing.
        max_pixels: Maximum number of pixels for resizing.

    Returns:
        Data URL string (e.g., "data:image/png;base64,...").
    """
    base64_str, mime_type = encode_file_to_base64(
        image_obj, min_pixels=min_pixels, max_pixels=max_pixels
    )
    return f"data:{mime_type};base64,{base64_str}"