"""Image encoding and loading utilities."""

import base64
import warnings
from io import BytesIO
from pathlib import Path
from typing import Tuple, Union, Optional, Callable

from PIL import Image
from importlib import metadata

# smart_resize is optional - only needed for encode_image_to_base64 with resizing
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
            UserWarning,
        )
except metadata.PackageNotFoundError:
    warnings.warn(
        "qwen-vl-utils is not installed. The encode_image_to_base64 function with "
        "smart resizing will not be available. Install it with: pip install qwen-vl-utils",
        UserWarning,
    )
except ImportError as e:
    warnings.warn(
        f"Failed to import qwen-vl-utils: {e}. The encode_image_to_base64 function with "
        "smart resizing will not be available. Install it with: pip install qwen-vl-utils",
        UserWarning,
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


def encode_image_to_base64(
    image_obj: Union[Image.Image, str],
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[str, str]:
    """Encode image to base64 string.

    If both min_pixels and max_pixels are provided, the image is resized via
    smart_resize (requires qwen-vl-utils) and encoded as PNG. Otherwise the
    raw file bytes are encoded as-is.

    Args:
        image_obj: File path or PIL Image.
        min_pixels: Minimum number of pixels for resizing. Ignored if None.
        max_pixels: Maximum number of pixels for resizing. Ignored if None.

    Returns:
        Tuple of (base64 string, MIME type string).

    Raises:
        ImportError: If resizing is requested but qwen-vl-utils is not installed.
    """
    if isinstance(image_obj, str):
        raw_bytes = Path(image_obj).read_bytes()
        ext = Path(image_obj).suffix.lower()
        mime_type = IMAGE_MIME_TYPES.get(ext, "image/jpeg")
    else:
        original_format = image_obj.format
        mime_type = (
            IMAGE_MIME_TYPES.get(f".{original_format}".lower(), "image/jpeg")
            if original_format
            else "image/jpeg"
        )
        buf = BytesIO()
        image_obj.save(buf, format=original_format or "PNG")
        raw_bytes = buf.getvalue()

    if min_pixels is not None and max_pixels is not None:
        if smart_resize is None:
            raise ImportError(
                "encode_image_to_base64 with resizing requires qwen-vl-utils. "
                "Install it with: pip install qwen-vl-utils"
            )
        if isinstance(image_obj, str):
            image = Image.open(image_obj)
        else:
            image = image_obj.copy()
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
    else:
        byte_data = raw_bytes

    return base64.b64encode(byte_data).decode("utf-8"), mime_type
