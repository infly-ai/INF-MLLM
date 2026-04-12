"""Utility functions for Infinity-Parser2."""

from .file import (
    get_files_from_directory,
    is_supported_file,
    normalize_input,
    save_results,
)
from .image import encode_file_to_base64, load_image
from .model import ModelCache, get_model_cache
from .pdf import convert_pdf_to_images
from .utils import (
    convert_json_to_markdown,
    extract_json_content,
    obtain_origin_hw,
    postprocess_doc2json_batch,
    postprocess_doc2json_result,
    restore_abs_bbox_coordinates,
    truncate_last_incomplete_element,
)

__all__ = [
    "convert_pdf_to_images",
    "convert_json_to_markdown",
    "extract_json_content",
    "encode_file_to_base64",
    "get_files_from_directory",
    "get_model_cache",
    "is_supported_file",
    "load_image",
    "ModelCache",
    "normalize_input",
    "obtain_origin_hw",
    "postprocess_doc2json_batch",
    "postprocess_doc2json_result",
    "restore_abs_bbox_coordinates",
    "save_results",
    "truncate_last_incomplete_element",
]
