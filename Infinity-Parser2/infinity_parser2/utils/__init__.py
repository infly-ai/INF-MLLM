"""Utility functions for Infinity-Parser2."""

from .file import (
    get_files_from_directory,
    is_supported_file,
    normalize_input,
    prepare_batch_entries,
    save_results,
    SUPPORTED_OUTPUT_FORMATS,
    compress_directory_to_zip,
    package_results_as_zip,
)
from .pdf import convert_pdf_to_images
from .image import encode_image_to_base64, load_image
from .model import ModelCache, get_model_cache


from .utils import (
    convert_json_to_markdown,
    extract_json_content,
    obtain_origin_hw,
    postprocess_doc2json_result,
    restore_abs_bbox_coordinates,
    postprocess_doc2md_result,
    truncate_last_incomplete_element,
    draw_bboxes_on_image,
)

__all__ = [
    "convert_json_to_markdown",
    "extract_json_content",
    "get_files_from_directory",
    "is_supported_file",
    "normalize_input",
    "obtain_origin_hw",
    "postprocess_doc2json_result",
    "postprocess_doc2md_result",
    "prepare_batch_entries",
    "restore_abs_bbox_coordinates",
    "save_results",
    "SUPPORTED_OUTPUT_FORMATS",
    "truncate_last_incomplete_element",
    "draw_bboxes_on_image",
    "compress_directory_to_zip",
    "package_results_as_zip",
    "load_image",
    "convert_pdf_to_images",
    "encode_image_to_base64",
    "ModelCache",
    "get_model_cache",
]
