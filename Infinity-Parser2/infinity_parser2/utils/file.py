"""File system utilities for Infinity-Parser2."""

import os
import uuid
from pathlib import Path
from typing import List, Union

from PIL import Image

from .utils import convert_json_to_markdown


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
SUPPORTED_DOC_EXTENSIONS = {".pdf"}


def normalize_input(input_data: Union[str, List[str], Image.Image]) -> List[Union[str, Image.Image]]:
    """Normalize input to a list of file paths or images.

    Args:
        input_data: Input can be:
            - str: Single file path or directory path
            - List[str]: List of file paths
            - PIL.Image.Image: Image object

    Returns:
        List of file paths or PIL Images.

    Raises:
        FileNotFoundError: If file or directory not found.
        TypeError: If list contains non-string items.
        ValueError: If directory is empty or file type is unsupported.
    """
    if isinstance(input_data, str):
        if os.path.isdir(input_data):
            file_paths = get_files_from_directory(input_data)
            if not file_paths:
                raise ValueError(f"No supported files found in directory: {input_data}")
            return file_paths
        elif os.path.isfile(input_data):
            if not is_supported_file(input_data):
                raise ValueError(f"Unsupported file type: {input_data}")
            return [input_data]
        else:
            raise FileNotFoundError(f"File or directory not found: {input_data}")
    elif isinstance(input_data, list):
        file_paths = []
        for item in input_data:
            if not isinstance(item, str):
                raise TypeError(f"Expected str in list, got {type(item)}")
            if not os.path.isfile(item):
                raise FileNotFoundError(f"File not found: {item}")
            if not is_supported_file(item):
                raise ValueError(f"Unsupported file type: {item}")
            file_paths.append(item)
        return file_paths
    elif isinstance(input_data, Image.Image):
        return [input_data]
    else:
        raise TypeError(
            f"Unsupported input type: {type(input_data)}. "
            "Expected str, List[str], or PIL.Image.Image."
        )


def is_supported_file(file_path: str) -> bool:
    """Check if file is supported."""
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_IMAGE_EXTENSIONS or ext in SUPPORTED_DOC_EXTENSIONS


def get_files_from_directory(directory: str) -> List[str]:
    """Get all supported files from a directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if is_supported_file(file_path):
                files.append(file_path)
    return sorted(files)


def save_results(
    inputs: List[Union[str, Image.Image]],
    results: List[str],
    output_dir: str,
    is_doc2json: bool = False,
) -> None:
    """Save parsing results to output directory.

    Unified entry point that delegates to save_results_json or save_results_md
    based on the is_doc2json flag. Prints the output directory path to console.

    Args:
        inputs: Original inputs (file paths or PIL Images).
        results: Parsed results (same order as inputs).
        output_dir: Base output directory.
        is_doc2json: If True, save as JSON + Markdown (doc2json mode).
    """
    keys = [uuid.uuid4().hex[:8] if isinstance(inp, Image.Image) else inp for inp in inputs]
    if is_doc2json:
        md_results = [convert_json_to_markdown(r) for r in results]
        save_results_json(keys, results, md_results, output_dir)
    else:
        save_results_md(keys, results, output_dir)
    print(f"[Infinity-Parser2] Results saved to: {os.path.abspath(output_dir)}")


def save_results_md(keys: List[str], results: List[str], output_dir: str) -> None:
    """Save markdown parsing results to output directory.

    Creates a subdirectory for each entry and writes result.md inside it.
    For file paths, the folder name is the filename (basename); for UUIDs,
    the folder name is the UUID itself.

    Args:
        keys: Identifiers (file paths or UUIDs).
        results: Parsed markdown text results (same order as keys).
        output_dir: Base output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, result in zip(keys, results):
        folder_name = Path(key).name
        file_dir = os.path.join(output_dir, folder_name)
        os.makedirs(file_dir, exist_ok=True)
        result_path = os.path.join(file_dir, "result.md")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(result)


def save_results_json(
    keys: List[str], json_results: List[str], md_results: List[str], output_dir: str
) -> None:
    """Save doc2json parsing results (JSON + Markdown) to output directory.

    Creates a subdirectory for each entry containing:
    - result.json: raw JSON output from the model
    - result.md: markdown converted from JSON

    Args:
        keys: Identifiers (file paths or UUIDs).
        json_results: Raw JSON results from the model.
        md_results: Markdown results converted from JSON.
        output_dir: Base output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, json_result, md_result in zip(keys, json_results, md_results):
        folder_name = Path(key).name
        file_dir = os.path.join(output_dir, folder_name)
        os.makedirs(file_dir, exist_ok=True)

        json_path = os.path.join(file_dir, "result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_result)

        md_path = os.path.join(file_dir, "result.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_result)
