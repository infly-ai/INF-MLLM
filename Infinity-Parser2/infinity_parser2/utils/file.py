"""File system utilities for Infinity-Parser2."""

import os
from pathlib import Path
from typing import Dict, List


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
SUPPORTED_DOC_EXTENSIONS = {".pdf"}


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


def save_results(keys: List[str], results: List[str], output_dir: str) -> Dict[str, str]:
    """Save parsing results to output directory.

    Creates a subdirectory for each entry and writes result.md inside it.
    For file paths, the folder name is the filename (basename); for UUIDs,
    the folder name is the UUID itself.

    Args:
        keys: Identifiers (file paths or UUIDs).
        results: Parsed text results (same order as keys).
        output_dir: Base output directory.

    Returns:
        Dict mapping each key to its saved result path.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: Dict[str, str] = {}

    for key, result in zip(keys, results):
        # Use Path(key).name to get filename, avoiding issues with os.path.join when key is absolute
        # os.path.join('/output', '/tmp/file.png') returns '/tmp/file.png' instead of '/output/tmp/file.png'
        folder_name = Path(key).name
        file_dir = os.path.join(output_dir, folder_name)
        os.makedirs(file_dir, exist_ok=True)
        result_path = os.path.join(file_dir, "result.md")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(result)
        saved_paths[key] = result_path

    return saved_paths
