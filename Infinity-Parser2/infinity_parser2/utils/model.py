"""Model downloading and management utilities for Infinity-Parser2."""

import os
from typing import Dict

from huggingface_hub import snapshot_download


def download_model(model_name: str, target_path: str) -> str:
    """Download model from HuggingFace Hub to local path.

    This function allows pre-downloading the model so that subsequent
    initializations can load from local disk without network access.

    Args:
        model_name: Model name on HuggingFace Hub (e.g., "infly/Infinity-Parser2-Pro").
        target_path: Local directory to store the model.

    Returns:
        The path where the model was downloaded.

    Example:
        >>> from infinity_parser2.utils import download_model
        >>> download_model("infly/Infinity-Parser2-Pro", "./models/infinity-parser2")
    """
    if os.path.exists(target_path):
        print(f"Model already exists at: {target_path}")
        return target_path

    print(f"Downloading model {model_name} to {target_path}...")
    os.makedirs(target_path, exist_ok=True)

    snapshot_download(
        repo_id=model_name,
        local_dir=target_path,
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded successfully to: {target_path}")
    return target_path


def get_model_info(model_path: str) -> Dict[str, str]:
    """Get information about a local model.

    Args:
        model_path: Path to the local model directory.

    Returns:
        Dictionary with model information including path and size.

    Example:
        >>> from infinity_parser2.utils import get_model_info
        >>> info = get_model_info("./models/infinity-parser2")
        >>> print(info)
        # {'status': 'found', 'path': './models/infinity-parser2', 'size_mb': 12345.67}
    """
    if not os.path.exists(model_path):
        return {"status": "not_found", "path": model_path}

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)

    return {
        "status": "found",
        "path": model_path,
        "size_mb": round(total_size / (1024 * 1024), 2),
    }
