"""Model cache management for Infinity-Parser2."""

import json
import os
from typing import Optional

from huggingface_hub import snapshot_download

# Default cache directory
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/infinity_parser2")


class ModelCache:
    """Manages local model cache for Infinity-Parser2.

    Automatically detects if a model is already downloaded and cached locally.
    If not, prompts the user and downloads it from HuggingFace Hub.

    Attributes:
        cache_dir: Directory where model cache metadata is stored.
        models_file: Path to the JSON file containing cached model information.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize ModelCache.

        Args:
            cache_dir: Custom cache directory. Defaults to ~/.cache/infinity_parser2.
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.models_file = os.path.join(self.cache_dir, "models_cache.json")
        self._ensure_cache_dir()
        self._models_cache: dict = self._load_cache()

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_cache(self) -> dict:
        """Load cached model information from JSON file."""
        if not os.path.exists(self.models_file):
            return {}
        try:
            with open(self.models_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_cache(self) -> None:
        """Save cached model information to JSON file."""
        with open(self.models_file, "w", encoding="utf-8") as f:
            json.dump(self._models_cache, f, indent=2, ensure_ascii=False)

    def is_cached(self, model_name: str) -> bool:
        """Check if a model is already cached locally.

        Args:
            model_name: HuggingFace model name (e.g., "infly/Infinity-Parser2-Pro").

        Returns:
            True if model is cached and the local path exists.
        """
        if model_name not in self._models_cache:
            return False
        local_path = self._models_cache[model_name].get("local_path")
        if not local_path or not os.path.exists(local_path):
            return False
        return True

    def get_cached_path(self, model_name: str) -> Optional[str]:
        """Get the cached local path for a model.

        Args:
            model_name: HuggingFace model name.

        Returns:
            Local path if cached, None otherwise.
        """
        if not self.is_cached(model_name):
            return None
        return self._models_cache[model_name].get("local_path")

    def cache_model(self, model_name: str, local_path: str) -> None:
        """Cache a model's local path.

        Args:
            model_name: HuggingFace model name.
            local_path: Local directory where the model is stored.
        """
        self._models_cache[model_name] = {
            "local_path": local_path,
            "cached": True,
        }
        self._save_cache()

    def download_and_cache(
        self,
        model_name: str,
        target_dir: Optional[str] = None,
        force_download: bool = False,
    ) -> str:
        """Download a model from HuggingFace Hub and cache its location.

        Args:
            model_name: HuggingFace model name (e.g., "infly/Infinity-Parser2-Pro").
            target_dir: Custom download directory. If None, uses cache_dir/model_name.
            force_download: If True, re-download even if cached.

        Returns:
            Local path where the model is stored.
        """
        if target_dir is None:
            safe_name = model_name.replace("/", "_")
            target_dir = os.path.join(self.cache_dir, safe_name)

        # If already cached and not forcing download, return cached path
        if self.is_cached(model_name) and not force_download:
            cached_path = self.get_cached_path(model_name)
            print(f"[Infinity-Parser2] Model already cached at: {cached_path}")
            return cached_path

        print(f"[Infinity-Parser2] Model '{model_name}' not found locally.")
        print(f"[Infinity-Parser2] Starting download to: {target_dir}")
        print("[Infinity-Parser2] This may take a few minutes depending on model size and network...")

        os.makedirs(target_dir, exist_ok=True)

        snapshot_download(
            repo_id=model_name,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )

        self.cache_model(model_name, target_dir)
        print(f"[Infinity-Parser2] Model downloaded and cached successfully!")
        print(f"[Infinity-Parser2] Cache location: {target_dir}")

        return target_dir

    def resolve_model_path(self, model_name: str) -> str:
        """Resolve the model path for loading.

        If model is not cached, downloads it automatically.
        If model is a local path, returns it directly.

        Args:
            model_name: HuggingFace model name or local path.

        Returns:
            Resolved local path for model loading.
        """
        # If it's already a local path, return it directly
        if os.path.exists(model_name):
            return model_name

        # If cached, return cached path
        if self.is_cached(model_name):
            cached_path = self.get_cached_path(model_name)
            print(f"[Infinity-Parser2] Found cached model at: {cached_path}")
            return cached_path

        # Otherwise, download and cache
        return self.download_and_cache(model_name)


# Global model cache instance
_model_cache: Optional[ModelCache] = None


def get_model_cache(cache_dir: Optional[str] = None) -> ModelCache:
    """Get or create the global ModelCache instance.

    Args:
        cache_dir: Custom cache directory for this session.

    Returns:
        The global ModelCache instance.
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache(cache_dir)
    return _model_cache
