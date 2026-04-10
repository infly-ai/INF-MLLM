"""Infinity-Parser2 main interface."""

import io
import os
from pathlib import Path
from typing import List, Union

from PIL import Image

from .backends import (
    BaseBackend,
    TransformersBackend,
    VLLMEngineBackend,
    VLLMServerBackend,
)


BACKEND_REGISTRY = {
    "transformers": TransformersBackend,
    "vllm-engine": VLLMEngineBackend,
    "vllm-server": VLLMServerBackend,
}


class InfinityParser2:
    """Document parser using Infinity-Parser2-Pro model.

    Supports parsing of PDF files and images (PNG, JPG, etc.) into structured text.

    Args:
        model_name: Model name or local path. Defaults to "infly/Infinity-Parser2-Pro".
        backend: Inference backend. Options:
            - "transformers": HuggingFace transformers (local inference)
            - "vllm-engine": vLLM Engine (local batch inference via LLM class)
            - "vllm-server": vLLM OpenAI-Compatible Server (HTTP API)
            Defaults to "vllm-engine".
        tensor_parallel_size: Tensor parallel size for vllm-engine. Defaults to 1.
        device: Device type, "cuda" or "cpu". Defaults to "cuda".
        api_url: API URL for vllm-server backend.
        api_key: API key for vllm-server backend.
        **kwargs: Additional arguments passed to the backend.

    Example:
        >>> from infinity_parser2 import InfinityParser2
        >>> parser = InfinityParser2()
        >>> result = parser.parse("document.pdf")
        >>> print(result)
    """

    SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    SUPPORTED_DOC_EXTENSIONS = {".pdf"}

    def __init__(
        self,
        model_name: str = "infly/Infinity-Parser2-Pro",
        backend: str = "vllm-engine",
        tensor_parallel_size: int = 1,
        device: str = "cuda",
        api_url: str = "http://localhost:8000/v1/chat/completions",
        api_key: str = "EMPTY",
        **kwargs,
    ):
        self.model_name = model_name
        self.backend_name = backend.lower()
        self.tensor_parallel_size = tensor_parallel_size
        self.device = device
        self.api_url = api_url
        self.api_key = api_key
        self.kwargs = kwargs

        if self.backend_name not in BACKEND_REGISTRY:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Supported backends: {list(BACKEND_REGISTRY.keys())}"
            )

        self._backend: BaseBackend = None

    @property
    def backend(self) -> BaseBackend:
        """Get or create the backend instance (lazy initialization)."""
        if self._backend is None:
            backend_cls = BACKEND_REGISTRY[self.backend_name]
            self._backend = backend_cls(
                model_name=self.model_name,
                device=self.device,
                tensor_parallel_size=self.tensor_parallel_size,
                api_url=self.api_url,
                api_key=self.api_key,
                **self.kwargs,
            )
        return self._backend

    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_IMAGE_EXTENSIONS or ext in self.SUPPORTED_DOC_EXTENSIONS

    def _get_files_from_directory(self, directory: str) -> List[str]:
        """Get all supported files from a directory."""
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if self._is_supported_file(file_path):
                    files.append(file_path)
        return sorted(files)

    def parse(
        self,
        input_data: Union[str, List[str], Image.Image, bytes],
        prompt: str = "Please parse this document and extract all text content.",
        **kwargs,
    ) -> Union[str, List[str]]:
        """Parse document(s) and extract text content.

        Args:
            input_data: Input can be:
                - str: Single file path or directory path
                - List[str]: List of file paths
                - PIL.Image.Image: Image object
                - bytes: Image or PDF bytes
            prompt: Prompt text for the model. Defaults to a general parsing prompt.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Parsed text content. Returns a list if multiple inputs are provided.

        Example:
            >>> parser = InfinityParser2()
            >>> # Parse single file
            >>> result = parser.parse("document.pdf")
            >>> # Parse multiple files
            >>> results = parser.parse(["doc1.pdf", "doc2.png"])
            >>> # Parse directory
            >>> results = parser.parse("/path/to/documents/")
        """
        if isinstance(input_data, str):
            if os.path.isdir(input_data):
                file_paths = self._get_files_from_directory(input_data)
                if not file_paths:
                    raise ValueError(f"No supported files found in directory: {input_data}")
                return self._parse_files(file_paths, prompt, **kwargs)
            elif os.path.isfile(input_data):
                if not self._is_supported_file(input_data):
                    raise ValueError(f"Unsupported file type: {input_data}")
                return self._parse_files([input_data], prompt, **kwargs)
            else:
                raise FileNotFoundError(f"File or directory not found: {input_data}")
        elif isinstance(input_data, list):
            file_paths = []
            for item in input_data:
                if not isinstance(item, str):
                    raise TypeError(f"Expected str in list, got {type(item)}")
                if not os.path.isfile(item):
                    raise FileNotFoundError(f"File not found: {item}")
                if not self._is_supported_file(item):
                    raise ValueError(f"Unsupported file type: {item}")
                file_paths.append(item)
            return self._parse_files(file_paths, prompt, **kwargs)
        elif isinstance(input_data, Image.Image):
            return self.backend.parse(input_data, prompt, **kwargs)
        elif isinstance(input_data, bytes):
            try:
                img = Image.open(io.BytesIO(input_data))
                return self.backend.parse(img, prompt, **kwargs)
            except Exception:
                raise ValueError("Could not decode bytes as image. Please provide a valid image or PDF.")
        else:
            raise TypeError(
                f"Unsupported input type: {type(input_data)}. "
                "Expected str, List[str], PIL.Image.Image, or bytes."
            )

    def _parse_files(
        self,
        file_paths: List[str],
        prompt: str,
        **kwargs,
    ) -> Union[str, List[str]]:
        """Parse multiple files.

        Args:
            file_paths: List of file paths.
            prompt: Prompt text for the model.
            **kwargs: Additional arguments.

        Returns:
            Parsed text content. Returns a single string if only one file.
        """
        results = self.backend.parse_batch(file_paths, prompt, **kwargs)
        return results if len(results) > 1 else results[0]
