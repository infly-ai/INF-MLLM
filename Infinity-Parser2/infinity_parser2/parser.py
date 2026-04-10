"""Infinity-Parser2 main interface."""

import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import Image

from .backends import (
    BaseBackend,
    TransformersBackend,
    VLLMEngineBackend,
    VLLMServerBackend,
)
from .utils import convert_pdf_to_images


BACKEND_REGISTRY = {
    "transformers": TransformersBackend,
    "vllm-engine": VLLMEngineBackend,
    "vllm-server": VLLMServerBackend,
}


class InfinityParser2:
    """Document parser using Infinity-Parser2-Pro model.

    Supports parsing of PDF files and images (PNG, JPG, etc.) into structured text.

    Args:
        model_name: Model name on HuggingFace Hub (e.g., "infly/Infinity-Parser2-Pro")
            or local path to a downloaded model. Defaults to "infly/Infinity-Parser2-Pro".
        backend: Inference backend. Options:
            - "transformers": HuggingFace transformers (local inference)
            - "vllm-engine": vLLM Engine (local batch inference via LLM class)
            - "vllm-server": vLLM OpenAI-Compatible Server (HTTP API)
          Defaults to "vllm-engine".
        tensor_parallel_size: Tensor parallel size for vllm-engine. Defaults to 1.
        device: Device type, "cuda" or "cpu". Defaults to "cuda".
        api_url: API URL for vllm-server backend.
        api_key: API key for vllm-server backend.
        min_pixels: Minimum number of pixels for image input (transformers backend only).
            Defaults to 2048.
        max_pixels: Maximum number of pixels for image input (transformers backend only).
            Defaults to 16777216 (~4096x4096).
        **kwargs: Additional arguments passed to the backend.

    Example:
        >>> from infinity_parser2 import InfinityParser2
        >>> parser = InfinityParser2(model_name="infly/Infinity-Parser2-Pro")
        >>> result = parser.parse("document.pdf")
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
        min_pixels: int = 2048,
        max_pixels: int = 16777216,
        **kwargs,
    ):
        self.model_name = model_name
        self.backend_name = backend.lower()
        self.tensor_parallel_size = tensor_parallel_size
        self.device = device
        self.api_url = api_url
        self.api_key = api_key
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
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
            # Pass different initialization parameters based on backend type
            if self.backend_name == "vllm-server":
                backend_kwargs = {
                    "model_name": self.model_name,
                    "device": self.device,
                    "api_url": self.api_url,
                    "api_key": self.api_key,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                    **self.kwargs,
                }
            elif self.backend_name == "vllm-engine":
                backend_kwargs = {
                    "model_name": self.model_name,
                    "device": self.device,
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                    **self.kwargs,
                }
            else:  # transformers
                backend_kwargs = {
                    "model_name": self.model_name,
                    "device": self.device,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                    **self.kwargs,
                }
            self._backend = backend_cls(**backend_kwargs)
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
        input_data: Union[str, List[str], Image.Image],
        prompt: str = "Please parse this document and extract all text content.",
        batch_size: int = 1,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> Union[str, List[str], Dict[str, str]]:
        """Parse document(s) and extract text content.

        Args:
            input_data: Input can be:
                - str: Single file path or directory path
                - List[str]: List of file paths
                - PIL.Image.Image: Image object
            prompt: Prompt text for the model. Defaults to a general parsing prompt.
            batch_size: Number of images to process in one batch. Defaults to 1.
            output_dir: If provided, results are saved to output_dir. Each input file
                gets its own subdirectory containing result.md. For Image inputs,
                a UUID-based folder name is generated. Returns a dict mapping
                input identifiers to their saved result paths when output_dir is set.
            **kwargs: Additional arguments passed to the model.

            Returns:
            - str: Parsed text for a single file (when output_dir is None).
            - List[str]: Parsed texts for multiple files (when output_dir is None).
            - Dict[str, str]: Mapping from input identifier to saved result path
                when output_dir is set. For file inputs, keys are file paths; for
                Image inputs, keys are synthetic identifiers.

        Example:
            >>> parser = InfinityParser2()
            >>> # Parse single file
            >>> result = parser.parse("document.pdf")
            >>> # Parse multiple files with batch_size=4
            >>> results = parser.parse(["doc1.pdf", "doc2.png"], batch_size=4)
            >>> # Parse and save to output directory
            >>> saved = parser.parse(["doc.pdf"], batch_size=4, output_dir="./output")
        """
        if isinstance(input_data, str):
            if os.path.isdir(input_data):
                file_paths = self._get_files_from_directory(input_data)
                if not file_paths:
                    raise ValueError(f"No supported files found in directory: {input_data}")
                return self._parse_files(file_paths, prompt, batch_size, output_dir, **kwargs)
            elif os.path.isfile(input_data):
                if not self._is_supported_file(input_data):
                    raise ValueError(f"Unsupported file type: {input_data}")
                return self._parse_files([input_data], prompt, batch_size, output_dir, **kwargs)
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
            return self._parse_files(file_paths, prompt, batch_size, output_dir, **kwargs)
        elif isinstance(input_data, Image.Image):
            return self._parse_files([input_data], prompt, batch_size, output_dir, **kwargs)
        else:
            raise TypeError(
                f"Unsupported input type: {type(input_data)}. "
                "Expected str, List[str], or PIL.Image.Image."
            )

    def _parse_files(
        self,
        inputs: List[Union[str, Image.Image]],
        prompt: str,
        batch_size: int = 1,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> Union[str, List[str], Dict[str, str]]:
        """Parse multiple files with batched inference and optional result saving.

        All images (including PDF pages) are collected and batched together for
        efficient inference. Results are then aggregated back to the original
        file-level granularity, and optionally saved to output_dir.

        Args:
            inputs: List of inputs, each can be:
                - str: File path (image or PDF)
                - PIL.Image.Image: Image object
            prompt: Prompt text for the model.
            batch_size: Number of images to process in one batch.
            output_dir: If provided, saves results to output_dir.
            **kwargs: Additional arguments passed to the backend.

        Returns:
            - str: Single result if one input.
            - List[str]: Results for multiple inputs (same order as inputs).
            - Dict[str, str]: Input identifier -> saved result path when output_dir is set.
        """
        batch_entries: list[tuple[int, Union[str, Image.Image]]] = []

        for idx, item in enumerate(inputs):
            if isinstance(item, str):
                ext = Path(item).suffix.lower()
                if ext == ".pdf":
                    page_images = convert_pdf_to_images(item)
                    for page_img in page_images:
                        batch_entries.append((idx, page_img))
                else:
                    batch_entries.append((idx, item))
            else:
                batch_entries.append((idx, item))

        if not batch_entries:
            return [] if len(inputs) > 1 else ""

        # Track which batch entries come from PDF (by checking if original input is PDF path)
        pdf_page_batch_indices = [
            entry_idx for entry_idx, (orig_idx, item) in enumerate(batch_entries)
            if isinstance(item, Image.Image) and isinstance(inputs[orig_idx], str)
            and Path(inputs[orig_idx]).suffix.lower() == ".pdf"
        ]

        raw_inputs = [entry[1] for entry in batch_entries]
        batch_results = self.backend.parse_batch(raw_inputs, prompt, batch_size=batch_size, **kwargs)

        file_results: List[str] = ["" for _ in inputs]
        for entry_idx, (_, input_item) in enumerate(batch_entries):
            page_file_idx = batch_entries[entry_idx][0]
            if entry_idx in pdf_page_batch_indices:
                # Merge PDF pages
                if file_results[page_file_idx]:
                    file_results[page_file_idx] += "\n\n"
                file_results[page_file_idx] += batch_results[entry_idx]
            else:
                file_results[page_file_idx] = batch_results[entry_idx]

        if output_dir is not None:
            keys = [
                uuid.uuid4().hex[:8] if isinstance(inp, Image.Image) else inp
                for inp in inputs
            ]
            return self._save_results(keys, file_results, output_dir)

        if len(inputs) == 1:
            return file_results[0]
        return file_results

    def _save_results(
        self,
        keys: List[str],
        results: List[str],
        output_dir: str,
    ) -> Dict[str, str]:
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
