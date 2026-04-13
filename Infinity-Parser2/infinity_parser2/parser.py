"""Infinity-Parser2 main interface."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image

from .backends import (
    BaseBackend,
    TransformersBackend,
    VLLMEngineBackend,
    VLLMServerBackend,
)
from .prompts import ParseMode, PROMPT_DOC2JSON, PROMPT_DOC2MD
from .utils import *


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
        tensor_parallel_size: Tensor parallel size for vllm-engine.
            Defaults to the number of available GPUs (via torch.cuda.device_count()).
        device: Device type, must be "cuda". Raises ValueError if set to anything else.
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

    def __init__(
        self,
        model_name: str = "infly/Infinity-Parser2-Pro",
        backend: str = "vllm-engine",
        tensor_parallel_size: Optional[int] = None,
        device: str = "cuda",
        api_url: str = "http://localhost:8000/v1/chat/completions",
        api_key: str = "EMPTY",
        min_pixels: int = 2048,
        max_pixels: int = 16777216,
        model_cache_dir: Optional[str] = None,
        **kwargs,
    ):
        if device != "cuda":
            raise ValueError("device must be 'cuda' for Infinity-Parser2-Pro.")

        self.model_name = model_name
        self.backend_name = backend.lower()
        self.tensor_parallel_size = (
            tensor_parallel_size
            if tensor_parallel_size is not None
            else torch.cuda.device_count()
        )
        self.device = device
        self.api_url = api_url
        self.api_key = api_key
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.kwargs = kwargs

        # Initialize model cache and resolve model path (stored separately)
        cache = get_model_cache(model_cache_dir)
        self._model_path = cache.resolve_model_path(self.model_name)

        self._backend: BaseBackend = self._init_backend()

    def _init_backend(self) -> BaseBackend:
        """Initialize and return the backend instance."""
        if self.backend_name not in BACKEND_REGISTRY:
            raise ValueError(
                f"Unsupported backend: {self.backend_name}. "
                f"Supported backends: {list(BACKEND_REGISTRY.keys())}"
            )
        backend_cls = BACKEND_REGISTRY[self.backend_name]
        common_kwargs = {
            "model_name": self._model_path,
            "device": self.device,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            **self.kwargs,
        }
        if self.backend_name == "vllm-server":
            backend_kwargs = {**common_kwargs, "api_url": self.api_url, "api_key": self.api_key}
        elif self.backend_name == "vllm-engine":
            backend_kwargs = {**common_kwargs, "tensor_parallel_size": self.tensor_parallel_size}
        else:  # transformers
            backend_kwargs = common_kwargs
        return backend_cls(**backend_kwargs)

    def parse(
        self,
        input_data: Union[str, List[str], Image.Image],
        prompt: Optional[str] = None,
        prompt_mode: ParseMode = ParseMode.DOC2JSON,
        batch_size: int = 4,
        output_dir: Optional[str] = None,
        output_format: str = "md",
        **kwargs,
    ) -> Optional[Union[str, List[str], Dict[str, str]]]:
        """Parse document(s) and extract text content.

        Args:
            input_data: Input can be:
                - str: Single file path or directory path
                - List[str]: List of file paths
                - PIL.Image.Image: Image object
            prompt: Custom prompt text for the model. If provided, takes precedence
                over prompt_mode and uses the default parse behavior (no special
                result processing). Defaults to None.
            prompt_mode: Parsing mode. Options:
                - ParseMode.DOC2JSON: Extract layout to JSON, return JSON string.
                - ParseMode.DOC2MD: Directly convert to Markdown, return Markdown.
                - None: Use default general parsing prompt.
            batch_size: Number of images to process in one batch. Defaults to 4.
            output_dir: If provided, results are saved to output_dir and this function
                returns None. If None, results are returned directly.
            output_format: Output format for results. Options: "md" or "json".
                Defaults to "md".
                - For DOC2JSON tasks:
                    - output_format="md": Returns markdown (converts JSON to markdown
                      via convert_json_to_markdown). If output_dir is set, saves only
                      the markdown result.
                    - output_format="json": Returns raw JSON result. If output_dir is
                      set, saves only the JSON result.
                - For DOC2MD tasks or when prompt is provided: Only "md" is supported.
                  If "json" is passed, a ValueError will be raised.
            **kwargs: Additional arguments passed to the model.

        Returns:
            When output_dir is None:
                - str: Parsed result for a single file or image.
                - List[str]: Parsed results for a list of files.
                - Dict[str, str]: Mapping from file path to parsed result for a directory.
            When output_dir is set, returns None.

        Example:
            >>> parser = InfinityParser2()
            >>> # Single file, returns str
            >>> result = parser.parse("document.pdf")
            >>> # Multiple files, returns List[str]
            >>> result = parser.parse(["doc1.pdf", "doc2.pdf"])
            >>> # Directory, returns Dict[str, str]
            >>> result = parser.parse("/path/to/docs")
            >>> # Save results to output_dir, returns None
            >>> parser.parse("document.pdf", output_dir="./output")
        """
        is_doc2json = prompt is None and prompt_mode == ParseMode.DOC2JSON
        is_doc2md = prompt_mode == ParseMode.DOC2MD

        if output_format not in ("md", "json"):
            raise ValueError(f"output_format must be 'md' or 'json', got '{output_format}'")

        if output_format == "json" and (is_doc2md or prompt is not None):
            raise ValueError(
                "output_format='json' is only supported for DOC2JSON tasks. "
                "For DOC2MD tasks or when custom prompt is provided, "
                "output_format must be 'md'."
            )

        is_directory = isinstance(input_data, str) and os.path.isdir(input_data)
        file_paths = normalize_input(input_data)
        file_results = self._parse_files(
            file_paths, prompt, prompt_mode, batch_size, output_format
        )

        if output_dir is not None:
            save_results(
                file_paths, file_results, output_dir,
                is_doc2json=is_doc2json, output_format=output_format
            )
        elif is_directory:
            return dict(zip(file_paths, file_results))
        elif len(file_results) == 1:
            return file_results[0]
        return file_results

    def _parse_files(
        self,
        inputs: List[Union[str, Image.Image]],
        prompt: Optional[str] = None,
        prompt_mode: ParseMode = ParseMode.DOC2JSON,
        batch_size: int = 4,
        output_format: str = "md",
    ) -> List[str]:
        """Parse multiple files with batched inference.

        All images (including PDF pages) are collected and batched together for
        efficient inference. Results are then aggregated back to the original
        file-level granularity.
        """
        batch_entries, pdf_page_batch_indices = prepare_batch_entries(inputs)

        if not batch_entries:
            return [] if len(inputs) > 1 else ""

        # Determine effective prompt
        if prompt is not None:
            effective_prompt = prompt
        elif prompt_mode == ParseMode.DOC2JSON:
            effective_prompt = PROMPT_DOC2JSON
        elif prompt_mode == ParseMode.DOC2MD:
            effective_prompt = PROMPT_DOC2MD
        else:
            effective_prompt = "Please transform the document's contents into Markdown format."

        raw_inputs = [entry[1] for entry in batch_entries]
        batch_results = self._backend.parse_batch(raw_inputs, effective_prompt, batch_size=batch_size)

        # Postprocess for DOC2JSON
        is_doc2json = prompt is None and prompt_mode == ParseMode.DOC2JSON
        if is_doc2json:
            batch_results = postprocess_doc2json_batch(batch_results, batch_entries)

        # Aggregate batch results back to file-level
        file_results: List[str] = ["" for _ in inputs]
        page_json_lists: List[List[str]] = [[] for _ in inputs]

        for entry_idx, (_, input_item) in enumerate(batch_entries):
            page_file_idx = batch_entries[entry_idx][0]
            if entry_idx in pdf_page_batch_indices:
                if is_doc2json and output_format == "json":
                    page_json_lists[page_file_idx].append(batch_results[entry_idx])
                elif is_doc2json:
                    md_text = convert_json_to_markdown(batch_results[entry_idx])
                    if file_results[page_file_idx]:
                        file_results[page_file_idx] += "\n\n"
                    file_results[page_file_idx] += md_text
                else:
                    if file_results[page_file_idx]:
                        file_results[page_file_idx] += "\n\n"
                    file_results[page_file_idx] += batch_results[entry_idx]
            else:
                file_results[page_file_idx] = batch_results[entry_idx]

        # For DOC2JSON + JSON, join page JSONs into a single JSON array
        if is_doc2json and output_format == "json":
            for idx in range(len(inputs)):
                if page_json_lists[idx]:
                    file_results[idx] = "[" + ",".join(page_json_lists[idx]) + "]"

        return file_results
