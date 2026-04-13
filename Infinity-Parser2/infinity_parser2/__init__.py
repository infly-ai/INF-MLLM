"""Infinity-Parser2: Document parsing Python package."""

__version__ = "0.1.0"

from .parser import InfinityParser2
from .backends import (
    BaseBackend,
    TransformersBackend,
    VLLMEngineBackend,
    VLLMServerBackend,
)
from .prompts import PROMPT_DOC2JSON, PROMPT_DOC2MD, SUPPORTED_TASK_TYPES
from .utils import convert_pdf_to_images

__all__ = [
    "InfinityParser2",
    "BaseBackend",
    "TransformersBackend",
    "VLLMEngineBackend",
    "VLLMServerBackend",
    "convert_pdf_to_images",
    "PROMPT_DOC2JSON",
    "PROMPT_DOC2MD",
    "SUPPORTED_TASK_TYPES",
    "__version__",
]
