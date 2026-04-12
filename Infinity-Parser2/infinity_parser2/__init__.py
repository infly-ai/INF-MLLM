"""Infinity-Parser2: Document parsing Python package."""

__version__ = "0.1.0"

from .parser import InfinityParser2
from .backends import (
    BaseBackend,
    TransformersBackend,
    VLLMEngineBackend,
    VLLMServerBackend,
)
from .prompts import ParseMode, PROMPT_DOC2JSON, PROMPT_DOC2MD
from .utils import convert_pdf_to_images

__all__ = [
    "InfinityParser2",
    "BaseBackend",
    "TransformersBackend",
    "VLLMEngineBackend",
    "VLLMServerBackend",
    "convert_pdf_to_images",
    "ParseMode",
    "PROMPT_DOC2JSON",
    "PROMPT_DOC2MD",
    "__version__",
]
