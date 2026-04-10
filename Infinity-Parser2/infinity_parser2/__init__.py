"""Infinity-Parser2: Document parsing Python package."""

__version__ = "0.1.0"

from .parser import InfinityParser2
from .backends import (
    BaseBackend,
    TransformersBackend,
    VLLMEngineBackend,
    VLLMServerBackend,
)
from .utils.pdf import convert_pdf_to_images

__all__ = [
    "InfinityParser2",
    "BaseBackend",
    "TransformersBackend",
    "VLLMEngineBackend",
    "VLLMServerBackend",
    "convert_pdf_to_images",
    "__version__",
]
