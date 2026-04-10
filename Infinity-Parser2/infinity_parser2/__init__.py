"""Infinity-Parser2: Document parsing Python package."""

__version__ = "0.1.0"

from .parser import InfinityParser2
from .backends import (
    BaseBackend,
    TransformersBackend,
    VLLMEngineBackend,
    VLLMServerBackend,
)

__all__ = [
    "InfinityParser2",
    "BaseBackend",
    "TransformersBackend",
    "VLLMEngineBackend",
    "VLLMServerBackend",
    "__version__",
]
