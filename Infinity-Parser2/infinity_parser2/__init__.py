"""Infinity-Parser2: Document parsing Python package."""

from importlib import import_module

__version__ = "0.4.0"

from .parser import InfinityParser2
from .backends import BaseBackend
from .prompts import PROMPT_DOC2JSON, PROMPT_DOC2MD, SUPPORTED_TASK_TYPES
from .utils import convert_pdf_to_images
from .cli import main as cli_main

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
    "cli_main",
]


def __getattr__(name):
    if name in {"TransformersBackend", "VLLMEngineBackend", "VLLMServerBackend"}:
        return getattr(import_module(".backends", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
