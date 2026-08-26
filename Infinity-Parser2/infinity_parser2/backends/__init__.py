"""Inference backends for Infinity-Parser2."""

from importlib import import_module

from .base import BaseBackend

_BACKEND_MODULES = {
    "TransformersBackend": ".transformers",
    "VLLMEngineBackend": ".vllm_engine",
    "VLLMServerBackend": ".vllm_server",
}


def __getattr__(name):
    module_name = _BACKEND_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, __name__), name)

__all__ = [
    "BaseBackend",
    "TransformersBackend",
    "VLLMEngineBackend",
    "VLLMServerBackend",
]
