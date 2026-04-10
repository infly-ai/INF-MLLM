"""Base backend interface for Infinity-Parser2."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from PIL import Image


class BaseBackend(ABC):
    """Abstract base class for inference backends.

    All backends must implement init() and parse_batch() methods.
    """

    def __init__(
        self,
        model_name: str = "infly/Infinity-Parser2-Pro",
        device: str = "cuda",
        **kwargs,
    ):
        """Initialize backend.

        Args:
            model_name: Model name on HuggingFace Hub or local path to the model.
            device: Device type, "cuda" or "cpu".
            **kwargs: Additional backend-specific arguments.
        """
        self.model_name = model_name
        self.device = device
        self.kwargs = kwargs

    @abstractmethod
    def init(self) -> None:
        """Initialize the model and processor.

        This method should be called before parse_batch().
        """
        pass

    @abstractmethod
    def parse_batch(
        self,
        input_data: list[Union[str, Image.Image]],
        prompt: str,
        batch_size: int = 1,
        **kwargs,
    ) -> list[str]:
        """Parse multiple documents.

        Args:
            input_data: List of inputs, each can be:
                - str: File path (image or PDF)
                - PIL.Image.Image: Image object
            prompt: Prompt text for the model.
            batch_size: Maximum number of images to process in one batch.
            **kwargs: Additional arguments passed to the model.

        Returns:
            List of parsed text content (one per input in the same order).
        """
        pass
