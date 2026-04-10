"""Base backend interface for Infinity-Parser2."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from PIL import Image


class BaseBackend(ABC):
    """Abstract base class for inference backends.

    All backends must implement init() and parse() methods.
    """

    def __init__(
        self,
        model_name: str = "infly/Infinity-Parser2-Pro",
        device: str = "cuda",
        **kwargs,
    ):
        """Initialize backend.

        Args:
            model_name: Model name or local path.
            device: Device type, "cuda" or "cpu".
            **kwargs: Additional backend-specific arguments.
        """
        self.model_name = model_name
        self.device = device
        self.kwargs = kwargs

    @abstractmethod
    def init(self) -> None:
        """Initialize the model and processor.

        This method should be called before parse().
        """
        pass

    @abstractmethod
    def parse(
        self,
        input_data: Union[str, Image.Image, bytes],
        prompt: str,
        **kwargs,
    ) -> str:
        """Parse document and extract text content.

        Args:
            input_data: Input can be:
                - str: File path
                - PIL.Image.Image: Image object
                - bytes: Image bytes
            prompt: Prompt text for the model.
            **kwargs: Additional arguments passed to the model.

        Returns:
            Parsed text content.
        """
        pass

    @abstractmethod
    def parse_batch(
        self,
        input_data: list[Union[str, Image.Image, bytes]],
        prompt: str,
        **kwargs,
    ) -> list[str]:
        """Parse multiple documents.

        Args:
            input_data: List of inputs.
            prompt: Prompt text for the model.
            **kwargs: Additional arguments passed to the model.

        Returns:
            List of parsed text content.
        """
        pass
