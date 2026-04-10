"""vLLM Engine backend for Infinity-Parser2.

Uses vLLM's LLM class for offline batch inference.
"""

import base64
import io
from pathlib import Path
from typing import Union

from PIL import Image

from .base import BaseBackend


class VLLMEngineBackend(BaseBackend):
    """Offline inference backend using vLLM Engine.

    Uses vLLM's LLM class for local batch inference with tensor parallelism.
    Reference: https://docs.vllm.ai/en/latest/serving/offline_inference.html
    """

    def __init__(
        self,
        model_name: str = "infly/Infinity-Parser2-Pro",
        device: str = "cuda",
        tensor_parallel_size: int = 1,
        **kwargs,
    ):
        """Initialize vLLM Engine backend.

        Args:
            model_name: Model name or local path.
            device: Device type, "cuda" or "cpu".
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            **kwargs: Additional arguments for vllm.LLM.
        """
        super().__init__(model_name, device, **kwargs)
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None

    def init(self) -> None:
        """Initialize the vLLM LLM instance."""
        if self._llm is not None:
            return

        from vllm import LLM

        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            **self.kwargs,
        )

    def _encode_file(self, input_data: Union[str, bytes]) -> tuple[str, str]:
        """Encode file to base64 and determine MIME type.

        Returns:
            Tuple of (base64_data, mime_type).
        """
        if isinstance(input_data, str):
            with open(input_data, "rb") as f:
                file_bytes = f.read()
        else:
            file_bytes = input_data

        ext = Path(input_data).suffix.lower() if isinstance(input_data, str) else ""
        if ext == ".pdf":
            mime_type = "application/pdf"
        else:
            mime_type = f"image/{ext[1:]}" if ext else "image/png"

        base64_data = base64.b64encode(file_bytes).decode()
        return base64_data, mime_type

    def _build_messages(self, base64_data: str, mime_type: str, prompt: str) -> list:
        """Build chat messages with base64-encoded image.

        Returns:
            List of message dictionaries.
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _load_image(self, input_data: Union[str, bytes]) -> Image.Image:
        """Load image from file path or bytes."""
        if isinstance(input_data, str):
            return Image.open(input_data).convert("RGB")
        elif isinstance(input_data, bytes):
            return Image.open(io.BytesIO(input_data)).convert("RGB")
        elif isinstance(input_data, Image.Image):
            return input_data.convert("RGB")
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

    def parse(
        self,
        input_data: Union[str, Image.Image, bytes],
        prompt: str,
        **kwargs,
    ) -> str:
        """Parse a single document.

        Args:
            input_data: File path, PIL Image, or bytes.
            prompt: Prompt text for the model.
            **kwargs: Additional arguments (max_new_tokens, temperature, etc.).

        Returns:
            Parsed text content.
        """
        self.init()

        from vllm import SamplingParams

        if isinstance(input_data, (str, bytes)):
            base64_data, mime_type = self._encode_file(input_data)
        else:
            image = self._load_image(input_data)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            base64_data = base64.b64encode(img_byte_arr.getvalue()).decode()
            mime_type = "image/png"

        messages = self._build_messages(base64_data, mime_type, prompt)

        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_new_tokens", 1024),
            temperature=kwargs.get("temperature", 0.0),
        )

        outputs = self._llm.chat([messages], sampling_params=sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

    def parse_batch(
        self,
        input_data: list[Union[str, Image.Image, bytes]],
        prompt: str,
        **kwargs,
    ) -> list[str]:
        """Parse multiple documents.

        Args:
            input_data: List of file paths, PIL Images, or bytes.
            prompt: Prompt text for the model.
            **kwargs: Additional arguments.

        Returns:
            List of parsed text content.
        """
        return [self.parse(item, prompt, **kwargs) for item in input_data]
