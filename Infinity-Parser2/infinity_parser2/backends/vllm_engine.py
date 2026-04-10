"""vLLM Engine backend for Infinity-Parser2.

Uses vLLM's LLM class for offline batch inference.
"""

from typing import Union

from PIL import Image
from vllm import LLM, SamplingParams

from .base import BaseBackend
from ..utils import encode_file_to_base64


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
        self.init()

    def init(self) -> None:
        """Initialize the vLLM LLM instance."""
        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            **self.kwargs,
        )

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

    def parse_batch(
        self,
        input_data: list[Union[str, Image.Image]],
        prompt: str,
        batch_size: int = 1,
        **kwargs,
    ) -> list[str]:
        """Parse multiple documents with batched inference.

        Args:
            input_data: List of file paths or PIL Images.
            prompt: Prompt text for the model.
            batch_size: Maximum number of images to process in one batch.
            **kwargs: Additional arguments.

        Returns:
            List of parsed text content (one per input in the same order).
        """

        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_new_tokens", 1024),
            temperature=kwargs.get("temperature", 0.0),
        )

        all_messages = []
        for item in input_data:
            base64_data, mime_type = encode_file_to_base64(item)
            all_messages.append(self._build_messages(base64_data, mime_type, prompt))

        results = [None] * len(input_data)
        for i in range(0, len(all_messages), batch_size):
            batch_messages = all_messages[i : i + batch_size]
            outputs = self._llm.chat(
                [batch_messages],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            for j, output in enumerate(outputs[0].outputs):
                results[i + j] = output.text

        return results
