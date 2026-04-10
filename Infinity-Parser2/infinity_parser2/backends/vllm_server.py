"""vLLM Server backend for Infinity-Parser2.

Uses vLLM OpenAI-Compatible Server for online inference.
"""

from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Union

import requests
from PIL import Image

from .base import BaseBackend
from ..utils import encode_file_to_base64


class VLLMServerBackend(BaseBackend):
    """Online inference backend using vLLM OpenAI-Compatible Server.

    Sends requests to a running vLLM server via HTTP API.
    Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """

    def __init__(
        self,
        model_name: str = "infly/Infinity-Parser2-Pro",
        api_url: str = "http://localhost:8000/v1/chat/completions",
        api_key: str = "EMPTY",
        timeout: int = 300,
        min_pixels: int = 2048,
        max_pixels: int = 16777216,
        **kwargs,
    ):
        """Initialize vLLM Server backend.

        Args:
            model_name: Model name (must match server).
            api_url: Full URL to the chat completions endpoint.
            api_key: API key for authentication.
            timeout: Request timeout in seconds.
            **kwargs: Additional arguments for requests.
        """
        super().__init__(model_name, "cuda", **kwargs)
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.init()

    def init(self) -> None:
        """Validate server connection.

        Note: This is a no-op as the server is started separately.
        Call this to verify connectivity.
        """
        health_url = self.api_url.replace("/v1/chat/completions", "/health")
        try:
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Cannot connect to vLLM server at {self.api_url}. "
                f"Please ensure the server is running. Error: {e}"
            )

    def parse_batch(
        self,
        input_data: list[Union[str, Image.Image]],
        prompt: str,
        batch_size: int = 1,
        **kwargs,
    ) -> list[str]:
        """Parse multiple documents via HTTP API with batched requests.

        Args:
            input_data: List of file paths or PIL Images.
            prompt: Prompt text for the model.
            batch_size: Maximum number of images to process in one batch.
            **kwargs: Additional arguments.

        Returns:
            List of parsed text content (one per input in the same order).
        """
        if not input_data:
            return []

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        max_tokens = kwargs.get("max_new_tokens", kwargs.get("max_tokens", 32768))
        temperature = kwargs.get("temperature", 0.01)
        top_p = kwargs.get("top_p", 0.95)

        def parse_one(item: Union[str, Image.Image]) -> str:
            base64_data, mime_type = encode_file_to_base64(item, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        results: list[str] = [None] * len(input_data)
        max_workers = max(1, batch_size)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(parse_one, item): idx
                for idx, item in enumerate(input_data)
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to parse input at index {idx}: {e}"
                    ) from e

        return results
