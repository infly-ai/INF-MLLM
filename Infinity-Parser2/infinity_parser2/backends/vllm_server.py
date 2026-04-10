"""vLLM Server backend for Infinity-Parser2.

Uses vLLM OpenAI-Compatible Server for online inference.
"""

import base64
from typing import Union

from PIL import Image

from .base import BaseBackend


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

    def init(self) -> None:
        """Validate server connection.

        Note: This is a no-op as the server is started separately.
        Call this to verify connectivity.
        """
        import requests

        health_url = self.api_url.replace("/v1/chat/completions", "/health")
        try:
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Cannot connect to vLLM server at {self.api_url}. "
                f"Please ensure the server is running. Error: {e}"
            )

    def _prepare_image_content(
        self, input_data: Union[str, Image.Image, bytes]
    ) -> dict:
        """Prepare image content for API request.

        Returns:
            Dictionary with image_url or image data.
        """
        import io

        if isinstance(input_data, str):
            with open(input_data, "rb") as f:
                file_bytes = f.read()
            ext = input_data.split(".")[-1].lower()
            if ext == "pdf":
                mime_type = "application/pdf"
            else:
                mime_type = f"image/{ext}"
        elif isinstance(input_data, bytes):
            file_bytes = input_data
            mime_type = "image/png"
        elif isinstance(input_data, Image.Image):
            buf = io.BytesIO()
            input_data.save(buf, format="PNG")
            file_bytes = buf.getvalue()
            mime_type = "image/png"
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        base64_data = base64.b64encode(file_bytes).decode()
        return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}}

    def parse(
        self,
        input_data: Union[str, Image.Image, bytes],
        prompt: str,
        **kwargs,
    ) -> str:
        """Parse a single document via HTTP API.

        Args:
            input_data: File path, PIL Image, or bytes.
            prompt: Prompt text for the model.
            **kwargs: Additional arguments (max_tokens, temperature, etc.).

        Returns:
            Parsed text content.
        """
        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        image_content = self._prepare_image_content(input_data)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        image_content,
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": kwargs.get("max_new_tokens", kwargs.get("max_tokens", 1024)),
            "temperature": kwargs.get("temperature", 0.0),
        }

        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def parse_batch(
        self,
        input_data: list[Union[str, Image.Image, bytes]],
        prompt: str,
        **kwargs,
    ) -> list[str]:
        """Parse multiple documents via HTTP API.

        Args:
            input_data: List of file paths, PIL Images, or bytes.
            prompt: Prompt text for the model.
            **kwargs: Additional arguments.

        Returns:
            List of parsed text content.
        """
        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        results = []
        for item in input_data:
            image_content = self._prepare_image_content(item)

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            image_content,
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": kwargs.get("max_new_tokens", kwargs.get("max_tokens", 1024)),
                "temperature": kwargs.get("temperature", 0.0),
            }

            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            results.append(result["choices"][0]["message"]["content"])

        return results
