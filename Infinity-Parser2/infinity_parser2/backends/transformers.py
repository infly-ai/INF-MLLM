"""Transformers backend for Infinity-Parser2."""

import io
from pathlib import Path
from typing import Union

from PIL import Image
import torch

from .base import BaseBackend


class TransformersBackend(BaseBackend):
    """Inference backend using HuggingFace transformers.

    Supports local model inference with automatic device mapping.
    """

    def __init__(
        self,
        model_name: str = "infly/Infinity-Parser2-Pro",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        """Initialize Transformers backend.

        Args:
            model_name: Model name or local path.
            device: Device type, "cuda" or "cpu".
            torch_dtype: Data type for model weights, "float16" or "bfloat16".
            **kwargs: Additional arguments for AutoModelForCausalLM.from_pretrained.
        """
        super().__init__(model_name, device, **kwargs)
        self.torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)
        self._model = None
        self._processor = None

    def init(self) -> None:
        """Initialize the model and processor."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            **self.kwargs,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)

    def _load_image(self, input_data: Union[str, bytes]) -> Image.Image:
        """Load image from file path or bytes."""
        if isinstance(input_data, str):
            ext = Path(input_data).suffix.lower()
            if ext == ".pdf":
                from pypdf import PdfReader

                reader = PdfReader(input_data)
                page = reader.pages[0]
                raise NotImplementedError(
                    "PDF rendering requires additional setup. "
                    "Please convert PDF to images first."
                )
            return Image.open(input_data).convert("RGB")
        elif isinstance(input_data, bytes):
            return Image.open(io.BytesIO(input_data)).convert("RGB")
        elif isinstance(input_data, Image.Image):
            return input_data.convert("RGB")
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

    def _process_inputs(
        self,
        inputs: list[Union[str, Image.Image, bytes]],
        prompt: str,
    ) -> tuple[list[str], list[Image.Image], list]:
        """Process inputs into messages, images, and video lists.

        Returns:
            Tuple of (texts, image_inputs, video_inputs).
        """
        from qwen_vl_utils import process_vision_info

        images = [self._load_image(item) for item in inputs]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
            for img in images
        ]

        texts = [
            self._processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        all_image_inputs, all_video_inputs = [], []
        for msg in messages:
            img_inp, vid_inp = process_vision_info([msg])
            all_image_inputs.extend(img_inp if img_inp else [])
            all_video_inputs.extend(vid_inp if vid_inp else [])

        return texts, all_image_inputs, all_video_inputs

    def _generate(self, texts: list[str], image_inputs: list, video_inputs: list, **kwargs) -> list[str]:
        """Run model generation and decode outputs.

        Args:
            texts: Processed text prompts.
            image_inputs: Processed image inputs.
            video_inputs: Processed video inputs.
            **kwargs: Generation arguments.

        Returns:
            List of generated text outputs.
        """
        inputs = self._processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self._model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        generated_ids = self._model.generate(
            **inputs, max_new_tokens=kwargs.get("max_new_tokens", 1024)
        )

        results = []
        for i, text in enumerate(texts):
            input_len = len(self._processor.tokenizer(text)["input_ids"])
            generated = generated_ids[i][input_len:]
            output_text = self._processor.batch_decode(
                [generated], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            results.append(output_text)
        return results

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
        texts, image_inputs, video_inputs = self._process_inputs([input_data], prompt)
        results = self._generate(texts, image_inputs, video_inputs, **kwargs)
        return results[0]

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
        self.init()
        texts, image_inputs, video_inputs = self._process_inputs(input_data, prompt)
        return self._generate(texts, image_inputs, video_inputs, **kwargs)
