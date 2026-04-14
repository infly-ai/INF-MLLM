"""Transformers backend for Infinity-Parser2."""

import sys
from typing import Union

from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from ..utils import load_image
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
        min_pixels: int = 2048,
        max_pixels: int = 16777216,
        **kwargs,
    ):
        """Initialize Transformers backend.

        Args:
            model_name: Model name on HuggingFace Hub or local path.
            device: Device type, "cuda" or "cpu".
            torch_dtype: Data type for model weights, "float16" or "bfloat16".
            min_pixels: Minimum number of pixels for image input.
            max_pixels: Maximum number of pixels for image input.
            **kwargs: Additional arguments for AutoModelForImageTextToText.from_pretrained.
        """
        super().__init__(model_name, device, **kwargs)
        self.torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.init()

    def init(self) -> None:
        """Initialize the model and processor."""
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            **self.kwargs,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)

    def _process_inputs(
        self,
        inputs: list[Union[str, Image.Image]],
        prompt: str,
        **kwargs,
    ) -> dict:
        """Process inputs for generation.

        Returns:
            Dictionary with processed inputs for the model.
        """
        images = [load_image(item) for item in inputs]

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for img in images
        ]

        chat_template_kwargs = {"enable_thinking": False}

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        image_inputs, _ = process_vision_info(messages, image_patch_size=16)

        inputs = self._processor(
            text=text,
            images=image_inputs,
            do_resize=False,
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        return inputs

    def _generate(self, inputs: dict, **kwargs) -> list[str]:
        """Run model generation and decode outputs.

        Args:
            inputs: Processed inputs from _process_inputs.
            **kwargs: Generation arguments.

        Returns:
            List of generated text outputs.
        """
        # Move tensors to device
        inputs = {
            k: v.to(self._model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 32768),
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

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
        results = [None] * len(input_data)

        for i in tqdm(range(0, len(input_data), batch_size), desc="Parsing", file=sys.stdout):
            batch = input_data[i : i + batch_size]
            inputs = self._process_inputs(batch, prompt, **kwargs)
            batch_results = self._generate(inputs, **kwargs)
            for j, result in enumerate(batch_results):
                results[i + j] = result

        return results
