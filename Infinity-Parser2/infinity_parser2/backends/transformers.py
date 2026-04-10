"""Transformers backend for Infinity-Parser2."""

from typing import Union

from PIL import Image
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor

from ..utils import convert_pdf_to_images, load_image
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
            **kwargs: Additional arguments for AutoModelForCausalLM.from_pretrained.
        """
        super().__init__(model_name, device, **kwargs)
        self.torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.init()

    def init(self) -> None:
        """Initialize the model and processor."""
        # model_name 可以是 HuggingFace 模型 ID 或本地路径
        self._model = AutoModelForCausalLM.from_pretrained(
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
    ) -> tuple[list[str], list[Image.Image], list]:
        """Process inputs into messages, images, and video lists.

        Returns:
            Tuple of (texts, image_inputs, video_inputs).
        """
        images = [load_image(item) for item in inputs]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
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
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 32768),
            temperature=kwargs.get("temperature", 0.01),
            top_p=kwargs.get("top_p", 0.95),
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
        indices = list(range(len(input_data)))

        for i in range(0, len(input_data), batch_size):
            batch = input_data[i : i + batch_size]
            batch_indices = indices[i : i + batch_size]
            texts, image_inputs, video_inputs = self._process_inputs(batch, prompt)
            batch_results = self._generate(texts, image_inputs, video_inputs, **kwargs)
            for idx, result in zip(batch_indices, batch_results):
                results[idx] = result

        return results
