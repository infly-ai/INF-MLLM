# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import random
from contextlib import contextmanager
from dataclasses import asdict, fields
from typing import Optional, List, Tuple
from PIL import Image
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.multimodal.image import convert_image_mode
from .utils import extract_markdown_content
import uuid


def apply_chat_template(question: str) -> str:

    placeholder = "<|image_pad|>"
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return prompt


class VllmBackend:
    def __init__(self, processor, args=None):

        default_engine_args = EngineArgs(
            model=getattr(args, "model", "Qwen/Qwen2.5-VL"),
            max_model_len=getattr(args, "max_model_len", 4096),
            max_num_seqs=getattr(args, "max_num_seqs", 5),
            mm_processor_kwargs=getattr(
                args,
                "mm_processor_kwargs",
                {"min_pixels": 28 * 28, "max_pixels": 1280 * 28 * 28, "fps": 1},
            ),
            limit_mm_per_prompt=getattr(args, "limit_mm_per_prompt", {"image": 1}),
            tensor_parallel_size=getattr(args, "tp", 1),
        )

        if args is not None:
            engine_kwargs = asdict(default_engine_args)
            arg_dict = vars(args)

            valid_fields = {f.name for f in fields(default_engine_args)}
            updates = {
                k: v for k, v in arg_dict.items() if v is not None and k in valid_fields
            }

            engine_kwargs.update(updates)
            self.engine_args = EngineArgs(**engine_kwargs)
        else:
            self.engine_args = default_engine_args

        self.processor = processor
        self.llm = LLM(**asdict(self.engine_args))

    def run(self, inputs: List[Tuple[str, str, Image.Image]], output: str | Path):

        llm_inputs = []
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=8192,
            stop_token_ids=[
                self.processor.tokenizer.eos_token_ids,
                self.processor.tokenizer.pad_token_ids,
            ],
            n=1,
        )

        file_names = []

        for file_name, entry, data in inputs:
            file_names.append(file_name)
            entry = apply_chat_template(entry)
            llm_inputs.append(
                {
                    "prompt_token_ids": self.processor(text=entry)["input_ids"][0],
                    "multi_modal_data": {"image": [data]},
                    "multi_modal_uuids": {"image": [str(uuid.uuid4())]},
                }
            )

        outputs = self.llm.generate(
            llm_inputs,
            sampling_params=sampling_params,
        )
        os.makedirs(output, exist_ok=True)
        result = []
        print(len(outputs))
        for idx, o in enumerate(outputs):
            md = self.processor.tokenizer.decode(o.outputs[0].token_ids)
            os.makedirs(Path(output) / file_names[idx], exist_ok=True)
            with open(Path(output) / file_names[idx] / "output.md", "w") as file:
                file.write(extract_markdown_content(md))
            result.append(md)
        
        return result
