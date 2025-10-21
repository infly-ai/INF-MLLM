import argparse
import os
import json
import math
from typing import List, Tuple
from dataclasses import dataclass, field
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from .vllm_backend import VllmBackend
from .consant import PROMPT
from .utils import load_inputs, update_config_from_args
from transformers import AutoProcessor
from pdf2image import convert_from_path


@dataclass
class Config:
    model: str
    max_model_len: int = 4096
    min_pixels: int = 28 * 28
    max_pixels: int = 1280 * 28 * 28
    fps: int = 1
    tp: int = 1

    @property
    def mm_processor_kwargs(self):
        return {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "fps": self.fps,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Infinity-Parser CLI for document-to-markdown conversion"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output Folder")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")

    parser.add_argument("--batch_size", type=int, default=128, help="batch size")

    args = parser.parse_args()

    print(f"🚀 Loading model from {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    config = Config(model=args.model)
    config = update_config_from_args(config, args)
    vllm_backend = VllmBackend(processor, config)

    print(f"📂 Reading input file: {args.input}")
    inputs = load_inputs(args.input, PROMPT)
    print(f"🧩 Loaded {len(inputs)} document images")
    batch_size = args.batch_size
    num_batches = math.ceil(len(inputs) / batch_size)

    print(f"⚙️ Running inference in {num_batches} batches (batch_size={batch_size}) ...")

    all_outputs = []
    for i in tqdm(range(num_batches), desc="Batch inference"):
        batch_inputs = inputs[i * batch_size : (i + 1) * batch_size]
        outputs = vllm_backend.run(batch_inputs, args.output)
        print(outputs)
        all_outputs.extend(outputs)

    print(f"✅ Done. Total processed: {len(all_outputs)} samples.")


if __name__ == "__main__":
    main()
