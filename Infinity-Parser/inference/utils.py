import re
import os
from PIL import Image
from typing import Optional, List, Tuple
from pathlib import Path

def extract_markdown_content(text):
    matches = re.search(r"```markdown\n(.*?)\n```", text, re.DOTALL)
    if matches:
        text = matches.group(1).strip()
    return text


def update_config_from_args(config, args):
    """
    动态更新 config 属性，仅在 config 存在该属性时才更新。
    """
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    return config


def load_inputs(input_path: str, prompt: str) -> List[Tuple[str, Image.Image]]:
    inputs = []
    # TODO: support json input
    if input_path.endswith(".json") and False:
        print(f"📜 Loading JSON file: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if "file" not in item:
                raise ValueError(f"Missing 'file' field in JSON element: {item}")
            file_path = item["file"]

            if not os.path.exists(file_path):
                print(f"⚠️  File not found: {file_path}")
                continue

            if file_path.lower().endswith(".pdf"):
                images = convert_from_path(file_path, dpi=200)
                for img in images:
                    inputs.append((prompt, img))
            elif file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                inputs.append((prompt, Image.open(file_path)))
            else:
                print(f"⚠️  Unsupported file type in JSON: {file_path}")

    elif input_path.lower().endswith(".pdf"):
        print(f"📄 Converting PDF to images: {input_path}")
        images = convert_from_path(input_path, dpi=200)
        for idx, img in enumerate(images):
            inputs.append((Path(input_path).stem + f"page_{idx+1}", prompt, img))

    elif os.path.isfile(input_path) and input_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        inputs.append((Path(input_path).stem, prompt, Image.open(input_path)))

    elif os.path.isdir(input_path):
        print(f"📁 Scanning directory: {input_path}")
        for root, _, files in os.walk(input_path):
            for name in sorted(files):
                file_path = os.path.join(root, name)
                if file_path.lower().endswith(".pdf"):
                    images = convert_from_path(file_path, dpi=200)
                    for idx, img in enumerate(images):
                        inputs.append((Path(file_path).stem + f"page_{idx+1}", prompt, img))
                elif file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    inputs.append((Path(file_path).stem, prompt, Image.open(file_path)))

    else:
        raise ValueError(f"❌ Unsupported input path: {input_path}")

    print(f"🧩 Loaded {len(inputs)} document pages from {input_path}")
    return inputs