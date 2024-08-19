import os, sys
import re
import torch
from PIL import Image 
import requests
import numpy as np
import random
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def expand2square(pil_img, background_color):
    # pad to middle for square shape
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def padding_336(b):
    width, height = b.size
    tar = int(np.ceil(height / 336) * 336)
    top_padding = int((tar - height)/2)
    bottom_padding = tar - height - top_padding

    left_padding = 0
    right_padding = 0

    mean_fill = 255*[0.48145466, 0.4578275, 0.40821073]
    b = transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255,255,255])

    return b

def HD_transform(img, hd_num=9):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width/ height)
    scale = int(np.ceil(width/336))
    # print(width, height, ratio, scale, scale*np.ceil(scale/ratio))
    while scale*np.ceil(scale/ratio) > hd_num:
        scale -= 1
        # print(scale*np.ceil(scale/ratio))
    new_w = int(scale * 336)
    new_h = int(new_w / ratio)

    img = transforms.functional.resize(img, [new_h, new_w],)
    img = padding_336(img)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img

class ImageTestProcessorHD:
    def __init__(self, image_size=224, mean=None, std=None, hd_num=-1):
        if mean is None:
            self.mean = mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            self.std = std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)
        self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        self.hd_num = hd_num

    def __call__(self, item):
        return self.transform(HD_transform(item, hd_num=self.hd_num))

def main(args):
    disable_torch_init()
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    model = model.cuda().eval()
    image_processor = ImageTestProcessorHD(336, hd_num=16)
    from bigmodelvis import Visualization
    Visualization(model).structure_graph()

    questions = [
        '将图中表格转成html格式.',
        '请解析输入的文档.'
    ]

    raw_image = Image.open('../infmllm2/docs/doc_02.png').convert('RGB') 
    image_tensor = image_processor(raw_image).cuda()

    history = []

    print("\n" + "=" * 20)
    for i, question in enumerate(questions):
        history.append({
            'from': 'human',
            'value': question,
        })
        history.append(
        {"from": 'gpt', "value": ""})
        samples = {
            'images': [image_tensor.unsqueeze(0)],
            'conversations': [history]
        }
        with torch.inference_mode():
            pred_answers, prompts = model.generate(
                    samples=samples,
                    max_length=args.max_new_tokens,
                    min_length=1,
                    num_beams=args.num_beams,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    return_prompts=True
                )
        answer = pred_answers[0]
        print(f"Q{i+1}: {question}")
        print(f"A{i+1}: {answer}")
        history[-1]['value'] = answer

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./InfMLLM_7B_Chat")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    args = parser.parse_args()

    main(args)