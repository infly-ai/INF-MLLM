import os, sys 
rootdir = os.path.abspath(os.path.dirname(__file__))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

import re
import torch
from PIL import Image 
import requests
from transformers import AutoModel, AutoTokenizer

from evaluate.infmllm_chat.utils import tokenizer_image_token
from evaluate.infmllm_chat.conversation import conv_templates, SeparatorStyle

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

def get_prompt(conv_mode, question, history=[]):
    conv = conv_templates[conv_mode].copy()
    if len(history) == 0:
        question = DEFAULT_IMAGE_TOKEN + '\n' + question
    else:
        if DEFAULT_IMAGE_TOKEN not in history[0][0]:
            history[0][0] = DEFAULT_IMAGE_TOKEN + '\n' + history[0][0]

    for qa in history:
        conv.append_message(conv.roles[0], qa[0])
        conv.append_message(conv.roles[1], qa[1])

    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    return prompt

def generate(model, tokenizer, stop_str, input_ids, image_tensor):
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True)
        
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    return outputs
        

def main(args):
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.cuda().eval()
    image_processor = model.get_model().get_vision_tower().image_processor

    stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2   # </s>

    img_url = 'https://farm5.staticflickr.com/4016/4349416002_e3743125b7_z.jpg'
    questions = [
        'Why this image is interesting ?',
        'What is the cat watching ?',
        'What is the scientific name of the bird in the picture?',
        'How is the weather outside?',
        'what season is it now ?'
    ]

    print(img_url)

    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')  
    image = expand2square(raw_image, tuple(int(x*255) for x in image_processor.image_mean))
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    history = []

    print("\n" + "=" * 20)
    for i, question in enumerate(questions):
        prompt = get_prompt(args.conv_mode, question, history)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        answer = generate(model, tokenizer, stop_str, input_ids, image_tensor)

        print(f"Q{i+1}: {question}")
        print(f"A{i+1}: {answer}")
        history.append([question, answer])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./InfMLLM_7B_Chat")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()
    
    main(args)

