import sys
import copy
import asyncio
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

import httpx
import aiofiles
import os
import numpy as np
import gradio as gr
from loguru import logger
from openai import AsyncOpenAI
import aiohttp
import uuid
import tqdm
import base64
import pathlib
from io import BytesIO
from pdf2image import convert_from_bytes, convert_from_path
from utils import (
    postprocess_doc2json_result,
    postprocess_doc2md_result,
    draw_bboxes_on_image,
    encode_image,
    images_to_pdf,
    package_results_as_zip,
    images_to_b64,
)
from prompts import _resolve_prompt, SUPPORTED_TASK_TYPES


def setup_poppler_linux():
    poppler_dir = "/tmp/poppler"
    if not os.path.exists(poppler_dir):
        os.makedirs(poppler_dir, exist_ok=True)
        subprocess.run(
            ["bash", "-lc", "rm -f /etc/apt/sources.list.d/*nodesource*.list || true"],
            check=False,
        )
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "poppler-utils"], check=True)


if sys.platform.startswith("linux"):
    setup_poppler_linux()

openai_api_key = "EMPTY"
openai_api_base = os.environ.get("INFINITY_API_BASE", "")
Authorization = os.environ.get("INFINITY_API_AUTH", "")
_http_client = httpx.AsyncClient(verify=False)

AVAILABLE_MODELS = {
    "Infinity-Parser2-Pro": {
        "name": "Infinity-Parser2-Pro",
        "client": AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base.rstrip("/") + "/v1",
            http_client=_http_client,
        ),
        "Authorization": Authorization,
    }
}


async def send_pdf_async_aiohttp(
    file_path, server_ip, route="/upload", Authorization=None
):
    """use aiohttp send pdf"""
    url = f"{server_ip}{route}"
    headers = {}
    if Authorization:
        headers["Authorization"] = f"Bearer {Authorization}"

    try:
        connector = aiohttp.TCPConnector(ssl=False)

        async with aiohttp.ClientSession(connector=connector) as session:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field(
                    "file",
                    f,
                    filename=os.path.basename(file_path),
                    content_type="application/pdf",
                )
                async with session.post(url, data=data, headers=headers) as response:
                    print(f"PDF sent successfully: {file_path}, status code: {response.status}")
                    return response
    except Exception as e:
        print(f"PDF sent failed: {file_path}, error: {e}")
        return None


async def request(messages, model_name, client, Authorization):

    chat_completion_from_base64 = await client.chat.completions.create(
        messages=messages,
        extra_headers={"Authorization": f"Bearer {Authorization}"},
        model=model_name,
        max_completion_tokens=4096,
        stream=True,
        temperature=0.0,
        top_p=0.95,
    )

    page = ""
    async for chunk in chat_completion_from_base64:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content

            choice = chunk.choices[0]
            if choice.finish_reason is not None:
                print(f"end reason = {choice.finish_reason}")
                break
            page += content

            yield content


def build_message(image_path, prompt):

    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"},
        },
        {"type": "text", "text": prompt},
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]

    return messages


async def doc_parser(doc_path, task_type, custom_prompt, model_id):
    model_name = AVAILABLE_MODELS[model_id]["name"]
    client = AVAILABLE_MODELS[model_id]["client"]
    Authorization = AVAILABLE_MODELS[model_id]["Authorization"]

    # get prompt
    prompt = _resolve_prompt(task_type, custom_prompt)

    doc_path = Path(doc_path)
    if not doc_path.is_file():
        raise FileNotFoundError(doc_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        queries = []
        img_paths = []  # save image paths for doc2json post processing
        if doc_path.suffix.lower() == ".pdf":
            pages = convert_from_path(doc_path, dpi=300)
            for idx, page in enumerate(pages, start=1):
                img_path = tmpdir / f"page_{idx}.png"
                page.save(img_path, "PNG")
                messages = build_message(img_path, prompt)
                queries.append(messages)
                img_paths.append(img_path)
        else:
            messages = build_message(doc_path, prompt)
            queries.append(messages)
            img_paths.append(doc_path)

        all_pages = []  # processed full text
        all_pages_raw = []  # raw full text
        all_bbox_images = (
            []
        )  # drawn bbox images (doc2json only), list of (PIL.Image, caption)

        for i, query in enumerate(queries):
            raw_page = ""
            async for chunk in request(query, model_name, client, Authorization):
                raw_page += chunk
                yield raw_page, [], raw_page, raw_page

            if task_type == "doc2json":
                processed = postprocess_doc2json_result(
                    raw_page, img_paths[i], output_format="md"
                )
                im = draw_bboxes_on_image(img_paths[i], raw_page)
                if im is not None:
                    all_bbox_images.append((im, f"Page {i+1}"))
            elif task_type == "doc2md":
                processed = postprocess_doc2md_result(raw_page)
            else:
                processed = raw_page

            all_pages.append(processed)
            all_pages_raw.append(raw_page)
            print(f"[page {i+1}] done, task_type={task_type}")

            final_processed = "\n---\n".join(all_pages)
            final_raw = "\n\n".join(all_pages_raw)

            yield final_processed, all_bbox_images, final_processed, final_raw


def compress_directory_to_zip(directory_path, output_zip_path):
    try:
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:

            for root, dirs, files in os.walk(directory_path):
                for file in files:

                    file_path = os.path.join(root, file)

                    arcname = os.path.relpath(file_path, directory_path)

                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


latex_delimiters = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
]


def check_task_input(task_type, custom_prompt):
    """validate: custom mode must fill in prompt"""
    if task_type == "custom" and (not custom_prompt or custom_prompt.strip() == ""):
        raise gr.Error("Please enter a custom prompt before parsing.")
    return task_type


def to_file(image_path):

    if image_path.endswith("Academic_Papers.png"):
        image_path = image_path.replace("Academic_Papers.png", "Academic_Papers.pdf")

    return image_path


def render_img(b64_list, idx, scale):
    """Render HTML based on current index idx and scale."""
    if not b64_list:
        return "<p style='color:gray'>Please upload an image first.</p>"
    idx %= len(b64_list)
    src = b64_list[idx]
    # return (
    #     f'<div style="overflow:auto;border:1px solid #ccc;'
    #     f'display:flex;justify-content:center;align-items:center;'   
    #     f'width:100%;height:800px;">'                               
    #     f'<img src="{src}" '
    #     f'style="transform:scale({scale});transform-origin:center center;" />'  
    #     f'</div>'
    # )

    percent = scale * 100

    if scale <= 1:
        return f"""
            <div style="
                width:100%;
                height:800px;
                overflow:auto;
                border:1px solid #ccc;
            ">
              <div style="
                  min-width:100%;           
                  display:flex;
                  justify-content:center;   
              ">
                <img src="{src}" style="
                    width:{percent}%;
                    height:auto;
                    display:block;
                ">
              </div>
            </div>
            """
    else:
        return (
            f'<div style="overflow:auto;border:1px solid #ccc;'
            f'width:100%;height:800px;">'
            f'  <img src="{src}" '
            f'       style="width:{percent}%;max-width:none;'
            f'              height:auto;display:block;" />'
            f"</div>"
        )


async def process_file(file_path):
    """Use asyncio for async processing."""
    if file_path is None:
        return None

    if not file_path.endswith(".pdf"):
        tmp_file_path = Path(file_path)
        tmp_file_path = tmp_file_path.with_suffix(".pdf")
        images_to_pdf(file_path, tmp_file_path)
    else:
        tmp_file_path = file_path
        asyncio.create_task(
            send_pdf_async_aiohttp(
                tmp_file_path, server_ip=openai_api_base, Authorization=Authorization
            )
        )

    return str(tmp_file_path)


def check_file(f):
    if f is None:
        raise gr.Error("Please upload a PDF or image before parsing.")
    return f


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(variant="panel", scale=5):

                file = gr.File(
                    label="Please upload a PDF or image",
                    file_types=[".pdf", ".png", ".jpeg", ".jpg"],
                    type="filepath",
                )

                task_selector = gr.Dropdown(
                    choices=SUPPORTED_TASK_TYPES,
                    label="Task Type",
                    info="doc2json / doc2md auto-load preset prompts; custom lets you write your own.",
                    value="doc2json",
                    interactive=True,
                )
                custom_prompt = gr.TextArea(
                    label="Custom Prompt",
                    placeholder="Enter your custom prompt here...",
                    lines=4,
                    visible=False,  # hidden by default, show when custom is selected
                )

                # switch task type to control custom prompt visibility
                def on_task_change(task_type):
                    return gr.update(visible=(task_type == "custom"))

                task_selector.change(
                    fn=on_task_change,
                    inputs=task_selector,
                    outputs=custom_prompt,
                )

                with gr.Row():
                    change_bu = gr.Button("Parse")
                    clear_bu = gr.ClearButton(value="Clear")

                zoom = gr.Slider(0.5, 3, value=1, step=0.1, label="Image Scale")
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Pre")
                    next_btn = gr.Button("Next ➡️")

                viewer = gr.HTML()

                example_root = os.path.join(os.path.dirname(__file__), "examples")
                images = [
                    os.path.join(example_root, f)
                    for f in os.listdir(example_root)
                    if f.lower().endswith(("png", "jpg", "jpeg"))
                ]

            with gr.Column(variant="panel", scale=5):

                model_selector = gr.Dropdown(
                    choices=[(k, k) for k, v in AVAILABLE_MODELS.items()],
                    value=list(AVAILABLE_MODELS.keys())[0],  # default to first model
                    label="Model Selection",
                    info="Select the model to use for parsing",
                    interactive=True,
                )

                with gr.Accordion("Examples", open=True):
                    example_root = "examples"
                    file_path = [
                        os.path.join(example_root, f)
                        for f in [
                            "Financial_Reports.png",
                            "Books.png",
                            "Magazines.png",
                            "Academic_Papers.png",
                        ]
                    ]

                    with gr.Row():
                        for i, label in enumerate(
                            [
                                "Financial Reports(IMG)",
                                "Books(IMG)",
                                "Magazines(IMG)",
                                "Academic Papers(PDF)",
                            ]
                        ):
                            with gr.Column(scale=1, min_width=120):
                                gr.Image(
                                    value=file_path[i],
                                    width=120,
                                    height=90,
                                    show_label=False,
                                )
                                gr.Button(label).click(
                                    fn=to_file,
                                    inputs=gr.State(file_path[i]),
                                    outputs=file,
                                )

                download_btn = gr.Button("⬇️ Generate download link", size="sm")
                output_file = gr.File(
                    label="Parse result",
                    interactive=False,
                    elem_id="down-file-box",
                    visible=False,
                )

                gr.HTML("""
                <style>
                #down-file-box {
                    max-height: 300px;
                }
                </style>
                """)
                with gr.Tabs():
                    with gr.Tab("Rendered result"):
                        md = gr.Markdown(
                            label="Markdown rendering",
                            height=1100,
                            latex_delimiters=latex_delimiters,
                            line_breaks=True,
                        )
                    with gr.Tab("Layout result"):
                        bbox_img = gr.Gallery(
                            label="Layout result",
                            columns=1,
                            rows=1,
                            height=1100,
                            object_fit="contain",
                            preview=True,
                        )
                    with gr.Tab("Processed result"):
                        processed_text = gr.TextArea(lines=45, label="Processed result")
                    with gr.Tab("Raw result"):
                        raw_text = gr.TextArea(lines=45, label="Raw result")

        img_list_state = gr.State([])
        idx_state = gr.State(0)

        async def upload_handler(files):

            if files is None:
                return [], 0, ""

            if files.lower().endswith(".pdf"):
                asyncio.create_task(
                    send_pdf_async_aiohttp(
                        files, server_ip=openai_api_base, Authorization=Authorization
                    )
                )

            b64s = images_to_b64(files)
            return b64s, 0, render_img(b64s, 0, 1)

        file.change(
            upload_handler,
            inputs=file,
            outputs=[img_list_state, idx_state, viewer],
        ).then(
            lambda: gr.update(value=1),  # no input, set zoom to 1
            None,  # inputs=None
            zoom,  # outputs=[zoom]
        )

        def show_prev(b64s, idx, scale):
            idx -= 1
            return idx, render_img(b64s, idx, scale)

        prev_btn.click(
            show_prev,
            inputs=[img_list_state, idx_state, zoom],
            outputs=[idx_state, viewer],
        )

        def show_next(b64s, idx, scale):
            idx += 1
            return idx, render_img(b64s, idx, scale)

        next_btn.click(
            show_next,
            inputs=[img_list_state, idx_state, zoom],
            outputs=[idx_state, viewer],
        )

        zoom.change(
            lambda b64s, idx, scale: render_img(b64s, idx, scale),
            inputs=[img_list_state, idx_state, zoom],
            outputs=viewer,
        )

        def auto_package_and_show(*args):
            # bind zip file path and component visibility together
            zip_path = package_results_as_zip(*args)
            return gr.update(value=zip_path, visible=True)

        change_bu.click(
            fn=check_task_input,
            inputs=[task_selector, custom_prompt],
            outputs=task_selector,
        ).then(fn=check_file, inputs=file, outputs=file).then(
            # hide and clear download box before starting
            lambda: gr.update(value=None, visible=False),
            inputs=None,
            outputs=output_file,
        ).then(
            fn=doc_parser,
            inputs=[file, task_selector, custom_prompt, model_selector],
            outputs=[md, bbox_img, processed_text, raw_text],
        )

        download_btn.click(
            fn=auto_package_and_show,
            inputs=[task_selector, processed_text, raw_text, bbox_img],
            outputs=output_file,
        )

        clear_bu.add([file, md, bbox_img, processed_text, raw_text])

    demo.launch(server_name="0.0.0.0", share=True)
