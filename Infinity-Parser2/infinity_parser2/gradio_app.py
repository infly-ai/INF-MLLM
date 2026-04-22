import sys
import asyncio
import tempfile
import subprocess
from pathlib import Path

import httpx
import os
import gradio as gr
from openai import AsyncOpenAI
import aiohttp
from pdf2image import convert_from_path
from utils import (
    postprocess_doc2json_result,
    postprocess_doc2md_result,
    draw_bboxes_on_image,
    encode_image,
    images_to_pdf,
    package_results_as_zip,
    images_to_b64,
)
from prompts import resolve_prompt, SUPPORTED_TASK_TYPES


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


class GradioApp:
    """Gradio 应用类，封装 Infinity-Parser2 的 Web UI"""

    LATEX_DELIMITERS = [
        {"left": "$$", "right": "$$", "display": True},
        {"left": "$", "right": "$", "display": False},
        {"left": "\\(", "right": "\\)", "display": False},
        {"left": "\\[", "right": "\\]", "display": True},
    ]

    def __init__(self):
        self.openai_api_key = "EMPTY"
        self.openai_api_base = os.environ.get("INFINITY_API_BASE", "")
        self.Authorization = os.environ.get("INFINITY_API_AUTH", "")
        self._http_client = httpx.AsyncClient(verify=False)
        self.available_models = self._init_models()
        self.demo = None

    def _init_models(self):
        """初始化可用模型"""
        return {
            "Infinity-Parser2-Pro": {
                "name": "Infinity-Parser2-Pro",
                "client": AsyncOpenAI(
                    api_key=self.openai_api_key,
                    base_url=self.openai_api_base.rstrip("/") + "/v1",
                    http_client=self._http_client,
                ),
                "Authorization": self.Authorization,
            }
        }

    # ==================== core methods ====================

    async def send_pdf_async_aiohttp(
        self, file_path, server_ip, route="/upload", Authorization=None
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
                    async with session.post(
                        url, data=data, headers=headers
                    ) as response:
                        print(
                            f"PDF sent successfully: {file_path}, status code: {response.status}"
                        )
                        return response
        except Exception as e:
            print(f"PDF sent failed: {file_path}, error: {e}")
            return None

    async def request(self, messages, model_name, client, Authorization):
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

    def build_message(self, image_path, prompt):
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                },
            },
            {"type": "text", "text": prompt},
        ]
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]

    async def infinity_parser2(self, doc_path, task_type, custom_prompt, model_id):
        model_name = self.available_models[model_id]["name"]
        client = self.available_models[model_id]["client"]
        Authorization = self.available_models[model_id]["Authorization"]

        prompt = resolve_prompt(task_type, custom_prompt)
        doc_path = Path(doc_path)
        if not doc_path.is_file():
            raise FileNotFoundError(doc_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            queries = []
            img_paths = []
            if doc_path.suffix.lower() == ".pdf":
                pages = convert_from_path(doc_path, dpi=300)
                for idx, page in enumerate(pages, start=1):
                    img_path = tmpdir / f"page_{idx}.png"
                    page.save(img_path, "PNG")
                    messages = self.build_message(img_path, prompt)
                    queries.append(messages)
                    img_paths.append(img_path)
            else:
                messages = self.build_message(doc_path, prompt)
                queries.append(messages)
                img_paths.append(doc_path)

            all_pages = []
            all_pages_raw = []
            all_bbox_images = []

            for i, query in enumerate(queries):
                raw_page = ""
                async for chunk in self.request(
                    query, model_name, client, Authorization
                ):
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

    # ==================== static helper methods ====================

    @staticmethod
    def check_task_input(task_type, custom_prompt):
        """validate: custom mode must fill in prompt"""
        if task_type == "custom" and (not custom_prompt or custom_prompt.strip() == ""):
            raise gr.Error("Please enter a custom prompt before parsing.")
        return task_type

    @staticmethod
    def to_file(image_path):
        """Convert image to file path."""
        if image_path.endswith("Academic_Papers.png"):
            image_path = image_path.replace(
                "Academic_Papers.png", "Academic_Papers.pdf"
            )
        return image_path

    @staticmethod
    def render_img(b64_list, idx, scale):
        """Render HTML based on current index idx and scale."""
        if not b64_list:
            return "<p style='color:gray'>Please upload an image first.</p>"
        idx %= len(b64_list)
        src = b64_list[idx]
        percent = scale * 100

        if scale <= 1:
            return f"""
                <div style="width:100%;height:800px;overflow:auto;border:1px solid #ccc;">
                  <div style="min-width:100%;display:flex;justify-content:center;">
                    <img src="{src}" style="width:{percent}%;height:auto;display:block;">
                  </div>
                </div>
                """
        else:
            return (
                f'<div style="overflow:auto;border:1px solid #ccc;width:100%;height:800px;">'
                f'  <img src="{src}" style="width:{percent}%;max-width:none;height:auto;display:block;" />'
                f"</div>"
            )

    @staticmethod
    def check_file(f):
        if f is None:
            raise gr.Error("Please upload a PDF or image before parsing.")
        return f

    @staticmethod
    def on_task_change(task_type):
        """task type change, control custom prompt visibility"""
        return gr.update(visible=(task_type == "custom"))

    @staticmethod
    def reset_zoom():
        """reset zoom value"""
        return gr.update(value=1)

    @staticmethod
    def hide_download_file():
        """hide download file box"""
        return gr.update(value=None, visible=False)

    # ==================== callback methods ====================

    async def upload_handler(self, files):
        """file upload handler"""
        if files is None:
            return [], 0, ""

        if files.lower().endswith(".pdf"):
            asyncio.create_task(
                self.send_pdf_async_aiohttp(
                    files,
                    server_ip=self.openai_api_base,
                    Authorization=self.Authorization,
                )
            )

        b64s = images_to_b64(files)
        return b64s, 0, self.render_img(b64s, 0, 1)

    def show_prev(self, b64s, idx, scale):
        """show previous page"""
        idx -= 1
        return idx, self.render_img(b64s, idx, scale)

    def show_next(self, b64s, idx, scale):
        """show next page"""
        idx += 1
        return idx, self.render_img(b64s, idx, scale)

    def on_zoom_change(self, b64s, idx, scale):
        """zoom change, update view"""
        return self.render_img(b64s, idx, scale)

    @staticmethod
    def package_zip(task_type, processed_text, raw_text, bbox_img):
        """package zip file and return update with visibility"""
        zip_path = package_results_as_zip(task_type, processed_text, raw_text, bbox_img)
        return gr.update(value=zip_path, visible=True)

    # ==================== UI components ====================

    def _build_left_column(self):
        """build left column"""
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
            visible=False,
        )

        with gr.Row():
            change_bu = gr.Button("Parse")
            clear_bu = gr.ClearButton(value="Clear")

        zoom = gr.Slider(0.5, 3, value=1, step=0.1, label="Image Scale")
        with gr.Row():
            prev_btn = gr.Button("⬅️ Pre")
            next_btn = gr.Button("Next ➡️")

        viewer = gr.HTML()

        return (
            file,
            task_selector,
            custom_prompt,
            change_bu,
            clear_bu,
            zoom,
            prev_btn,
            next_btn,
            viewer,
        )

    def _build_right_column(self, file, demo_data_root):
        """build right column"""
        model_selector = gr.Dropdown(
            choices=[(k, k) for k, v in self.available_models.items()],
            value=list(self.available_models.keys())[0],
            label="Model Selection",
            info="Select the model to use for parsing",
            interactive=True,
        )

        with gr.Accordion("Examples", open=True):
            file_paths = [
                os.path.join(demo_data_root, f)
                for f in [
                    "Financial_Reports.png",
                    "Books.png",
                    "Magazines.png",
                    "Academic_Papers.png",
                ]
            ]

            labels = [
                "Financial Reports(IMG)",
                "Books(IMG)",
                "Magazines(IMG)",
                "Academic Papers(PDF)",
            ]

            with gr.Row():
                for i, label in enumerate(labels):
                    with gr.Column(scale=1, min_width=120):
                        gr.Image(
                            value=file_paths[i],
                            width=120,
                            height=90,
                            show_label=False,
                        )
                        gr.Button(label).click(
                            fn=self.to_file,
                            inputs=gr.State(file_paths[i]),
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
                max-height: 80px;
            }
            </style>
        """)

        with gr.Tabs():
            with gr.Tab("Rendered result"):
                md = gr.Markdown(
                    label="Markdown rendering",
                    height=1100,
                    latex_delimiters=self.LATEX_DELIMITERS,
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

        return (
            model_selector,
            download_btn,
            output_file,
            md,
            bbox_img,
            processed_text,
            raw_text,
        )

    def _bind_events(
        self,
        file,
        task_selector,
        custom_prompt,
        change_bu,
        clear_bu,
        zoom,
        prev_btn,
        next_btn,
        viewer,
        model_selector,
        download_btn,
        output_file,
        md,
        bbox_img,
        processed_text,
        raw_text,
    ):
        """bind all events"""
        img_list_state = gr.State([])
        idx_state = gr.State(0)

        # file upload event
        file.change(
            self.upload_handler,
            inputs=file,
            outputs=[img_list_state, idx_state, viewer],
        ).then(
            self.reset_zoom,
            inputs=None,
            outputs=zoom,
        )

        # task type change
        task_selector.change(
            fn=self.on_task_change,
            inputs=task_selector,
            outputs=custom_prompt,
        )

        # prev btn click
        prev_btn.click(
            self.show_prev,
            inputs=[img_list_state, idx_state, zoom],
            outputs=[idx_state, viewer],
        )

        next_btn.click(
            self.show_next,
            inputs=[img_list_state, idx_state, zoom],
            outputs=[idx_state, viewer],
        )

        # zoom change
        zoom.change(
            self.on_zoom_change,
            inputs=[img_list_state, idx_state, zoom],
            outputs=viewer,
        )

        # parse button click
        change_bu.click(
            fn=self.check_task_input,
            inputs=[task_selector, custom_prompt],
            outputs=task_selector,
        ).then(
            fn=self.check_file,
            inputs=file,
            outputs=file,
        ).then(
            self.hide_download_file,
            inputs=None,
            outputs=output_file,
        ).then(
            fn=self.infinity_parser2,
            inputs=[file, task_selector, custom_prompt, model_selector],
            outputs=[md, bbox_img, processed_text, raw_text],
        )

        # download button click
        download_btn.click(
            fn=self.package_zip,
            inputs=[task_selector, processed_text, raw_text, bbox_img],
            outputs=output_file,
        )

        # clear button click
        clear_bu.add([file, md, bbox_img, processed_text, raw_text])

    def _build_ui(self):
        """build Gradio UI"""
        demo_data_root = os.path.join(os.path.dirname(__file__), "..", "demo_data")

        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(variant="panel", scale=5):
                    left_components = self._build_left_column()

                with gr.Column(variant="panel", scale=5):
                    right_components = self._build_right_column(
                        left_components[0], demo_data_root
                    )

            self._bind_events(
                *left_components,
                *right_components,
            )

        return demo

    def run(self, server_name="0.0.0.0", share=True):
        """run Gradio app"""
        self.demo = self._build_ui()
        self.demo.launch(server_name=server_name, share=share)


def main():
    """main function entry"""
    app = GradioApp()
    app.run()


if __name__ == "__main__":
    main()
