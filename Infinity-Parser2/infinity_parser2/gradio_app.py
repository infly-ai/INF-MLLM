import uuid
import tempfile
import base64
from pathlib import Path
import httpx
import os
import gradio as gr
from utils import (
    convert_pdf_to_images,
    postprocess_doc2json_result,
    postprocess_doc2md_result,
    draw_bboxes_on_image,
    package_results_as_zip,
    encode_image_to_base64,
)
from prompts import SUPPORTED_TASK_TYPES


class GradioApp:
    """Gradio Application Class, encapsulating the Web UI of Infinity-Parser2"""

    LATEX_DELIMITERS = [
        {"left": "$$", "right": "$$", "display": True},
        {"left": "$", "right": "$", "display": False},
        {"left": "\\(", "right": "\\)", "display": False},
        {"left": "\\[", "right": "\\]", "display": True},
    ]

    def __init__(self):
        self.openai_api_base = os.environ.get("INFINITY_API_BASE", "")
        self.Authorization = os.environ.get("INFINITY_API_AUTH", "")
        self._http_client = httpx.AsyncClient(verify=False, timeout=600.0)
        self.available_models = self._init_models()
        self.demo = None

    def _init_models(self):
        return ["Infinity-Parser2-Pro", "Infinity-Parser2-Flash"]

    # ==================== core methods ====================

    async def request_with_file_content(
        self,
        file_base64,
        file_name,
        task_type,
        custom_prompt,
        output_format,
        model_name,
    ):
        """Use Base64 string directly to request the Flask API, no local file path needed."""
        url = f"{self.openai_api_base}/v1/chat/completions"

        ext = Path(file_name).suffix.lower()
        content_type_map = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        content_type = content_type_map.get(ext, "application/octet-stream")

        payload = {
            "file_base64": file_base64,
            "file_name": file_name,
            "content_type": content_type,
            "task_type": task_type,
            "output_format": output_format,
            "model": model_name,
        }
        if custom_prompt:
            payload["custom_prompt"] = custom_prompt

        headers = {"Content-Type": "application/json"}
        if self.Authorization:
            headers["Authorization"] = f"Bearer {self.Authorization}"

        response = await self._http_client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content

    async def infinity_parser2(self, file_state, task_type, custom_prompt, model_name):
        """Parse using Base64 data in State; fully decouples from original upload path."""
        if not file_state:
            raise gr.Error("File state lost, please re-upload.")

        file_base64, file_name = file_state
        output_format = "json" if task_type == "doc2json" else "md"

        session_id = uuid.uuid4().hex
        session_dir = Path(tempfile.gettempdir()) / f"infinity_parse_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Decode Base64 to local temp file, used only for PDF-to-image and bbox drawing.
        file_bytes = base64.b64decode(file_base64)
        ext = Path(file_name).suffix.lower()

        img_paths = []
        if ext == ".pdf":
            MAX_PAGES = 10
            temp_pdf = session_dir / file_name
            with open(temp_pdf, "wb") as f:
                f.write(file_bytes)

            pages = convert_pdf_to_images(temp_pdf, dpi=300)
            pages = pages[:MAX_PAGES]
            for idx, page in enumerate(pages, start=1):
                img_path = session_dir / f"parse_page_{idx}.png"
                page.save(img_path, "PNG")
                img_paths.append(str(img_path))
        else:
            temp_img = session_dir / file_name
            with open(temp_img, "wb") as f:
                f.write(file_bytes)
            img_paths = [str(temp_img)]

        # Pass Base64 directly to API.
        raw_result = await self.request_with_file_content(
            file_base64, file_name, task_type, custom_prompt, output_format, model_name
        )

        all_bbox_images = []
        if task_type == "doc2json":
            processed = postprocess_doc2json_result(
                raw_result, img_paths[0], output_format="md"
            )
            for i, img_path in enumerate(img_paths):
                im = draw_bboxes_on_image(img_path, raw_result, page_index=i + 1)
                if im is not None:
                    saved_path = session_dir / f"bbox_page_{i+1}.png"
                    im.save(str(saved_path), "PNG")
                    all_bbox_images.append((str(saved_path), f"Page {i+1}"))
        elif task_type == "doc2md":
            processed = postprocess_doc2md_result(raw_result)
        else:
            processed = raw_result

        print(f"[parse] done, task_type={task_type}")
        yield processed, all_bbox_images, processed, raw_result

    # ==================== static helper methods ====================

    @staticmethod
    def check_task_input(task_type, custom_prompt):
        if task_type == "custom" and (not custom_prompt or custom_prompt.strip() == ""):
            raise gr.Error("Please enter a custom prompt before parsing.")
        return task_type

    def _load_example(self, file_path):
        """Replace the old to_file: read a server-side example file and wrap it into State."""
        file_path = Path(file_path)
        file_name = file_path.name

        # Read file as Base64.
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")

        # Generate preview images.
        session_id = uuid.uuid4().hex
        session_dir = Path(tempfile.gettempdir()) / f"infinity_preview_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        img_b64_list = []
        if file_path.suffix.lower() == ".pdf":
            MAX_PAGES = 10
            pages = convert_pdf_to_images(file_path, dpi=300)
            pages = pages[:MAX_PAGES]
            for idx, page in enumerate(pages, start=1):
                img_path = session_dir / f"preview_page_{idx}.png"
                page.save(img_path, "PNG")
                img_b64_list.append(self.encode_img_base64(str(img_path)))
        else:
            img_b64_list = [self.encode_img_base64(str(file_path))]

        # Return: file_state (Base64, filename), preview list, current page index, rendered HTML.
        return (
            (file_base64, file_name),
            img_b64_list,
            0,
            self.render_img_base64(img_b64_list, 0, 1),
        )

    @staticmethod
    def encode_img_base64(img_path, min_pixels=None, max_pixels=None):
        base64_str, mime_type = encode_image_to_base64(img_path, min_pixels, max_pixels)
        return f"data:{mime_type};base64,{base64_str}"

    @staticmethod
    def get_pdf_thumbnail(pdf_path, dpi=150):
        """Generate a thumbnail (first page) from a PDF for use as a Gradio Image component."""
        pages = convert_pdf_to_images(pdf_path, dpi=dpi)
        if not pages:
            return None
        thumb = pages[0]
        thumb_id = uuid.uuid4().hex[:8]
        thumb_dir = Path(tempfile.gettempdir()) / "infinity_thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumb_dir / f"thumb_{thumb_id}.png"
        thumb.save(thumb_path, "PNG")
        return str(thumb_path)

    @staticmethod
    def render_img_base64(img_b64_list, idx, scale):
        if not img_b64_list:
            return "<p style='color:gray'>Please upload an image first.</p>"
        idx %= len(img_b64_list)
        src = img_b64_list[idx]
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
    def check_file_state(file_state):
        if file_state is None:
            raise gr.Error("Please upload a PDF or image before parsing.")
        return file_state

    @staticmethod
    def on_task_change(task_type):
        return gr.update(visible=(task_type == "custom"))

    @staticmethod
    def reset_zoom():
        return gr.update(value=1)

    @staticmethod
    def hide_download_file():
        return gr.update(value=None, visible=False)

    # ==================== callback methods ====================

    async def upload_handler(self, files):
        """Convert user-uploaded file to Base64 State."""
        if files is None:
            return None, [], 0, ""

        if hasattr(files, "path"):
            file_path = files.path
        elif isinstance(files, list) and len(files) > 0:
            first = files[0]
            file_path = first.path if hasattr(first, "path") else str(first)
        else:
            file_path = str(files)

        file_name = Path(file_path).name
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")

        session_id = uuid.uuid4().hex
        session_dir = Path(tempfile.gettempdir()) / f"infinity_preview_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        img_b64_list = []
        if file_path.lower().endswith(".pdf"):
            MAX_PAGES = 10
            pages = convert_pdf_to_images(file_path, dpi=300)
            pages = pages[:MAX_PAGES]

            for idx, page in enumerate(pages, start=1):
                img_path = session_dir / f"preview_page_{idx}.png"
                page.save(img_path, "PNG")
                img_b64_list.append(self.encode_img_base64(str(img_path)))
        else:
            img_b64_list = [self.encode_img_base64(file_path)]

        # Return state tuple.
        file_state = (file_base64, file_name)
        return file_state, img_b64_list, 0, self.render_img_base64(img_b64_list, 0, 1)

    def show_prev(self, img_b64_list, idx, scale):
        idx -= 1
        return idx, self.render_img_base64(img_b64_list, idx, scale)

    def show_next(self, img_b64_list, idx, scale):
        idx += 1
        return idx, self.render_img_base64(img_b64_list, idx, scale)

    def on_zoom_change(self, img_b64_list, idx, scale):
        return self.render_img_base64(img_b64_list, idx, scale)

    @staticmethod
    def package_zip(task_type, processed_text, raw_text, bbox_img):
        zip_path = package_results_as_zip(task_type, processed_text, raw_text, bbox_img)
        return gr.update(value=zip_path, visible=True)

    # ==================== UI components ====================

    def _build_left_column(self):
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
            prev_btn = gr.Button(" Pre")
            next_btn = gr.Button("Next ")

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

    def _build_right_column(self, demo_data_root):
        model_selector = gr.Dropdown(
            choices=self.available_models,
            value=self.available_models[0],
            label="Model Selection",
            info="Select the model to use for parsing",
            interactive=True,
        )

        # Core: State for carrying file content in memory.
        file_state = gr.State(None)

        with gr.Accordion("Examples", open=True):
            # PDF thumbnail: use the first page as PNG.
            pdf_thumb = self.get_pdf_thumbnail(
                os.path.join(demo_data_root, "Academic_Papers.pdf")
            )
            demo_thumbs = [
                os.path.join(demo_data_root, "Financial_Reports.png"),
                os.path.join(demo_data_root, "Books.png"),
                os.path.join(demo_data_root, "Magazines.png"),
                pdf_thumb,
            ]
            demo_paths = [
                os.path.join(demo_data_root, "Financial_Reports.png"),
                os.path.join(demo_data_root, "Books.png"),
                os.path.join(demo_data_root, "Magazines.png"),
                os.path.join(demo_data_root, "Academic_Papers.pdf"),
            ]
            labels = [
                "Financial Reports (IMG)",
                "Books (IMG)",
                "Magazines (IMG)",
                "Academic Papers (PDF)",
            ]

            with gr.Row():
                for i, label in enumerate(labels):
                    with gr.Column(scale=1, min_width=120):
                        gr.Image(
                            value=demo_thumbs[i], width=120, height=90, show_label=False
                        )
                        gr.Button(label)

        download_btn = gr.Button(" Generate download link", size="sm")
        output_file = gr.File(
            label="Parse result",
            interactive=False,
            elem_id="down-file-box",
            visible=False,
        )

        gr.HTML("<style>#down-file-box { max-height: 80px; }</style>")

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
            file_state,
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
        file_state,
        demo_data_root,
    ):
        img_list_state = gr.State([])
        idx_state = gr.State(0)

        # ================= Bind example button events =================
        demo_paths = [
            os.path.join(demo_data_root, "Financial_Reports.png"),
            os.path.join(demo_data_root, "Books.png"),
            os.path.join(demo_data_root, "Magazines.png"),
            os.path.join(demo_data_root, "Academic_Papers.pdf"),
        ]

        # Safely locate demo buttons and bind events.
        for block in list(self.demo.blocks.values()):
            if isinstance(block, gr.Button):
                val = getattr(block, "value", "")
                labels = [
                    "Financial Reports (IMG)",
                    "Books (IMG)",
                    "Magazines (IMG)",
                    "Academic Papers (PDF)",
                ]
                if val in labels:
                    idx = labels.index(val)
                    block.click(
                        fn=self._load_example,
                        inputs=gr.State(demo_paths[idx]),
                        outputs=[file_state, img_list_state, idx_state, viewer],
                    )

        # ================= Remaining event bindings =================
        file.change(
            self.upload_handler,
            inputs=file,
            outputs=[file_state, img_list_state, idx_state, viewer],
        ).then(
            self.reset_zoom,
            inputs=None,
            outputs=zoom,
        )

        task_selector.change(
            self.on_task_change, inputs=task_selector, outputs=custom_prompt
        )
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
        zoom.change(
            self.on_zoom_change,
            inputs=[img_list_state, idx_state, zoom],
            outputs=viewer,
        )

        change_bu.click(
            fn=self.check_task_input,
            inputs=[task_selector, custom_prompt],
            outputs=task_selector,
        ).then(
            fn=self.check_file_state,
            inputs=file_state,
            outputs=file_state,
        ).then(
            self.hide_download_file,
            inputs=None,
            outputs=output_file,
        ).then(
            fn=self.infinity_parser2,
            inputs=[file_state, task_selector, custom_prompt, model_selector],
            outputs=[md, bbox_img, processed_text, raw_text],
        )

        download_btn.click(
            fn=self.package_zip,
            inputs=[task_selector, processed_text, raw_text, bbox_img],
            outputs=output_file,
        )

        clear_bu.add([file, md, bbox_img, processed_text, raw_text])

    def _build_ui(self):
        demo_data_root = os.path.join(os.path.dirname(__file__), "..", "demo_data")

        with gr.Blocks() as self.demo:
            with gr.Row():
                with gr.Column(variant="panel", scale=5):
                    left_components = self._build_left_column()

                with gr.Column(variant="panel", scale=5):
                    right_components = self._build_right_column(demo_data_root)

            self._bind_events(*left_components, *right_components, demo_data_root)

        return self.demo

    def run(self, server_name="0.0.0.0", share=True):
        self.demo = self._build_ui()
        self.demo.launch(server_name=server_name, share=share)


def main():
    app = GradioApp()
    app.run()


if __name__ == "__main__":
    main()
