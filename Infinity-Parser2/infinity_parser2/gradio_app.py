import uuid
import tempfile
import base64
import io
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

    async def _upload_file(self, file_path: str, file_name: str) -> tuple[str, str]:
        """
        POST a file to /upload endpoint.
        Returns (upload_id, file_name).
        Raises on failure.
        """
        mime_type = self._guess_mime(file_name)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        response = await self._http_client.post(
            f"{self.openai_api_base}/upload",
            files={"file": (file_name, io.BytesIO(file_bytes), mime_type)},
            headers={"Authorization": f"Bearer {self.Authorization}"},
            timeout=120.0,
        )
        if response.status_code != 200:
            raise Exception(f"Upload failed {response.status_code}: {response.text}")
        data = response.json()
        return data["upload_id"], data["file_name"]

    @staticmethod
    def _guess_mime(file_name: str) -> str:
        ext = Path(file_name).suffix.lower()
        return {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
            ext, "application/octet-stream"
        )

    async def request_with_file_content(
        self,
        upload_id,
        file_name,
        task_type,
        custom_prompt,
        output_format,
        model_name,
    ):
        """Use upload_id to request the Flask API."""
        url = f"{self.openai_api_base}/v1/chat/completions"

        payload = {
            "upload_id": upload_id,
            "file_name": file_name,
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
        """
        Parse using upload_id: call API directly, decode to temp file only
        when PDF-to-image or bbox drawing is needed.
        """
        import time
        t0 = time.perf_counter()

        if not file_state:
            raise gr.Error("File state lost, please re-upload.")

        upload_id = file_state.get("upload_id") if isinstance(file_state, dict) else file_state[0]
        file_name = file_state.get("file_name") if isinstance(file_state, dict) else file_state[1]
        file_path = file_state.get("file_path") if isinstance(file_state, dict) else None
        output_format = "json" if task_type == "doc2json" else "md"

        session_id = uuid.uuid4().hex
        session_dir = Path(tempfile.gettempdir()) / f"infinity_parse_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        # 1. Resolve local file path: use cached file_path if available,
        #    otherwise decode from base64 in file_state.
        if file_path and os.path.exists(file_path):
            local_path = file_path
        else:
            file_base64 = file_state.get("file_base64") if isinstance(file_state, dict) else None
            if file_base64:
                file_bytes = base64.b64decode(file_base64)
                local_path = session_dir / file_name
                with open(local_path, "wb") as f:
                    f.write(file_bytes)
            else:
                raise gr.Error("Local file not available for preview rendering.")

        # 2. Build img_paths (for bbox drawing).
        ext = Path(local_path).suffix.lower()
        img_paths = []
        if ext == ".pdf":
            MAX_PAGES = 10
            pages = convert_pdf_to_images(local_path, dpi=300)
            pages = pages[:MAX_PAGES]
            for idx, page in enumerate(pages, start=1):
                img_path = session_dir / f"parse_page_{idx}.png"
                page.save(img_path, "PNG")
                img_paths.append(str(img_path))
        else:
            img_paths = [str(local_path)]
        t_decode_done = time.perf_counter()

        # 3. Call API with upload_id.
        raw_result = await self.request_with_file_content(
            upload_id, file_name, task_type, custom_prompt, output_format, model_name
        )
        t_api_done = time.perf_counter()

        # 4. Postprocess.
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
        t_postprocess_done = time.perf_counter()

        t_decode_ms = (t_decode_done - t0) * 1000
        t_api_ms = (t_api_done - t_decode_done) * 1000
        t_postprocess_ms = (t_postprocess_done - t_api_done) * 1000
        t_total_ms = (t_postprocess_done - t0) * 1000
        print(
            f"[infinity_parser2] file={file_name} task={task_type} model={model_name}  "
            f"decode={t_decode_ms:.1f}ms  api={t_api_ms:.1f}ms  "
            f"postprocess={t_postprocess_ms:.1f}ms  total={t_total_ms:.1f}ms"
        )
        yield processed, all_bbox_images, processed, raw_result

    # ==================== static helper methods ====================

    @staticmethod
    def check_task_input(task_type, custom_prompt):
        if task_type == "custom" and (not custom_prompt or custom_prompt.strip() == ""):
            raise gr.Error("Please enter a custom prompt before parsing.")
        return task_type

    async def _load_example(self, file_path):
        """POST example file to /upload, then generate preview images."""
        import time
        t0 = time.perf_counter()

        file_path = Path(file_path)
        file_name = file_path.name

        # 1. Upload → get upload_id.
        t_upload = time.perf_counter()
        upload_id, returned_name = await self._upload_file(str(file_path), file_name)
        t_upload_done = time.perf_counter()

        # 2. Read file bytes (for preview generation).
        t_read = time.perf_counter()
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")
        t_read_done = time.perf_counter()

        # 3. Generate preview images.
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
        t_preview_done = time.perf_counter()

        # 4. Render HTML.
        viewer_html = self.render_img_base64(img_b64_list, 0, 1)
        t_render_done = time.perf_counter()

        t_upload_ms = (t_upload_done - t_upload) * 1000
        t_read_ms = (t_read_done - t_read) * 1000
        t_preview_ms = (t_preview_done - t_read_done) * 1000
        t_render_ms = (t_render_done - t_preview_done) * 1000
        t_total_ms = (t_render_done - t0) * 1000
        print(
            f"[_load_example] file={file_name}  "
            f"upload={t_upload_ms:.1f}ms  read={t_read_ms:.1f}ms  "
            f"preview={t_preview_ms:.1f}ms  render={t_render_ms:.1f}ms  "
            f"total={t_total_ms:.1f}ms"
        )

        file_state = {"upload_id": upload_id, "file_name": file_name,
                      "file_path": str(file_path), "file_base64": file_base64}
        return (
            file_state,
            img_b64_list,
            0,
            viewer_html,
            str(file_path),
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
        if isinstance(file_state, dict):
            if not file_state.get("upload_id"):
                raise gr.Error("File upload incomplete, please re-upload.")
        elif isinstance(file_state, (list, tuple)):
            if not file_state[0]:
                raise gr.Error("Please upload a PDF or image before parsing.")
        return file_state

    @staticmethod
    def on_task_change(task_type):
        return gr.update(visible=(task_type == "custom")), gr.update(visible=(task_type != "doc2md"))

    @staticmethod
    def reset_zoom():
        return gr.update(value=1)

    @staticmethod
    def hide_download_file():
        return gr.update(value=None, visible=False)

    # ==================== callback methods ====================

    async def upload_handler(self, files):
        """
        Upload file via /upload endpoint to get upload_id.
        Still generates preview images locally (no change needed).
        file_state is now a dict: {upload_id, file_name, file_path, file_base64}
        """
        import time
        t0 = time.perf_counter()

        if files is None:
            return None, [], 0, ""

        if hasattr(files, "path"):
            file_path = files.path
            # Prefer orig_name (original filename) over the temp path name.
            orig_name = getattr(files, "orig_name", None) or Path(file_path).name
        elif isinstance(files, list) and len(files) > 0:
            first = files[0]
            file_path = first.path if hasattr(first, "path") else str(first)
            orig_name = getattr(first, "orig_name", None) or Path(file_path).name
        else:
            file_path = str(files)
            orig_name = Path(file_path).name

        # 1. POST to /upload → get upload_id.
        t_upload = time.perf_counter()
        upload_id, returned_name = await self._upload_file(file_path, orig_name)
        t_upload_done = time.perf_counter()

        # 2. Read file bytes (for preview generation only, no base64 to API).
        t_read = time.perf_counter()
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")
        t_read_done = time.perf_counter()

        # 3. Generate preview images.
        session_id = uuid.uuid4().hex
        session_dir = Path(tempfile.gettempdir()) / f"infinity_preview_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        img_b64_list = []
        if orig_name.lower().endswith(".pdf"):
            MAX_PAGES = 10
            pages = convert_pdf_to_images(file_path, dpi=300)
            pages = pages[:MAX_PAGES]
            for idx, page in enumerate(pages, start=1):
                img_path = session_dir / f"preview_page_{idx}.png"
                page.save(img_path, "PNG")
                img_b64_list.append(self.encode_img_base64(str(img_path)))
        else:
            img_b64_list = [self.encode_img_base64(file_path)]
        t_preview_done = time.perf_counter()

        # 4. Render HTML.
        viewer_html = self.render_img_base64(img_b64_list, 0, 1)
        t_render_done = time.perf_counter()

        t_upload_ms = (t_upload_done - t_upload) * 1000
        t_read_ms = (t_read_done - t_read) * 1000
        t_preview_ms = (t_preview_done - t_read_done) * 1000
        t_render_ms = (t_render_done - t_preview_done) * 1000
        t_total_ms = (t_render_done - t0) * 1000
        print(
            f"[upload_handler] file={orig_name} upload={t_upload_ms:.1f}ms  "
            f"read={t_read_ms:.1f}ms  preview={t_preview_ms:.1f}ms  "
            f"render={t_render_ms:.1f}ms  total={t_total_ms:.1f}ms"
        )

        file_state = {"upload_id": upload_id, "file_name": orig_name,
                      "file_path": file_path, "file_base64": file_base64}
        return file_state, img_b64_list, 0, viewer_html

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
            with gr.Tab("Layout result") as layout_tab:
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
            layout_tab,
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
        layout_tab,
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
                        outputs=[file_state, img_list_state, idx_state, viewer, file],
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
            self.on_task_change, inputs=task_selector, outputs=[custom_prompt, layout_tab]
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
        self.demo_data_root = os.path.join(os.path.dirname(__file__), "..", "demo_data")

        with gr.Blocks() as self.demo:
            with gr.Row():
                with gr.Column(variant="panel", scale=5):
                    left_components = self._build_left_column()

                with gr.Column(variant="panel", scale=5):
                    right_components = self._build_right_column(self.demo_data_root)

            self._bind_events(*left_components, *right_components, self.demo_data_root)

        return self.demo

    def run(self, server_name="0.0.0.0", share=True):
        self.demo = self._build_ui()
        self.demo.launch(server_name=server_name, share=share, allowed_paths=[self.demo_data_root])


def main():
    app = GradioApp()
    app.run()


if __name__ == "__main__":
    main()
