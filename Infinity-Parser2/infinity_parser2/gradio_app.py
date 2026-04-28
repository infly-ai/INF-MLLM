import uuid
import threading
import tempfile
import base64
import io
from pathlib import Path
import httpx
import os
import gradio as gr
from PyPDF2 import PdfReader, PdfWriter
from utils import (
    convert_pdf_to_images,
    postprocess_doc2json_result,
    postprocess_doc2md_result,
    draw_bboxes_on_image,
    package_results_as_zip
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
        # Model configuration: model_name -> {api_base, auth}
        # Reads from environment variables first, falls back to defaults (modify for deployment)
        self.model_configs = {
            "Infinity-Parser2-Pro": {
                "api_base": os.environ.get(
                    "INFINITY_API_BASE_PRO", "http://localhost:8000"
                ),
                "auth": os.environ.get("INFINITY_API_AUTH_PRO", ""),
            },
            "Infinity-Parser2-Flash": {
                "api_base": os.environ.get(
                    "INFINITY_API_BASE_FLASH", "http://localhost:8002"
                ),
                "auth": os.environ.get("INFINITY_API_AUTH_FLASH", ""),
            },
        }
        self.available_models = list(self.model_configs.keys())
        self._http_client = httpx.AsyncClient(verify=False, timeout=600.0)
        self.demo = None
        self._skip_file_change = False  # guard to prevent double-upload from example clicks

    def _init_models(self):
        return self.available_models

    # ==================== core methods ====================

    async def _upload_file(
        self, file_path: str, file_name: str, model_name: str
    ) -> tuple[str, str]:
        """
        POST a file to /upload endpoint of the specified model's backend.
        Returns (upload_id, file_name).
        """
        config = self.model_configs.get(model_name)
        if not config:
            raise Exception(f"Unknown model: {model_name}")
        base_url = config["api_base"]
        auth = config["auth"]
        mime_type = self._guess_mime(file_name)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        response = await self._http_client.post(
            f"{base_url}/upload",
            files={"file": (file_name, io.BytesIO(file_bytes), mime_type)},
            headers={
                "Authorization": (
                    auth if auth.startswith("Bearer ") else f"Bearer {auth}"
                )
            },
            timeout=120.0,
        )
        if response.status_code != 200:
            raise Exception(f"Upload failed {response.status_code}: {response.text}")
        data = response.json()
        return data["upload_id"], data["file_name"]

    @staticmethod
    def _guess_mime(file_name: str) -> str:
        ext = Path(file_name).suffix.lower()
        return {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(ext, "application/octet-stream")

    @staticmethod
    def _split_pdf(file_path: str, first_n: int) -> tuple[str, str]:
        """
        Split a PDF into two parts: first_n pages and the rest.
        Returns (preview_path, remaining_path) — both saved to temp files.
        """

        reader = PdfReader(file_path)
        total = len(reader.pages)
        if total <= first_n:
            return file_path, None  # No split needed

        preview_writer = PdfWriter()
        for i in range(first_n):
            preview_writer.add_page(reader.pages[i])

        session_id = uuid.uuid4().hex
        session_dir = Path(tempfile.gettempdir()) / f"infinity_split_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        base_stem = Path(file_path).stem
        preview_path = str(session_dir / f"{base_stem}_preview.pdf")
        with open(preview_path, "wb") as f:
            preview_writer.write(f)

        remaining_writer = PdfWriter()
        for i in range(first_n, total):
            remaining_writer.add_page(reader.pages[i])
        remaining_path = str(session_dir / f"{base_stem}_remaining.pdf")
        with open(remaining_path, "wb") as f:
            remaining_writer.write(f)

        return preview_path, remaining_path

    @staticmethod
    def _get_pdf_page_count(file_path: str) -> int:
        """Return the total page count of a PDF file."""
        from PyPDF2 import PdfReader

        return len(PdfReader(file_path).pages)

    def _sync_upload_remaining(
        self,
        remaining_path: str,
        file_name: str,
        upload_id: str,
        model_name: str,
    ) -> bool:
        """
        Upload the remaining pages to the server (SYNCHRONOUS).
        Designed to run in a background thread — no event loop needed.
        Returns True on success, False on failure.
        """
        import time as _time

        config = self.model_configs.get(model_name)
        if not config:
            return False
        base_url = config["api_base"]
        auth = config["auth"]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(remaining_path, "rb") as f:
                    file_bytes = f.read()

                headers = {}
                if auth:
                    headers["Authorization"] = (
                        auth if auth.startswith("Bearer ") else f"Bearer {auth}"
                    )

                with httpx.Client(verify=False, timeout=300.0) as client:
                    response = client.post(
                        f"{base_url}/upload",
                        files={
                            "file": (file_name, io.BytesIO(file_bytes), "application/pdf")
                        },
                        data={"append_to": upload_id},
                        headers=headers,
                    )
                if response.status_code == 200:
                    return True
            except Exception:
                pass

            if attempt < max_retries - 1:
                _time.sleep(2 ** attempt)

        return False

    async def request_with_file_content(
        self,
        upload_id,
        file_name,
        task_type,
        custom_prompt,
        output_format,
        model_name,
        max_pages=10,
    ):
        """Use upload_id to request the Flask API of the specified model."""
        config = self.model_configs.get(model_name)
        if not config:
            raise Exception(f"Unknown model: {model_name}")
        base_url = config["api_base"]
        auth = config["auth"]
        url = f"{base_url}/v1/chat/completions"

        payload = {
            "upload_id": upload_id,
            "file_name": file_name,
            "task_type": task_type,
            "output_format": output_format,
            "model": model_name,
            "max_pages": int(max_pages),
        }
        if custom_prompt:
            payload["custom_prompt"] = custom_prompt

        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = (
                auth if auth.startswith("Bearer ") else f"Bearer {auth}"
            )

        response = await self._http_client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content

    async def infinity_parser2(
        self, file_state, task_type, custom_prompt, model_name, max_pages=10
    ):
        """
        Parse using upload_id: call API directly, decode to temp file only
        when PDF-to-image or bbox drawing is needed.

        For large PDFs that were split on upload, the remaining pages are
        uploaded in the background before parsing begins.
        """

        if not file_state:
            raise gr.Error("File state lost, please re-upload.")

        upload_id = (
            file_state.get("upload_id")
            if isinstance(file_state, dict)
            else file_state[0]
        )
        file_name = (
            file_state.get("file_name")
            if isinstance(file_state, dict)
            else file_state[1]
        )
        file_path = (
            file_state.get("file_path") if isinstance(file_state, dict) else None
        )
        remaining_path = (
            file_state.get("remaining_path") if isinstance(file_state, dict) else None
        )
        output_format = "json" if task_type == "doc2json" else "md"

        session_id = uuid.uuid4().hex
        session_dir = Path(tempfile.gettempdir()) / f"infinity_parse_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        # 1. Resolve local file path: use cached file_path if available,
        #    otherwise decode from base64 in file_state.
        if file_path and os.path.exists(file_path):
            local_path = file_path
        else:
            file_base64 = (
                file_state.get("file_base64") if isinstance(file_state, dict) else None
            )
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
        max_pages = int(max_pages)
        if ext == ".pdf":
            pages = convert_pdf_to_images(local_path, dpi=300)
            pages = pages[:max_pages]
            for idx, page in enumerate(pages, start=1):
                img_path = session_dir / f"parse_page_{idx}.png"
                page.save(img_path, "PNG")
                img_paths.append(str(img_path))
        else:
            img_paths = [str(local_path)]

        # Call API with upload_id (server has only the preview pages, no truncation needed).
        raw_result = await self.request_with_file_content(
            upload_id,
            file_name,
            task_type,
            custom_prompt,
            output_format,
            model_name,
            max_pages=max_pages,
        )

        # Upload remaining pages in background thread (sync, no event loop needed).
        if remaining_path:
            _rp, _fn, _uid, _mn = remaining_path, file_name, upload_id, model_name
            threading.Thread(
                target=self._sync_upload_remaining,
                args=(_rp, _fn, _uid, _mn),
                daemon=True,
            ).start()

        # Postprocess.
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

        yield processed, all_bbox_images, processed, raw_result

    # ==================== static helper methods ====================

    @staticmethod
    def check_task_input(task_type, custom_prompt):
        if task_type == "custom" and (not custom_prompt or custom_prompt.strip() == ""):
            raise gr.Error("Please enter a custom prompt before parsing.")
        return task_type

    async def _load_example(self, file_path, model_name, max_pages=10):
        """POST example file to /upload of the selected model, then generate preview images.

        Delegates to upload_handler() to avoid duplicating upload/preview logic.
        """
        file_path = str(file_path)

        # Build a lightweight namespace that mimics a Gradio UploadFile object.
        class _FakeFile:
            def __init__(self, path):
                self.path = path
                self.orig_name = Path(path).name

        fake_file = _FakeFile(file_path)

        # Reuse upload_handler — returns (file_state, img_b64_list, idx, viewer_html, pdf_pages_update)
        file_state, img_b64_list, idx, viewer_html, pdf_pages_update = (
            await self.upload_handler(fake_file, model_name, max_pages)
        )

        # Patch file_path back to the original example file (upload_handler may
        # store preview_path for large PDFs, but _load_example callers need the
        # full original path for later processing).
        if file_state is not None:
            file_state["file_path"] = file_path

        # Set guard so the subsequent file.change won't trigger upload_handler again.
        self._skip_file_change = True

        return (
            file_state,
            img_b64_list,
            idx,
            viewer_html,
            file_path,
            pdf_pages_update,
        )

    @staticmethod
    def _file_to_base64(file_path: str) -> str:
        """Read a file and return its base64-encoded content as a string."""
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        return base64.b64encode(file_bytes).decode("utf-8")

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
    def render_img_base64(img_b64_list, idx):
        if not img_b64_list:
            return "<p style='color:gray'>Please upload a PDF or image first.</p>"
        idx %= len(img_b64_list)
        src = img_b64_list[idx]
        return f"""
            <div style="width:100%;height:800px;overflow:auto;border:1px solid #ccc;">
              <div style="min-width:100%;display:flex;justify-content:center;">
                <img src="{src}" style="width:100%;height:auto;display:block;">
              </div>
            </div>
            """

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
        return gr.update(visible=(task_type == "custom")), gr.update(
            visible=(task_type != "doc2md")
        )

    @staticmethod
    def hide_download_file():
        return gr.update(value=None, visible=False)

    # ==================== callback methods ====================

    async def upload_handler(self, files, model_name, max_pages=10):
        """
        Upload file to the selected model's backend via /upload endpoint.

        For large PDFs (> max_pages pages):
          1. Split into preview (first max_pages) + remaining
          2. Upload preview immediately
          3. Upload remaining in background (async, non-blocking)
        """

        # Guard: skip if this was triggered by _load_example updating gr.File.
        if self._skip_file_change:
            self._skip_file_change = False
            return (
                gr.update(),   # file_state — keep unchanged
                gr.update(),   # img_list_state
                gr.update(),   # idx_state
                gr.update(),   # viewer
                gr.update(),   # pdf_pages
            )

        if files is None:
            return None, [], 0, "", gr.update(visible=False)

        if hasattr(files, "path"):
            file_path = files.path
            orig_name = getattr(files, "orig_name", None) or Path(file_path).name
        elif isinstance(files, list) and len(files) > 0:
            first = files[0]
            file_path = first.path if hasattr(first, "path") else str(first)
            orig_name = getattr(first, "orig_name", None) or Path(file_path).name
        else:
            file_path = str(files)
            orig_name = Path(file_path).name

        max_pages = int(max_pages)
        is_pdf = orig_name.lower().endswith(".pdf")

        # For large PDFs, split before uploading
        preview_path = file_path
        remaining_path = None
        if is_pdf:
            try:
                total_pages = self._get_pdf_page_count(file_path)
                SPLIT_THRESHOLD = max_pages
                if total_pages > SPLIT_THRESHOLD:
                    preview_path, remaining_path = self._split_pdf(
                        file_path, SPLIT_THRESHOLD
                    )
            except Exception as exc:
                preview_path = file_path
                remaining_path = None

        # POST preview (or whole file for non-PDF/small PDF) to /upload -> get upload_id.
        upload_id, returned_name = await self._upload_file(
            preview_path, orig_name, model_name
        )

        # Read file bytes for preview generation.
        file_base64 = self._file_to_base64(preview_path)

        # Generate preview images — always fixed at max_pages pages.
        session_id = uuid.uuid4().hex
        session_dir = Path(tempfile.gettempdir()) / f"infinity_preview_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        img_b64_list = []
        if is_pdf:
            pages = convert_pdf_to_images(preview_path, dpi=100)
            for page in pages:
                buf = io.BytesIO()
                page.save(buf, format="PNG")
                page_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                img_b64_list.append(f"data:image/png;base64,{page_b64}")
        else:
            mime = self._guess_mime(orig_name)
            img_b64_list.append(f"data:{mime};base64,{file_base64}")

        # Render HTML.
        viewer_html = self.render_img_base64(img_b64_list, 0)

        file_state = {
            "upload_id": upload_id,
            "file_name": orig_name,
            "file_path": preview_path,
            "file_base64": file_base64,
            "remaining_path": remaining_path,
        }
        pdf_pages_update = gr.update(visible=is_pdf)
        return file_state, img_b64_list, 0, viewer_html, pdf_pages_update

    def show_prev(self, img_b64_list, idx):
        idx -= 1
        return idx, self.render_img_base64(img_b64_list, idx)

    def show_next(self, img_b64_list, idx):
        idx += 1
        return idx, self.render_img_base64(img_b64_list, idx)

    @staticmethod
    def package_zip(task_type, processed_text, raw_text, bbox_img):
        zip_path = package_results_as_zip(task_type, processed_text, raw_text, bbox_img)
        return gr.update(value=zip_path, visible=True)

    def on_model_change(self, model_name, file_state):
        """When user switches model, clear result display if file_state exists, but do not delete file_state."""
        if file_state and file_state.get("upload_id"):
            gr.Info(
                f"Model switched to {model_name}, click 'Parse' to re-parse (no need to re-upload)."
            )
            # Return updates: keep file_state unchanged via gr.update(), clear result display
            return (
                gr.update(),  # file_state unchanged
                gr.update(value=""),  # md cleared
                gr.update(value=[]),  # bbox_img cleared
                gr.update(value=""),  # processed_text cleared
                gr.update(value=""),  # raw_text cleared
            )
        else:
            gr.Warning("Please upload a file before switching models.")
            return (
                gr.update(value=None),  # file_state
                gr.update(value=""),
                gr.update(value=[]),
                gr.update(value=""),
                gr.update(value=""),
            )

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

        pdf_pages = gr.Slider(
            1,
            10,
            value=10,
            step=1,
            label="PDF Pages",
            info="Number of PDF pages to parse (1–10)",
            visible=False,
        )

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
            pdf_pages,
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
        pdf_pages,
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
                        inputs=[gr.State(demo_paths[idx]), model_selector, pdf_pages],
                        outputs=[
                            file_state,
                            img_list_state,
                            idx_state,
                            viewer,
                            file,
                            pdf_pages,
                        ],
                    )

        # ================= Remaining event bindings =================
        file.change(
            self.upload_handler,
            inputs=[file, model_selector, pdf_pages],
            outputs=[file_state, img_list_state, idx_state, viewer, pdf_pages],
        )

        # Slider only controls parse page count, no preview re-generation.
        task_selector.change(
            self.on_task_change,
            inputs=task_selector,
            outputs=[custom_prompt, layout_tab],
        )
        prev_btn.click(
            self.show_prev,
            inputs=[img_list_state, idx_state],
            outputs=[idx_state, viewer],
        )
        next_btn.click(
            self.show_next,
            inputs=[img_list_state, idx_state],
            outputs=[idx_state, viewer],
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
            inputs=[
                file_state,
                task_selector,
                custom_prompt,
                model_selector,
                pdf_pages,
            ],
            outputs=[md, bbox_img, processed_text, raw_text],
        )

        download_btn.click(
            fn=self.package_zip,
            inputs=[task_selector, processed_text, raw_text, bbox_img],
            outputs=output_file,
        )

        clear_bu.add([file, md, bbox_img, processed_text, raw_text])

        # Clear result display when model is switched, but keep file_state
        model_selector.change(
            self.on_model_change,
            inputs=[model_selector, file_state],
            outputs=[file_state, md, bbox_img, processed_text, raw_text],
        )

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
        self.demo.launch(
            server_name=server_name, share=share, allowed_paths=[self.demo_data_root]
        )


def main():
    app = GradioApp()
    app.run()


if __name__ == "__main__":
    main()
