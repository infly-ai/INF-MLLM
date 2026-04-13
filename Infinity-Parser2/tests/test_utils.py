"""Unit tests for InfinityParser2 utility functions."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from infinity_parser2.utils import (
    convert_pdf_to_images,
    encode_file_to_base64,
    extract_json_content,
    load_image,
    truncate_last_incomplete_element,
    obtain_origin_hw,
    restore_abs_bbox_coordinates,
    convert_json_to_markdown,
    postprocess_doc2json_result,
    save_results,
)
from infinity_parser2.utils.model import ModelCache, _resolve_hf_endpoint


class TestLoadImage(unittest.TestCase):
    """Tests for load_image utility function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test.png")
        self.test_image = Image.new("RGB", (100, 100), color="red")
        self.test_image.save(self.test_image_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_image_from_path(self):
        """Test loading image from file path."""
        loaded = load_image(self.test_image_path)
        self.assertIsInstance(loaded, Image.Image)
        self.assertEqual(loaded.size, (100, 100))
        self.assertEqual(loaded.mode, "RGB")

    def test_load_image_from_pil_image(self):
        """Test loading image from PIL Image object."""
        original = Image.new("RGB", (50, 50), color="blue")
        loaded = load_image(original)
        self.assertIsInstance(loaded, Image.Image)
        self.assertEqual(loaded.size, (50, 50))
        self.assertEqual(loaded.mode, "RGB")

    def test_load_image_converts_to_rgb(self):
        """Test that loaded image is always in RGB mode."""
        rgba_image = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        loaded = load_image(rgba_image)
        self.assertEqual(loaded.mode, "RGB")

    def test_load_image_unsupported_type_raises_error(self):
        """Test that unsupported input type raises TypeError."""
        with self.assertRaises(TypeError):
            load_image(12345)
        with self.assertRaises(TypeError):
            load_image([1, 2, 3])


class TestEncodeFileToBase64(unittest.TestCase):
    """Tests for encode_file_to_base64 utility function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_encode_png_file(self):
        """Test encoding PNG file to base64."""
        png_path = os.path.join(self.temp_dir, "test.png")
        Image.new("RGB", (100, 100), color="green").save(png_path)
        base64_str, mime_type = encode_file_to_base64(png_path)
        self.assertIsInstance(base64_str, str)
        self.assertTrue(len(base64_str) > 0)
        self.assertEqual(mime_type, "image/png")

    def test_encode_jpg_file(self):
        """Test encoding JPG file to base64."""
        jpg_path = os.path.join(self.temp_dir, "test.jpg")
        Image.new("RGB", (100, 100), color="yellow").save(jpg_path, "JPEG")
        base64_str, mime_type = encode_file_to_base64(jpg_path)
        self.assertIsInstance(base64_str, str)
        self.assertTrue(len(base64_str) > 0)
        self.assertEqual(mime_type, "image/jpeg")

    def test_encode_jpeg_file(self):
        """Test encoding JPEG file with .jpeg extension."""
        jpeg_path = os.path.join(self.temp_dir, "test.jpeg")
        Image.new("RGB", (100, 100), color="orange").save(jpeg_path, "JPEG")
        base64_str, mime_type = encode_file_to_base64(jpeg_path)
        self.assertEqual(mime_type, "image/jpeg")

    def test_encode_webp_file(self):
        """Test encoding WebP file."""
        webp_path = os.path.join(self.temp_dir, "test.webp")
        Image.new("RGB", (100, 100), color="purple").save(webp_path, "WEBP")
        base64_str, mime_type = encode_file_to_base64(webp_path)
        self.assertEqual(mime_type, "image/webp")

    def test_encode_bmp_file(self):
        """Test encoding BMP file."""
        bmp_path = os.path.join(self.temp_dir, "test.bmp")
        Image.new("RGB", (100, 100), color="cyan").save(bmp_path, "BMP")
        base64_str, mime_type = encode_file_to_base64(bmp_path)
        self.assertEqual(mime_type, "image/bmp")

    def test_encode_pil_image(self):
        """Test encoding PIL Image object."""
        img = Image.new("RGB", (100, 100), color="magenta")
        base64_str, mime_type = encode_file_to_base64(img)
        self.assertIsInstance(base64_str, str)
        self.assertTrue(len(base64_str) > 0)
        self.assertEqual(mime_type, "image/jpeg")  # Default for PIL without format

    def test_encode_pil_image_with_format(self):
        """Test encoding PIL Image with explicit format."""
        img = Image.new("RGB", (100, 100), color="white")
        self.assertIsNone(img.format)
        _, mime_type = encode_file_to_base64(img)
        self.assertEqual(mime_type, "image/jpeg")

        png_path = os.path.join(self.temp_dir, "test.png")
        Image.new("RGB", (100, 100), color="white").save(png_path)
        with Image.open(png_path) as loaded_img:
            self.assertEqual(loaded_img.format, "PNG")
            _, mime_type = encode_file_to_base64(loaded_img)
            self.assertEqual(mime_type, "image/png")

    def test_encode_with_custom_min_max_pixels(self):
        """Test encoding with custom min_pixels and max_pixels parameters."""
        large_path = os.path.join(self.temp_dir, "large.png")
        Image.new("RGB", (1000, 1000), color="blue").save(large_path)
        base64_str, _ = encode_file_to_base64(large_path, min_pixels=100, max_pixels=50000)
        self.assertIsInstance(base64_str, str)
        self.assertTrue(len(base64_str) > 0)

    def test_encode_unknown_extension_defaults_to_jpeg(self):
        """Test that unknown extension defaults to image/jpeg."""
        unknown_path = os.path.join(self.temp_dir, "test.unknown")
        Image.new("RGB", (100, 100), color="gray").save(unknown_path, format="PNG")
        _, mime_type = encode_file_to_base64(unknown_path)
        self.assertEqual(mime_type, "image/jpeg")

    def test_base64_decoding(self):
        """Test that base64 output can be decoded correctly."""
        import base64
        png_path = os.path.join(self.temp_dir, "test.png")
        original = Image.new("RGB", (100, 100), color="red")
        original.save(png_path)
        base64_str, _ = encode_file_to_base64(png_path)
        decoded_bytes = base64.b64decode(base64_str)
        decoded_image = Image.open(io.BytesIO(decoded_bytes))
        self.assertIsInstance(decoded_image, Image.Image)


class TestConvertPdfToImages(unittest.TestCase):
    """Tests for convert_pdf_to_images utility function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_convert_pdf_returns_list(self):
        """Test that convert_pdf_to_images returns a list."""
        pdf_path = self._create_simple_pdf()
        result = convert_pdf_to_images(pdf_path)
        self.assertIsInstance(result, list)

    def test_convert_pdf_returns_pil_images(self):
        """Test that result contains PIL Image objects."""
        pdf_path = self._create_simple_pdf()
        result = convert_pdf_to_images(pdf_path)
        for img in result:
            self.assertIsInstance(img, Image.Image)

    def test_convert_pdf_returns_rgb_images(self):
        """Test that returned images are in RGB mode."""
        pdf_path = self._create_simple_pdf()
        result = convert_pdf_to_images(pdf_path)
        for img in result:
            self.assertEqual(img.mode, "RGB")

    def _create_simple_pdf(self):
        """Helper to create a simple PDF for testing."""
        try:
            import fitz
            pdf_path = os.path.join(self.temp_dir, "test.pdf")
            doc = fitz.open()
            page = doc.new_page(width=595, height=842)
            page.insert_text((100, 100), "Test PDF", fontsize=12)
            doc.save(pdf_path)
            doc.close()
            return pdf_path
        except ImportError:
            self.skipTest("PyMuPDF not available for creating test PDF")


class TestExtractJsonContent(unittest.TestCase):
    """Tests for extract_json_content utility function."""

    def test_extract_from_markdown_block(self):
        """Test extracting JSON from markdown code block."""
        text = '```json\n{"key": "value"}\n```'
        result = extract_json_content(text)
        self.assertEqual(result, '{"key": "value"}')

    def test_extract_partial_markdown_block(self):
        """Test extracting from partial markdown block."""
        text = '```json\n{"key": "value"'
        result = extract_json_content(text)
        self.assertEqual(result, '{"key": "value"')

    def test_return_plain_text_if_no_block(self):
        """Test returning text as-is if no markdown block."""
        text = '{"key": "value"}'
        result = extract_json_content(text)
        self.assertEqual(result, text)


class TestTruncateLastIncompleteElement(unittest.TestCase):
    """Tests for truncate_last_incomplete_element utility function."""

    def test_no_truncation_for_short_text(self):
        """Test that short text is not truncated."""
        text = '[{"bbox": [0,0,100,100], "text": "hello"}]'
        result, was_truncated = truncate_last_incomplete_element(text)
        self.assertEqual(result, text)
        self.assertFalse(was_truncated)

    def test_truncate_incomplete_element(self):
        """Test truncating incomplete element when text does not end with ]."""
        # Text must NOT end with "]" to trigger truncation. The first bbox dict is
        # complete, the second is truncated at the comma after its opening brace.
        text = '[{"bbox": [0,0,100,100], "text": "hello"}, {"bbox": [0,0,200,200], "text": "incomplete'
        result, was_truncated = truncate_last_incomplete_element(text)
        self.assertTrue(was_truncated)
        self.assertIn('[{"bbox": [0,0,100,100], "text": "hello"}]', result)

    def test_no_truncation_for_single_element(self):
        """Test that single element is not truncated."""
        text = '[{"bbox": [0,0,100,100]'
        result, was_truncated = truncate_last_incomplete_element(text)
        self.assertFalse(was_truncated)


class TestObtainOriginHw(unittest.TestCase):
    """Tests for obtain_origin_hw utility function."""

    def test_from_pil_image(self):
        """Test getting dimensions from PIL Image."""
        img = Image.new("RGB", (800, 600))
        h, w = obtain_origin_hw(img)
        self.assertEqual(h, 600)  # height
        self.assertEqual(w, 800)  # width

    def test_from_file_path(self):
        """Test getting dimensions from file path."""
        temp_dir = tempfile.mkdtemp()
        try:
            img_path = os.path.join(temp_dir, "test.png")
            Image.new("RGB", (1024, 768)).save(img_path)
            h, w = obtain_origin_hw(img_path)
            self.assertEqual(h, 768)
            self.assertEqual(w, 1024)
        finally:
            import shutil
            shutil.rmtree(temp_dir)

    def test_fallback_on_error(self):
        """Test fallback dimensions on error."""
        h, w = obtain_origin_hw("/nonexistent/file.png")
        self.assertEqual(h, 1000)
        self.assertEqual(w, 1000)


class TestRestoreAbsBboxCoordinates(unittest.TestCase):
    """Tests for restore_abs_bbox_coordinates utility function."""

    def test_convert_normalized_to_absolute(self):
        """Test converting normalized [0-1000] bboxes to pixel coordinates."""
        ans = '[{"bbox": [0, 0, 500, 500], "text": "hello"}]'
        result = restore_abs_bbox_coordinates(ans, 1000, 1000)
        data = json.loads(result)
        self.assertEqual(data[0]["bbox"], [0, 0, 500, 500])

    def test_convert_with_actual_dimensions(self):
        """Test converting with actual image dimensions."""
        ans = '[{"bbox": [0, 0, 500, 500], "text": "hello"}]'
        result = restore_abs_bbox_coordinates(ans, 2000, 3000)
        data = json.loads(result)
        self.assertEqual(data[0]["bbox"], [0, 0, 1500, 1000])

    def test_invalid_json_unchanged(self):
        """Test that invalid JSON is returned unchanged."""
        ans = "not valid json"
        result = restore_abs_bbox_coordinates(ans, 1000, 1000)
        self.assertEqual(result, ans)


class TestConvertJsonToMarkdown(unittest.TestCase):
    """Tests for convert_json_to_markdown utility function."""

    def test_convert_layout_to_markdown(self):
        """Test converting layout JSON to markdown."""
        ans = json.dumps([
            {"category": "title", "text": "# Document Title"},
            {"category": "text", "text": "Paragraph content"},
            {"category": "figure", "text": ""},
        ])
        result = convert_json_to_markdown(ans)
        self.assertIn("# Document Title", result)
        self.assertIn("Paragraph content", result)

    def test_strip_header_footer_by_default(self):
        """Test that headers and footers are stripped by default."""
        ans = json.dumps([
            {"category": "header", "text": "Header content"},
            {"category": "text", "text": "Main content"},
            {"category": "footer", "text": "Footer content"},
        ])
        result = convert_json_to_markdown(ans)
        self.assertNotIn("Header content", result)
        self.assertIn("Main content", result)
        self.assertNotIn("Footer content", result)

    def test_keep_header_footer_when_requested(self):
        """Test keeping header and footer when keep_header_footer=True."""
        ans = json.dumps([
            {"category": "header", "text": "Header content"},
            {"category": "text", "text": "Main content"},
            {"category": "footer", "text": "Footer content"},
        ])
        result = convert_json_to_markdown(ans, keep_header_footer=True)
        self.assertIn("Header content", result)
        self.assertIn("Main content", result)
        self.assertIn("Footer content", result)

    def test_invalid_json_returned_unchanged(self):
        """Test that invalid JSON is returned unchanged."""
        ans = "not valid json"
        result = convert_json_to_markdown(ans)
        self.assertEqual(result, ans)


class TestPostprocessDoc2JsonResult(unittest.TestCase):
    """Tests for postprocess_doc2json_result utility function."""

    def test_full_postprocess_pipeline(self):
        """Test the full postprocessing pipeline."""
        raw_text = '```json\n[{"bbox": [0, 0, 500, 500], "category": "text", "text": "Test"}]\n```'
        img = Image.new("RGB", (1000, 1000))
        result = postprocess_doc2json_result(raw_text, img)
        data = json.loads(result)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["category"], "text")


class TestImageMimeTypes(unittest.TestCase):
    """Tests for IMAGE_MIME_TYPES constant."""

    def test_mime_types_defined(self):
        """Test that all expected MIME types are defined."""
        from infinity_parser2.utils.image import IMAGE_MIME_TYPES
        expected_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }
        self.assertEqual(IMAGE_MIME_TYPES, expected_types)


class TestIsSupportedFile(unittest.TestCase):
    """Tests for is_supported_file utility function."""

    def test_supported_image_extensions(self):
        """Test that all expected image extensions are supported."""
        from infinity_parser2.utils import is_supported_file
        expected_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        for ext in expected_extensions:
            self.assertTrue(is_supported_file(f"test/file{ext}"))

    def test_supported_pdf(self):
        """Test that PDF extension is supported."""
        from infinity_parser2.utils import is_supported_file
        self.assertTrue(is_supported_file("test/file.pdf"))
        self.assertTrue(is_supported_file("test/file.PDF"))

    def test_case_insensitive(self):
        """Test that file extension check is case-insensitive."""
        from infinity_parser2.utils import is_supported_file
        self.assertTrue(is_supported_file("test/file.PNG"))
        self.assertTrue(is_supported_file("test/file.JPEG"))
        self.assertTrue(is_supported_file("test/file.TIFF"))

    def test_unsupported_files(self):
        """Test that unsupported files return False."""
        from infinity_parser2.utils import is_supported_file
        self.assertFalse(is_supported_file("test/file.txt"))
        self.assertFalse(is_supported_file("test/file.doc"))
        self.assertFalse(is_supported_file("test/file.xlsx"))


class TestGetFilesFromDirectory(unittest.TestCase):
    """Tests for get_files_from_directory utility function."""

    def setUp(self):
        """Set up temporary test directory with test files."""
        import shutil
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = []
        for ext in [".pdf", ".png", ".jpg", ".txt"]:
            filepath = os.path.join(self.temp_dir, f"test{ext}")
            Path(filepath).touch()
            self.test_files.append(filepath)

    def tearDown(self):
        """Clean up temporary test directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_files_from_directory(self):
        """Test getting supported files from directory."""
        from infinity_parser2.utils import get_files_from_directory
        files = get_files_from_directory(self.temp_dir)
        self.assertEqual(len(files), 3)  # .pdf, .png, .jpg (not .txt)
        for f in files:
            self.assertTrue(f.endswith((".pdf", ".png", ".jpg")))

    def test_files_sorted(self):
        """Test that files are returned in sorted order."""
        from infinity_parser2.utils import get_files_from_directory
        files = get_files_from_directory(self.temp_dir)
        self.assertEqual(files, sorted(files))

    def test_empty_directory(self):
        """Test getting files from empty directory."""
        from infinity_parser2.utils import get_files_from_directory
        empty_dir = tempfile.mkdtemp()
        try:
            files = get_files_from_directory(empty_dir)
            self.assertEqual(len(files), 0)
        finally:
            import shutil
            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_nested_directory(self):
        """Test getting files from nested directories."""
        from infinity_parser2.utils import get_files_from_directory
        nested_dir = os.path.join(self.temp_dir, "nested")
        os.makedirs(nested_dir)
        nested_file = os.path.join(nested_dir, "nested.pdf")
        Path(nested_file).touch()

        files = get_files_from_directory(self.temp_dir)
        self.assertTrue(any(f.endswith("nested.pdf") for f in files))


class TestSaveResults(unittest.TestCase):
    """Tests for save_results utility function.

    New signature: save_results(inputs, results, output_dir, task_type="doc2json", output_format="md")
    Returns None (writes files to disk).
    output_format controls what file is saved: 'md' saves result.md, 'json' saves result.json (doc2json only).
    """

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_results_returns_none(self):
        """Test that save_results returns None."""
        keys = ["test_key"]
        results = ["Test result content"]
        result = save_results(keys, results, self.temp_dir, task_type="doc2md")
        self.assertIsNone(result)

    def test_save_results_creates_directory(self):
        """Test that save_results creates output directory."""
        keys = ["test_key"]
        results = ["Test result content"]
        save_results(keys, results, self.temp_dir, task_type="doc2md")
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_key")))

    def test_save_results_writes_md_file(self):
        """Test that save_results writes result.md for non-doc2json mode."""
        keys = ["test_key"]
        results = ["Test result content"]
        save_results(keys, results, self.temp_dir, task_type="doc2md")
        result_path = os.path.join(self.temp_dir, "test_key", "result.md")
        self.assertTrue(os.path.exists(result_path))
        with open(result_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "Test result content")

    def test_save_results_doc2json_mode_md(self):
        """Test that save_results creates result.md for doc2json mode with output_format='md'."""
        keys = ["test_key"]
        json_result = json.dumps([{"bbox": [0, 0, 100, 100], "category": "text", "text": "Hello"}])
        results = [json_result]
        save_results(keys, results, self.temp_dir, task_type="doc2json", output_format="md")

        json_path = os.path.join(self.temp_dir, "test_key", "result.json")
        md_path = os.path.join(self.temp_dir, "test_key", "result.md")
        self.assertFalse(os.path.exists(json_path))
        self.assertTrue(os.path.exists(md_path))

    def test_save_results_doc2json_mode_json(self):
        """Test that save_results creates result.json for doc2json mode with output_format='json'."""
        keys = ["test_key"]
        json_result = json.dumps([{"bbox": [0, 0, 100, 100], "category": "text", "text": "Hello"}])
        results = [json_result]
        save_results(keys, results, self.temp_dir, task_type="doc2json", output_format="json")

        json_path = os.path.join(self.temp_dir, "test_key", "result.json")
        md_path = os.path.join(self.temp_dir, "test_key", "result.md")
        self.assertTrue(os.path.exists(json_path))
        self.assertFalse(os.path.exists(md_path))
        with open(json_path, "r") as f:
            self.assertEqual(f.read(), json_result)

    def test_save_results_handles_multiple_keys(self):
        """Test saving multiple results."""
        keys = ["key1", "key2", "key3"]
        results = ["Result 1", "Result 2", "Result 3"]
        save_results(keys, results, self.temp_dir, task_type="doc2md")
        for key in keys:
            result_path = os.path.join(self.temp_dir, key, "result.md")
            self.assertTrue(os.path.exists(result_path))

    def test_save_results_output_dir_already_exists(self):
        """Test that save_results works when output dir already exists."""
        os.makedirs(self.temp_dir, exist_ok=True)
        keys = ["test_key"]
        results = ["Test content"]
        save_results(keys, results, self.temp_dir, task_type="doc2md")
        result_path = os.path.join(self.temp_dir, "test_key", "result.md")
        self.assertTrue(os.path.exists(result_path))


class TestModelCache(unittest.TestCase):
    """Tests for ModelCache utility class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, "test_model")
        os.makedirs(self.model_dir)
        self.cache = ModelCache(cache_dir=self.temp_dir)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_dir_creation(self):
        """Test that cache directory is created if it doesn't exist."""
        self.assertTrue(os.path.exists(self.temp_dir))
        # models_file is created lazily on first save
        self.assertIsInstance(self.cache.models_file, str)

    def test_is_cached_returns_false_initially(self):
        """Test that is_cached returns False for uncached models."""
        self.assertFalse(self.cache.is_cached("uncached/model"))

    def test_cache_model(self):
        """Test caching a model."""
        self.cache.cache_model("test/model", self.model_dir)
        self.assertTrue(self.cache.is_cached("test/model"))

    def test_get_cached_path(self):
        """Test getting cached model path."""
        self.cache.cache_model("test/model", self.model_dir)
        self.assertEqual(self.cache.get_cached_path("test/model"), self.model_dir)

    def test_get_cached_path_returns_none_for_uncached(self):
        """Test that get_cached_path returns None for uncached models."""
        self.assertIsNone(self.cache.get_cached_path("uncached/model"))

    def test_resolve_local_path(self):
        """Test resolving a local path directly."""
        result = self.cache.resolve_model_path(self.model_dir)
        self.assertEqual(result, self.model_dir)

    def test_resolve_cached_model(self):
        """Test resolving a cached model returns cached path."""
        self.cache.cache_model("cached/model", self.model_dir)
        result = self.cache.resolve_model_path("cached/model")
        self.assertEqual(result, self.model_dir)

    def test_resolve_nonexistent_model_triggers_download(self):
        """Test that resolving nonexistent model attempts download."""
        with patch.object(self.cache, 'download_and_cache') as mock_download:
            mock_download.return_value = "/downloaded/path"
            result = self.cache.resolve_model_path("nonexistent/model")
            mock_download.assert_called_once_with("nonexistent/model")
            self.assertEqual(result, "/downloaded/path")

    def test_cache_persistence(self):
        """Test that cache persists after reinitialization."""
        self.cache.cache_model("test/model", self.model_dir)
        new_cache = ModelCache(cache_dir=self.temp_dir)
        self.assertTrue(new_cache.is_cached("test/model"))
        self.assertEqual(new_cache.get_cached_path("test/model"), self.model_dir)

    def test_invalid_json_cache_file(self):
        """Test handling of invalid JSON in cache file."""
        cache_file = os.path.join(self.temp_dir, "models_cache.json")
        with open(cache_file, "w") as f:
            f.write("invalid json")
        new_cache = ModelCache(cache_dir=self.temp_dir)
        self.assertFalse(new_cache.is_cached("any/model"))


class TestResolveHFEndpoint(unittest.TestCase):
    """Tests for _resolve_hf_endpoint function."""

    @patch('infinity_parser2.utils.model._check_endpoint_reachable')
    def test_resolves_default_endpoint(self, mock_check):
        """Test that default endpoint is returned when reachable."""
        mock_check.return_value = True
        result = _resolve_hf_endpoint()
        self.assertEqual(result, "https://huggingface.co")

    @patch('infinity_parser2.utils.model._check_endpoint_reachable')
    def test_falls_back_to_mirror(self, mock_check):
        """Test that mirror is used when default is not reachable."""
        mock_check.return_value = False
        result = _resolve_hf_endpoint()
        self.assertEqual(result, "https://hf-mirror.com")


class TestCheckEndpointReachable(unittest.TestCase):
    """Tests for _check_endpoint_reachable function."""

    @patch('urllib.request.urlopen')
    def test_returns_true_on_200(self, mock_urlopen):
        """Test that True is returned on HTTP 200."""
        from infinity_parser2.utils.model import _check_endpoint_reachable
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        result = _check_endpoint_reachable("https://example.com")
        self.assertTrue(result)

    @patch('urllib.request.urlopen')
    def test_returns_false_on_error(self, mock_urlopen):
        """Test that False is returned on connection error."""
        from infinity_parser2.utils.model import _check_endpoint_reachable
        import socket
        mock_urlopen.side_effect = socket.timeout()
        result = _check_endpoint_reachable("https://example.com")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
