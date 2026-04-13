"""Unit tests for InfinityParser2 main parser class."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from infinity_parser2 import InfinityParser2, SUPPORTED_TASK_TYPES
from infinity_parser2.backends import TransformersBackend


class TestInfinityParser2Initialization(unittest.TestCase):
    """Tests for InfinityParser2 initialization and configuration."""

    @patch("infinity_parser2.parser.get_model_cache")
    @patch("infinity_parser2.backends.vllm_engine.VLLMEngineBackend.__init__", return_value=None)
    def test_default_initialization(self, mock_backend_init, mock_get_cache):
        """Test default initialization with all default parameters."""
        mock_cache = MagicMock()
        mock_cache.resolve_model_path.return_value = "/cached/path/infly_Infinity-Parser2-Pro"
        mock_get_cache.return_value = mock_cache
        parser = InfinityParser2()
        self.assertEqual(parser.model_name, "infly/Infinity-Parser2-Pro")
        self.assertEqual(parser.backend_name, "vllm-engine")
        self.assertEqual(parser.device, "cuda")
        self.assertEqual(parser.api_url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(parser.api_key, "EMPTY")
        self.assertEqual(parser.min_pixels, 2048)
        self.assertEqual(parser.max_pixels, 16777216)
        self.assertIsInstance(parser.tensor_parallel_size, int)

    @patch("infinity_parser2.parser.get_model_cache")
    @patch("infinity_parser2.backends.transformers.TransformersBackend.__init__", return_value=None)
    def test_custom_initialization(self, mock_backend_init, mock_get_cache):
        """Test initialization with custom parameters."""
        mock_cache = MagicMock()
        mock_cache.resolve_model_path.return_value = "/cached/path/custom/model"
        mock_get_cache.return_value = mock_cache
        parser = InfinityParser2(
            model_name="custom/model",
            backend="transformers",
            tensor_parallel_size=2,
            api_url="http://custom:8000/v1",
            api_key="test-key",
            min_pixels=1024,
            max_pixels=8192,
        )
        self.assertEqual(parser.model_name, "custom/model")
        self.assertEqual(parser.backend_name, "transformers")
        self.assertEqual(parser.tensor_parallel_size, 2)
        self.assertEqual(parser.device, "cuda")
        self.assertEqual(parser.api_url, "http://custom:8000/v1")
        self.assertEqual(parser.api_key, "test-key")
        self.assertEqual(parser.min_pixels, 1024)
        self.assertEqual(parser.max_pixels, 8192)

    def test_device_must_be_cuda(self):
        """Test that device must be 'cuda', raising ValueError otherwise."""
        with self.assertRaises(ValueError) as context:
            InfinityParser2(device="cpu")
        self.assertIn("cuda", str(context.exception))

    @patch("infinity_parser2.parser.get_model_cache")
    @patch("infinity_parser2.backends.vllm_engine.VLLMEngineBackend.__init__", return_value=None)
    def test_backend_case_insensitive(self, mock_backend_init, mock_get_cache):
        """Test that backend name is case-insensitive."""
        mock_cache = MagicMock()
        mock_cache.resolve_model_path.return_value = "/cached/path/model"
        mock_get_cache.return_value = mock_cache
        parser1 = InfinityParser2(backend="VLLM-ENGINE")
        parser2 = InfinityParser2(backend="vllm-engine")
        self.assertEqual(parser1.backend_name, "vllm-engine")
        self.assertEqual(parser2.backend_name, "vllm-engine")
        with self.assertRaises(ValueError) as context:
            InfinityParser2(backend="VLLMEngine")
        self.assertIn("Unsupported backend", str(context.exception))

    def test_unsupported_backend_raises_error(self):
        """Test that unsupported backend raises ValueError."""
        with self.assertRaises(ValueError) as context:
            InfinityParser2(backend="unsupported-backend")
        self.assertIn("Unsupported backend", str(context.exception))
        self.assertIn("unsupported-backend", str(context.exception))

    def test_backend_registry_contains_all_backends(self):
        """Test that BACKEND_REGISTRY contains all expected backends."""
        from infinity_parser2.parser import BACKEND_REGISTRY
        expected_backends = {"transformers", "vllm-engine", "vllm-server"}
        self.assertEqual(set(BACKEND_REGISTRY.keys()), expected_backends)


class TestInfinityParser2BackendProperty(unittest.TestCase):
    """Tests for backend property."""

    @patch("infinity_parser2.backends.vllm_engine.LLM")
    def test_backend_is_initialized_on_init(self, mock_llm):
        """Test that backend is initialized during __init__."""
        mock_llm.return_value = MagicMock()
        parser = InfinityParser2(backend="vllm-engine")
        self.assertIsNotNone(parser._backend)

    @patch("infinity_parser2.backends.transformers.AutoModelForCausalLM")
    @patch("infinity_parser2.backends.transformers.AutoProcessor")
    def test_backend_returns_correct_type(self, mock_processor, mock_model):
        """Test that backend returns correct backend instance."""
        mock_model.from_pretrained.return_value = MagicMock()
        mock_processor.from_pretrained.return_value = MagicMock()
        parser = InfinityParser2(backend="transformers")
        self.assertIsInstance(parser._backend, TransformersBackend)

    @patch("infinity_parser2.backends.vllm_engine.LLM")
    def test_backend_same_instance_on_multiple_accesses(self, mock_llm):
        """Test that backend returns the same instance."""
        mock_llm.return_value = MagicMock()
        parser = InfinityParser2(backend="vllm-engine")
        backend1 = parser._backend
        backend2 = parser._backend
        self.assertIs(backend1, backend2)


class TestInfinityParser2ParseInputValidation(unittest.TestCase):
    """Tests for parse method input validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test.png")
        Image.new("RGB", (100, 100), color="white").save(self.temp_file)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_parser(self):
        """Create parser with mocked backend."""
        parser = InfinityParser2(backend="vllm-engine")
        parser._backend = MagicMock()
        return parser

    def test_parse_nonexistent_file_raises_error(self):
        """Test that parsing nonexistent file raises FileNotFoundError."""
        parser = self._make_parser()
        with self.assertRaises(FileNotFoundError):
            parser.parse("nonexistent_file.pdf")

    def test_parse_unsupported_file_raises_error(self):
        """Test that parsing unsupported file type raises ValueError."""
        txt_file = os.path.join(self.temp_dir, "test.txt")
        Path(txt_file).touch()
        parser = self._make_parser()
        with self.assertRaises(ValueError) as context:
            parser.parse(txt_file)
        self.assertIn("Unsupported file type", str(context.exception))

    def test_parse_list_with_invalid_item_raises_error(self):
        """Test that parsing list with invalid item raises TypeError."""
        parser = self._make_parser()
        with self.assertRaises(TypeError):
            parser.parse([123, "not_a_string"])

    def test_parse_list_with_nonexistent_file_raises_error(self):
        """Test that parsing list with nonexistent file raises FileNotFoundError."""
        parser = self._make_parser()
        with self.assertRaises(FileNotFoundError):
            parser.parse([self.temp_file, "nonexistent.pdf"])

    def test_parse_list_with_unsupported_file_raises_error(self):
        """Test that parsing list with unsupported file raises ValueError."""
        txt_file = os.path.join(self.temp_dir, "test.txt")
        Path(txt_file).touch()
        parser = self._make_parser()
        with self.assertRaises(ValueError):
            parser.parse([self.temp_file, txt_file])

    def test_parse_unsupported_input_type_raises_error(self):
        """Test that parsing unsupported input type raises TypeError."""
        parser = self._make_parser()
        with self.assertRaises(TypeError) as context:
            parser.parse(12345)
        self.assertIn("Unsupported input type", str(context.exception))

    def test_parse_directory_with_no_supported_files_raises_error(self):
        """Test that parsing directory with no supported files raises ValueError."""
        empty_dir = tempfile.mkdtemp()
        txt_file = os.path.join(empty_dir, "test.txt")
        Path(txt_file).touch()
        try:
            parser = self._make_parser()
            with self.assertRaises(ValueError) as context:
                parser.parse(empty_dir)
            self.assertIn("No supported files found", str(context.exception))
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)


class TestInfinityParser2MockedParse(unittest.TestCase):
    """Tests for parse method with mocked backend."""

    def setUp(self):
        """Set up test fixtures with mocked backend."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_parser(self):
        """Create parser with mocked backend."""
        parser = InfinityParser2(backend="vllm-engine")
        parser._backend = MagicMock()
        return parser

    def test_parse_single_file_returns_string(self):
        """Test that parsing single file returns string."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ["Parsed content"]
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            result = parser.parse(temp_file.name)
            self.assertIsInstance(result, str)
            self.assertEqual(result, "Parsed content")
        finally:
            os.unlink(temp_file.name)

    def test_parse_single_image_doc2json_converts_to_markdown(self):
        """Test that DOC2JSON mode converts JSON to Markdown for single image."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ['[{"category": "title", "text": "# Document Title"}]']
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            result = parser.parse(temp_file.name)
            self.assertIsInstance(result, str)
            self.assertIn("# Document Title", result)
            self.assertNotIn("[{", result)
        finally:
            os.unlink(temp_file.name)

    def test_parse_list_returns_list(self):
        """Test that parsing list returns list of strings."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ["Result 1", "Result 2"]
        temp_files = []
        for i in range(2):
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_files.append(f.name)
        try:
            result = parser.parse(temp_files)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
        finally:
            for f in temp_files:
                os.unlink(f)

    def test_parse_pil_image_doc2json(self):
        """Test parsing PIL Image object in DOC2JSON mode converts JSON to Markdown."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ['[{"category": "text", "text": "Image content"}]']
        img = Image.new("RGB", (100, 100), color="white")
        result = parser.parse(img)
        self.assertIsInstance(result, str)
        self.assertIn("Image content", result)
        self.assertNotIn("[{", result)

    def test_parse_with_output_dir_creates_subdirectories(self):
        """Test that parsing with output_dir creates subdirectories."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ["Result content"]
        temp_file = os.path.join(self.temp_dir, "test.png")
        Image.new("RGB", (100, 100), color="white").save(temp_file)
        output_dir = tempfile.mkdtemp()
        try:
            parser.parse(temp_file, output_dir=output_dir)
            subdir = os.path.join(output_dir, "test.png")
            self.assertTrue(os.path.exists(subdir))
            result_file = os.path.join(subdir, "result.md")
            self.assertTrue(os.path.exists(result_file))
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_parse_batch_size_passed_to_backend(self):
        """Test that batch_size is passed to backend correctly."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ["Result"]
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            parser.parse(temp_file.name, batch_size=4)
            parser._backend.parse_batch.assert_called_once()
            call_kwargs = parser._backend.parse_batch.call_args[1]
            self.assertEqual(call_kwargs["batch_size"], 4)
        finally:
            os.unlink(temp_file.name)

    def test_parse_with_task_type_doc2md(self):
        """Test parsing with doc2md task type."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ["# Title\n\nParagraph text"]
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            result = parser.parse(temp_file.name, task_type="doc2md")
            self.assertIsInstance(result, str)
            self.assertIn("# Title", result)
        finally:
            os.unlink(temp_file.name)

    def test_parse_with_output_format_json(self):
        """Test parsing with output_format='json' returns raw JSON."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ['[{"bbox": [0,0,100,100], "category": "text", "text": "Hello"}]']
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            result = parser.parse(temp_file.name, output_format="json")
            self.assertIsInstance(result, str)
            self.assertIn('"category": "text"', result)
            self.assertIn('"Hello"', result)
        finally:
            os.unlink(temp_file.name)

    def test_parse_with_output_format_invalid(self):
        """Test that invalid output_format raises ValueError."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ["Result"]
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            with self.assertRaises(ValueError) as context:
                parser.parse(temp_file.name, output_format="xml")
            self.assertIn("output_format must be one of", str(context.exception))
        finally:
            os.unlink(temp_file.name)

    def test_parse_doc2md_cannot_use_output_format_json(self):
        """Test that DOC2MD mode cannot use output_format='json'."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ["# Title"]
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            with self.assertRaises(ValueError) as context:
                parser.parse(temp_file.name, task_type="doc2md", output_format="json")
            self.assertIn("output_format='json' is only supported for doc2json tasks", str(context.exception))
        finally:
            os.unlink(temp_file.name)

    def test_parse_custom_prompt_cannot_use_output_format_json(self):
        """Test that custom prompt cannot use output_format='json'."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ["Custom result"]
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            with self.assertRaises(ValueError) as context:
                parser.parse(temp_file.name, custom_prompt="Custom instruction", output_format="json")
            self.assertIn("output_format='json' is only supported for doc2json tasks", str(context.exception))
        finally:
            os.unlink(temp_file.name)

    def test_parse_with_output_dir_and_output_format_json(self):
        """Test parsing with output_dir and output_format='json' saves only JSON."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ['[{"bbox": [0,0,100,100], "category": "text", "text": "Hello"}]']
        temp_file = os.path.join(self.temp_dir, "test.png")
        Image.new("RGB", (100, 100), color="white").save(temp_file)
        output_dir = tempfile.mkdtemp()
        try:
            parser.parse(temp_file, output_dir=output_dir, output_format="json")
            subdir = os.path.join(output_dir, "test.png")
            self.assertTrue(os.path.exists(subdir))
            json_file = os.path.join(subdir, "result.json")
            self.assertTrue(os.path.exists(json_file))
            md_file = os.path.join(subdir, "result.md")
            self.assertFalse(os.path.exists(md_file))
            with open(json_file, "r") as f:
                content = f.read()
            self.assertIn('"category": "text"', content)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_parse_with_output_dir_and_output_format_md(self):
        """Test parsing with output_dir and output_format='md' (default) saves only Markdown."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ['[{"bbox": [0,0,100,100], "category": "text", "text": "Hello"}]']
        temp_file = os.path.join(self.temp_dir, "test.png")
        Image.new("RGB", (100, 100), color="white").save(temp_file)
        output_dir = tempfile.mkdtemp()
        try:
            parser.parse(temp_file, output_dir=output_dir, output_format="md")
            subdir = os.path.join(output_dir, "test.png")
            self.assertTrue(os.path.exists(subdir))
            md_file = os.path.join(subdir, "result.md")
            self.assertTrue(os.path.exists(md_file))
            json_file = os.path.join(subdir, "result.json")
            self.assertFalse(os.path.exists(json_file))
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_parse_with_custom_prompt(self):
        """Test parsing with custom_prompt (task_type='custom')."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ["Custom result"]
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            result = parser.parse(temp_file.name, task_type="custom", custom_prompt="Custom instruction")
            self.assertIsInstance(result, str)
            self.assertEqual(result, "Custom result")
            call_args = parser._backend.parse_batch.call_args
            self.assertEqual(call_args[0][1], "Custom instruction")
        finally:
            os.unlink(temp_file.name)

    def test_parse_directory(self):
        """Test parsing a directory of files."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ["Result1", "Result2"]
        dir_path = tempfile.mkdtemp()
        try:
            file1 = os.path.join(dir_path, "file1.png")
            file2 = os.path.join(dir_path, "file2.png")
            Image.new("RGB", (100, 100), color="red").save(file1)
            Image.new("RGB", (100, 100), color="blue").save(file2)
            result = parser.parse(dir_path)
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 2)
            for path, content in result.items():
                self.assertIsInstance(path, str)
                self.assertIsInstance(content, str)
        finally:
            shutil.rmtree(dir_path)


class TestTaskType(unittest.TestCase):
    """Tests for task_type parameter and SUPPORTED_TASK_TYPES."""

    def test_supported_task_types(self):
        """Test SUPPORTED_TASK_TYPES contains expected values."""
        self.assertEqual(SUPPORTED_TASK_TYPES, ["doc2json", "doc2md", "custom"])

    def test_task_type_doc2json(self):
        """Test that 'doc2json' is a supported task type."""
        self.assertIn("doc2json", SUPPORTED_TASK_TYPES)

    def test_task_type_doc2md(self):
        """Test that 'doc2md' is a supported task type."""
        self.assertIn("doc2md", SUPPORTED_TASK_TYPES)

    def test_task_type_custom(self):
        """Test that 'custom' is a supported task type."""
        self.assertIn("custom", SUPPORTED_TASK_TYPES)

    def test_prompt_doc2json_defined(self):
        """Test that PROMPT_DOC2JSON is defined."""
        from infinity_parser2 import PROMPT_DOC2JSON
        self.assertIsInstance(PROMPT_DOC2JSON, str)
        self.assertIn("layout", PROMPT_DOC2JSON.lower())

    def test_prompt_doc2md_defined(self):
        """Test that PROMPT_DOC2MD is defined."""
        from infinity_parser2 import PROMPT_DOC2MD
        self.assertIsInstance(PROMPT_DOC2MD, str)
        self.assertIn("markdown", PROMPT_DOC2MD.lower())


if __name__ == "__main__":
    unittest.main()
