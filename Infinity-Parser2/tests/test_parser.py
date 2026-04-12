"""Unit tests for InfinityParser2 main parser class."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from infinity_parser2 import InfinityParser2
from infinity_parser2.backends import TransformersBackend


class TestInfinityParser2Initialization(unittest.TestCase):
    """Tests for InfinityParser2 initialization and configuration."""

    def test_default_initialization(self):
        """Test default initialization with all default parameters."""
        parser = InfinityParser2()
        self.assertEqual(parser.model_name, "infly/Infinity-Parser2-Pro")
        self.assertEqual(parser.backend_name, "vllm-engine")
        self.assertEqual(parser.device, "cuda")
        self.assertEqual(parser.api_url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(parser.api_key, "EMPTY")
        self.assertEqual(parser.min_pixels, 2048)
        self.assertEqual(parser.max_pixels, 16777216)
        # tensor_parallel_size defaults to number of available GPUs
        self.assertIsInstance(parser.tensor_parallel_size, int)

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
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

    def test_backend_case_insensitive(self):
        """Test that backend name is case-insensitive."""
        parser1 = InfinityParser2(backend="VLLM-ENGINE")
        parser2 = InfinityParser2(backend="vllm-engine")
        # VLLMEngine (no hyphen) is not a valid backend name, should raise error
        # Valid names are only: transformers, vllm-engine, vllm-server
        self.assertEqual(parser1.backend_name, "vllm-engine")
        self.assertEqual(parser2.backend_name, "vllm-engine")
        # Verify VLLMEngine format raises error (not a valid backend name)
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

    def test_parse_pil_image(self):
        """Test parsing PIL Image object."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ["Image content"]
        img = Image.new("RGB", (100, 100), color="white")
        result = parser.parse(img)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Image content")

    def test_parse_with_output_dir_returns_dict(self):
        """Test that parsing with output_dir returns dict."""
        parser = self._make_parser()
        parser._backend.parse_batch.return_value = ["Result content"]
        temp_file = os.path.join(self.temp_dir, "test.png")
        Image.new("RGB", (100, 100), color="white").save(temp_file)
        output_dir = tempfile.mkdtemp()
        try:
            result = parser.parse(temp_file, output_dir=output_dir)
            self.assertIsInstance(result, dict)
            self.assertIn(temp_file, result)
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


if __name__ == "__main__":
    unittest.main()
