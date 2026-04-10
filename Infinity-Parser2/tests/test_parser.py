"""Unit tests for InfinityParser2 main parser class."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from infinity_parser2 import InfinityParser2


class TestInfinityParser2Initialization(unittest.TestCase):
    """Tests for InfinityParser2 initialization and configuration."""

    def test_default_initialization(self):
        """Test default initialization with all default parameters."""
        parser = InfinityParser2()
        self.assertEqual(parser.model_name, "infly/Infinity-Parser2-Pro")
        self.assertEqual(parser.backend_name, "vllm-engine")
        self.assertEqual(parser.tensor_parallel_size, 1)
        self.assertEqual(parser.device, "cuda")
        self.assertEqual(parser.api_url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(parser.api_key, "EMPTY")
        self.assertEqual(parser.min_pixels, 2048)
        self.assertEqual(parser.max_pixels, 16777216)

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        parser = InfinityParser2(
            model_name="custom/model",
            backend="transformers",
            tensor_parallel_size=2,
            device="cpu",
            api_url="http://custom:8000/v1",
            api_key="test-key",
            min_pixels=1024,
            max_pixels=8192,
        )
        self.assertEqual(parser.model_name, "custom/model")
        self.assertEqual(parser.backend_name, "transformers")
        self.assertEqual(parser.tensor_parallel_size, 2)
        self.assertEqual(parser.device, "cpu")
        self.assertEqual(parser.api_url, "http://custom:8000/v1")
        self.assertEqual(parser.api_key, "test-key")
        self.assertEqual(parser.min_pixels, 1024)
        self.assertEqual(parser.max_pixels, 8192)

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


class TestInfinityParser2SupportedFormats(unittest.TestCase):
    """Tests for supported file format detection."""

    def test_supported_image_extensions(self):
        """Test that all expected image extensions are supported."""
        parser = InfinityParser2()
        expected_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        self.assertEqual(parser.SUPPORTED_IMAGE_EXTENSIONS, expected_extensions)

    def test_supported_doc_extensions(self):
        """Test that PDF extension is supported."""
        parser = InfinityParser2()
        expected_extensions = {".pdf"}
        self.assertEqual(parser.SUPPORTED_DOC_EXTENSIONS, expected_extensions)

    def test_is_supported_file_with_image(self):
        """Test supported file detection for image files."""
        parser = InfinityParser2()
        for ext in parser.SUPPORTED_IMAGE_EXTENSIONS:
            self.assertTrue(parser._is_supported_file(f"test/file{ext}"))

    def test_is_supported_file_with_pdf(self):
        """Test supported file detection for PDF files."""
        parser = InfinityParser2()
        self.assertTrue(parser._is_supported_file("test/file.pdf"))
        self.assertTrue(parser._is_supported_file("test/file.PDF"))

    def test_is_supported_file_case_insensitive(self):
        """Test that file extension check is case-insensitive."""
        parser = InfinityParser2()
        self.assertTrue(parser._is_supported_file("test/file.PNG"))
        self.assertTrue(parser._is_supported_file("test/file.JPEG"))
        self.assertTrue(parser._is_supported_file("test/file.TIFF"))

    def test_is_supported_file_unsupported(self):
        """Test supported file detection for unsupported files."""
        parser = InfinityParser2()
        self.assertFalse(parser._is_supported_file("test/file.txt"))
        self.assertFalse(parser._is_supported_file("test/file.doc"))
        self.assertFalse(parser._is_supported_file("test/file.xlsx"))


class TestInfinityParser2BackendProperty(unittest.TestCase):
    """Tests for backend lazy initialization."""

    def setUp(self):
        """Set up test fixtures with mocked backend."""
        self.parser = InfinityParser2(backend="vllm-engine")
        # Mock the backend to avoid actual model loading
        self.mock_backend = MagicMock()
        self.parser._backend = self.mock_backend

    def test_backend_lazy_initialization(self):
        """Test that backend is not initialized until accessed."""
        # Create new parser to test lazy loading behavior
        parser = InfinityParser2(backend="vllm-engine")
        self.assertIsNone(parser._backend)
        # Directly set mock backend to simulate state after lazy loading
        parser._backend = MagicMock()
        self.assertIsNotNone(parser._backend)

    def test_backend_returns_correct_type(self):
        """Test that backend property returns correct backend instance."""
        # Verify parser._backend is the mock backend we set
        self.assertIs(self.parser.backend, self.mock_backend)

    def test_backend_cached_after_first_access(self):
        """Test that backend is cached after first access."""
        backend1 = self.parser.backend
        backend2 = self.parser.backend
        self.assertIs(backend1, backend2)


class TestInfinityParser2DirectoryScanning(unittest.TestCase):
    """Tests for directory scanning functionality."""

    def setUp(self):
        """Set up temporary test directory with test files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_files = []
        for ext in [".pdf", ".png", ".jpg", ".txt"]:
            filepath = os.path.join(self.test_dir, f"test{ext}")
            Path(filepath).touch()
            self.test_files.append(filepath)

    def tearDown(self):
        """Clean up temporary test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_get_files_from_directory(self):
        """Test getting supported files from directory."""
        parser = InfinityParser2()
        files = parser._get_files_from_directory(self.test_dir)
        self.assertEqual(len(files), 3)  # .pdf, .png, .jpg (not .txt)
        for f in files:
            self.assertTrue(f.endswith((".pdf", ".png", ".jpg")))

    def test_get_files_sorted(self):
        """Test that files are returned in sorted order."""
        parser = InfinityParser2()
        files = parser._get_files_from_directory(self.test_dir)
        self.assertEqual(files, sorted(files))

    def test_get_files_empty_directory(self):
        """Test getting files from empty directory."""
        empty_dir = tempfile.mkdtemp()
        try:
            parser = InfinityParser2()
            files = parser._get_files_from_directory(empty_dir)
            self.assertEqual(len(files), 0)
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_get_files_nested_directory(self):
        """Test getting files from nested directories."""
        nested_dir = os.path.join(self.test_dir, "nested")
        os.makedirs(nested_dir)
        nested_file = os.path.join(nested_dir, "nested.pdf")
        Path(nested_file).touch()

        parser = InfinityParser2()
        files = parser._get_files_from_directory(self.test_dir)
        self.assertTrue(any(f.endswith("nested.pdf") for f in files))


class TestInfinityParser2ParseInputValidation(unittest.TestCase):
    """Tests for parse method input validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = InfinityParser2(backend="vllm-engine")
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test.png")
        Image.new("RGB", (100, 100), color="white").save(self.temp_file)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_nonexistent_file_raises_error(self):
        """Test that parsing nonexistent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.parser.parse("nonexistent_file.pdf")

    def test_parse_unsupported_file_raises_error(self):
        """Test that parsing unsupported file type raises ValueError."""
        txt_file = os.path.join(self.temp_dir, "test.txt")
        Path(txt_file).touch()
        with self.assertRaises(ValueError) as context:
            self.parser.parse(txt_file)
        self.assertIn("Unsupported file type", str(context.exception))

    def test_parse_list_with_invalid_item_raises_error(self):
        """Test that parsing list with invalid item raises TypeError."""
        with self.assertRaises(TypeError):
            self.parser.parse([123, "not_a_string"])

    def test_parse_list_with_nonexistent_file_raises_error(self):
        """Test that parsing list with nonexistent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.parser.parse([self.temp_file, "nonexistent.pdf"])

    def test_parse_list_with_unsupported_file_raises_error(self):
        """Test that parsing list with unsupported file raises ValueError."""
        txt_file = os.path.join(self.temp_dir, "test.txt")
        Path(txt_file).touch()
        with self.assertRaises(ValueError):
            self.parser.parse([self.temp_file, txt_file])

    def test_parse_unsupported_input_type_raises_error(self):
        """Test that parsing unsupported input type raises TypeError."""
        with self.assertRaises(TypeError) as context:
            self.parser.parse(12345)
        self.assertIn("Unsupported input type", str(context.exception))

    def test_parse_directory_with_no_supported_files_raises_error(self):
        """Test that parsing directory with no supported files raises ValueError."""
        empty_dir = tempfile.mkdtemp()
        txt_file = os.path.join(empty_dir, "test.txt")
        Path(txt_file).touch()
        try:
            with self.assertRaises(ValueError) as context:
                self.parser.parse(empty_dir)
            self.assertIn("No supported files found", str(context.exception))
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)


class TestInfinityParser2SaveResults(unittest.TestCase):
    """Tests for result saving functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = InfinityParser2(backend="vllm-engine")
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_save_results_creates_directory(self):
        """Test that save_results creates output directory."""
        keys = ["test_key"]
        results = ["Test result content"]
        saved_paths = self.parser._save_results(keys, results, self.output_dir)
        self.assertIn("test_key", saved_paths)
        self.assertTrue(os.path.exists(saved_paths["test_key"]))

    def test_save_results_writes_content(self):
        """Test that save_results writes correct content to file."""
        keys = ["test_key"]
        results = ["Test result content"]
        saved_paths = self.parser._save_results(keys, results, self.output_dir)
        with open(saved_paths["test_key"], "r") as f:
            content = f.read()
        self.assertEqual(content, "Test result content")

    def test_save_results_creates_subdirectory(self):
        """Test that save_results creates subdirectory for each key."""
        keys = ["key1", "key2"]
        results = ["Result 1", "Result 2"]
        saved_paths = self.parser._save_results(keys, results, self.output_dir)
        for key in keys:
            self.assertTrue(os.path.isdir(os.path.join(self.output_dir, key)))

    def test_save_results_handles_multiple_keys(self):
        """Test saving multiple results."""
        keys = ["key1", "key2", "key3"]
        results = ["Result 1", "Result 2", "Result 3"]
        saved_paths = self.parser._save_results(keys, results, self.output_dir)
        self.assertEqual(len(saved_paths), 3)
        for key, result in zip(keys, results):
            with open(saved_paths[key], "r") as f:
                self.assertEqual(f.read(), result)

    def test_save_results_output_dir_already_exists(self):
        """Test that save_results works when output dir already exists."""
        os.makedirs(self.output_dir, exist_ok=True)
        keys = ["test_key"]
        results = ["Test content"]
        saved_paths = self.parser._save_results(keys, results, self.output_dir)
        self.assertTrue(os.path.exists(saved_paths["test_key"]))


class TestInfinityParser2MockedParse(unittest.TestCase):
    """Tests for parse method with mocked backend."""

    def setUp(self):
        """Set up test fixtures with mocked backend."""
        self.parser = InfinityParser2(backend="vllm-engine")
        self.mock_backend = MagicMock()
        self.parser._backend = self.mock_backend
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_single_file_returns_string(self):
        """Test that parsing single file returns string."""
        self.mock_backend.parse_batch.return_value = ["Parsed content"]
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            result = self.parser.parse(temp_file.name)
            self.assertIsInstance(result, str)
            self.assertEqual(result, "Parsed content")
        finally:
            os.unlink(temp_file.name)

    def test_parse_list_returns_list(self):
        """Test that parsing list returns list of strings."""
        self.mock_backend.parse_batch.return_value = ["Result 1", "Result 2"]
        temp_files = []
        for i in range(2):
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_files.append(f.name)
        try:
            result = self.parser.parse(temp_files)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
        finally:
            for f in temp_files:
                os.unlink(f)

    def test_parse_pil_image(self):
        """Test parsing PIL Image object."""
        self.mock_backend.parse_batch.return_value = ["Image content"]
        img = Image.new("RGB", (100, 100), color="white")
        result = self.parser.parse(img)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Image content")

    def test_parse_with_output_dir_returns_dict(self):
        """Test that parsing with output_dir returns dict."""
        self.mock_backend.parse_batch.return_value = ["Result content"]
        # Use real temp directory file instead of NamedTemporaryFile
        # because output_dir uses file path as subdirectory name
        temp_file = os.path.join(self.temp_dir, "test.png")
        Image.new("RGB", (100, 100), color="white").save(temp_file)
        output_dir = tempfile.mkdtemp()
        try:
            result = self.parser.parse(temp_file, output_dir=output_dir)
            self.assertIsInstance(result, dict)
            self.assertIn(temp_file, result)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_parse_batch_size_passed_to_backend(self):
        """Test that batch_size is passed to backend correctly."""
        self.mock_backend.parse_batch.return_value = ["Result"]
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            self.parser.parse(temp_file.name, batch_size=4)
            self.mock_backend.parse_batch.assert_called_once()
            call_kwargs = self.mock_backend.parse_batch.call_args[1]
            self.assertEqual(call_kwargs["batch_size"], 4)
        finally:
            os.unlink(temp_file.name)


if __name__ == "__main__":
    unittest.main()
