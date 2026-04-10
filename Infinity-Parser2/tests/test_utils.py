"""Unit tests for InfinityParser2 utility functions."""

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from infinity_parser2.utils import (
    convert_pdf_to_images,
    encode_file_to_base64,
    get_model_info,
    load_image,
)


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
        """Test encoding PIL Image with explicit format.

        PIL Image format attribute is read-only, set only when loading a file.
        We test encode_file_to_base64 behavior with different Image formats.
        """
        # Create a new PIL Image (no format attribute), should default to image/jpeg
        img = Image.new("RGB", (100, 100), color="white")
        # Newly created Image has no format attribute
        self.assertIsNone(img.format)
        _, mime_type = encode_file_to_base64(img)
        self.assertEqual(mime_type, "image/jpeg")  # Default to jpeg

        # Test saving as PNG then loading to check format attribute
        png_path = os.path.join(self.temp_dir, "test.png")
        Image.new("RGB", (100, 100), color="white").save(png_path)
        # Use with statement to ensure file is properly closed before reopening
        with Image.open(png_path) as loaded_img:
            # For PIL Image objects, use image.format to infer mime_type
            # loaded_img.format will be "PNG" (uppercase)
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
        """Test that unknown extension defaults to image/jpeg.

        For files with unknown extensions, encode_file_to_base64 defaults to image/jpeg.
        This test uses a real .png file to verify this behavior (not .xyz).
        """
        # Create a file with unknown extension
        unknown_path = os.path.join(self.temp_dir, "test.unknown")
        # Save as PNG format first, then rename to .unknown
        Image.new("RGB", (100, 100), color="gray").save(unknown_path, format="PNG")
        # Since extension is unknown, should default to image/jpeg
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

    def test_convert_pdf_requires_pypdf(self):
        """Test that ImportError is raised if pypdf is not available."""
        import sys
        # Temporarily remove pypdf from sys.modules
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("pypdf")]
        original_modules = {}
        for mod in modules_to_remove:
            original_modules[mod] = sys.modules.pop(mod)

        try:
            with patch.dict(sys.modules, {"pypdf": None}):
                with self.assertRaises(ImportError) as context:
                    convert_pdf_to_images("/fake/path.pdf")
            self.assertIn("pypdf", str(context.exception))
        finally:
            # Restore modules
            sys.modules.update(original_modules)

    def test_convert_pdf_requires_pymupdf(self):
        """Test that ImportError is raised if PyMuPDF is not available."""
        import sys
        # Temporarily remove fitz from sys.modules
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("fitz")]
        original_modules = {}
        for mod in modules_to_remove:
            original_modules[mod] = sys.modules.pop(mod)

        try:
            with patch.dict(sys.modules, {"fitz": None}):
                with self.assertRaises(ImportError) as context:
                    convert_pdf_to_images("/fake/path.pdf")
            self.assertIn("PyMuPDF", str(context.exception))
        finally:
            # Restore modules
            sys.modules.update(original_modules)

    def test_convert_pdf_returns_list(self):
        """Test that convert_pdf_to_images returns a list."""
        # Create a simple PDF
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
            page = doc.new_page(width=595, height=842)  # A4 size
            page.insert_text((100, 100), "Test PDF", fontsize=12)
            doc.save(pdf_path)
            doc.close()
            return pdf_path
        except ImportError:
            self.skipTest("PyMuPDF not available for creating test PDF")


class TestGetModelInfo(unittest.TestCase):
    """Tests for get_model_info utility function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_model_info_not_found(self):
        """Test get_model_info for nonexistent path."""
        result = get_model_info("/nonexistent/path")
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["path"], "/nonexistent/path")

    def test_get_model_info_found(self):
        """Test get_model_info for existing directory."""
        # Create some test files with sufficient size (>= 1KB to get non-zero size_mb)
        model_dir = os.path.join(self.temp_dir, "model")
        os.makedirs(model_dir)
        for i in range(3):
            with open(os.path.join(model_dir, f"file{i}.bin"), "w") as f:
                # Write enough bytes to get size_mb > 0 after rounding
                # 1024 * 100 bytes = 100KB
                f.write("x" * (1024 * 100))

        result = get_model_info(model_dir)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["path"], model_dir)
        self.assertIn("size_mb", result)
        self.assertGreater(result["size_mb"], 0)

    def test_get_model_info_empty_directory(self):
        """Test get_model_info for empty directory."""
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir)
        result = get_model_info(empty_dir)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["size_mb"], 0)

    def test_get_model_info_nested_directories(self):
        """Test get_model_info with nested directories."""
        model_dir = os.path.join(self.temp_dir, "model")
        nested_dir = os.path.join(model_dir, "nested")
        os.makedirs(nested_dir)
        for i in range(2):
            with open(os.path.join(nested_dir, f"nested{i}.bin"), "w") as f:
                # Write enough bytes to get non-zero size_mb
                f.write("y" * (1024 * 100))

        result = get_model_info(model_dir)
        self.assertEqual(result["status"], "found")
        self.assertGreater(result["size_mb"], 0)


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


if __name__ == "__main__":
    unittest.main()
