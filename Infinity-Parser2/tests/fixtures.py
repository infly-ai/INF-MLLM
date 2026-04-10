"""Shared unittest fixtures for InfinityParser2 tests.

This module provides reusable test utility functions and base classes
for consistent test environment management.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class TestFixtures(unittest.TestCase):
    """Base test class with common fixtures."""

    def setUp(self):
        """Set up common test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test.png")
        self.test_image = Image.new("RGB", (100, 100), color="white")
        self.test_image.save(self.test_image_path)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def create_test_image(directory: str, filename: str, size: tuple = (100, 100),
                     color: str = "red") -> str:
    """Helper function to create a test image.

    Args:
        directory: Directory to save the image.
        filename: Filename for the image.
        size: Image size as (width, height).
        color: Fill color for the image.

    Returns:
        Full path to the created image.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    Image.new("RGB", size, color=color).save(filepath)
    return filepath


def create_test_pdf(directory: str, filename: str, num_pages: int = 1) -> str:
    """Helper function to create a simple test PDF.

    Args:
        directory: Directory to save the PDF.
        filename: Filename for the PDF.
        num_pages: Number of pages in the PDF.

    Returns:
        Full path to the created PDF, or None if PyMuPDF is not available.
    """
    try:
        import fitz
    except ImportError:
        return None

    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    doc = fitz.open()

    for _ in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((100, 100), "Test Content", fontsize=12)

    doc.save(filepath)
    doc.close()
    return filepath
