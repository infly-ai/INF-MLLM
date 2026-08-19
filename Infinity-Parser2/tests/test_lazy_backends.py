import sys
import unittest

from PIL import Image


class TestLazyBackendImports(unittest.TestCase):
    def test_vllm_server_import_skips_local_backends(self):
        from infinity_parser2.backends import VLLMServerBackend

        self.assertIsNotNone(VLLMServerBackend)
        self.assertNotIn("infinity_parser2.backends.transformers", sys.modules)
        self.assertNotIn("infinity_parser2.backends.vllm_engine", sys.modules)

    def test_image_encoding_falls_back_without_qwen_utils(self):
        from infinity_parser2.utils.image import encode_image_to_base64

        encoded, mime_type = encode_image_to_base64(Image.new("RGB", (10, 10)))

        self.assertTrue(encoded)
        self.assertEqual(mime_type, "image/jpeg")
