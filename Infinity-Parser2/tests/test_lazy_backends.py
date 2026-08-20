"""Isolated regression tests for lazy backend/dependency loading.

Each scenario runs in a fresh subprocess with an import blocker installed
before infinity_parser2 is touched, so the result reflects what that
scenario actually imports rather than whatever a sibling test already
pulled into this process's sys.modules.
"""

import subprocess
import sys
import textwrap
import unittest


def _blocker_code(blocked) -> str:
    return (
        "import builtins\n"
        "_real_import = builtins.__import__\n"
        f"_blocked = {tuple(blocked)!r}\n"
        "def _blocking_import(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in _blocked:\n"
        "        raise ImportError('blocked for test isolation: ' + name)\n"
        "    return _real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = _blocking_import\n"
    )


def _run_isolated(body: str, blocked) -> subprocess.CompletedProcess:
    script = _blocker_code(blocked) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestLazyBackendImports(unittest.TestCase):
    """infinity_parser2 must not require torch/torchvision/vllm/transformers
    just to be imported or to use the remote vllm-server backend."""

    HEAVY_DEPS = ("torch", "torchvision", "vllm", "transformers")

    def test_bare_import_does_not_load_heavy_backends(self):
        result = _run_isolated(
            """
            import sys
            import infinity_parser2  # noqa: F401

            assert "infinity_parser2.backends.transformers" not in sys.modules
            assert "infinity_parser2.backends.vllm_engine" not in sys.modules
            assert "torch" not in sys.modules
            assert "vllm" not in sys.modules
            print("OK")
            """,
            self.HEAVY_DEPS,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)

    def test_vllm_server_backend_avoids_torch_and_auto_tensor_parallel(self):
        result = _run_isolated(
            """
            import sys
            from unittest.mock import patch
            from infinity_parser2.backends.vllm_server import VLLMServerBackend

            with patch.object(VLLMServerBackend, "__init__", return_value=None):
                from infinity_parser2 import InfinityParser2
                parser = InfinityParser2(backend="vllm-server")

            assert "torch" not in sys.modules
            # The auto tensor-parallel-size default only applies to vllm-engine.
            assert parser.tensor_parallel_size is None, parser.tensor_parallel_size
            print("OK")
            """,
            self.HEAVY_DEPS,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)


class TestImageFallbackResize(unittest.TestCase):
    """encode_image_to_base64 must still work when qwen-vl-utils is absent."""

    def test_falls_back_to_basic_resize_without_qwen_vl_utils(self):
        result = _run_isolated(
            """
            from PIL import Image
            from infinity_parser2.utils.image import (
                encode_image_to_base64,
                smart_resize,
                _fallback_smart_resize,
            )

            assert smart_resize is _fallback_smart_resize

            encoded, mime_type = encode_image_to_base64(Image.new("RGB", (10, 10)))
            assert encoded
            assert mime_type == "image/jpeg"
            print("OK")
            """,
            {"qwen_vl_utils"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
