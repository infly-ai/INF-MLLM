"""Unit tests for InfinityParser2 backend classes."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from infinity_parser2.backends import (
    BaseBackend,
    TransformersBackend,
    VLLMEngineBackend,
    VLLMServerBackend,
)


class TestBaseBackend(unittest.TestCase):
    """Tests for BaseBackend abstract class."""

    def test_base_backend_is_abstract(self):
        """Test that BaseBackend cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseBackend()

    def test_base_backend_has_abstract_methods(self):
        """Test that BaseBackend has required abstract methods."""
        self.assertTrue(hasattr(BaseBackend, 'init'))
        self.assertTrue(hasattr(BaseBackend, 'parse_batch'))

    def test_base_backend_subclass_interface(self):
        """Test that subclasses implement required interface."""
        class ConcreteBackend(BaseBackend):
            def init(self):
                pass

            def parse_batch(self, input_data, prompt, batch_size=1, **kwargs):
                return []

        backend = ConcreteBackend(model_name="test/model", device="cuda")
        self.assertEqual(backend.model_name, "test/model")
        self.assertEqual(backend.device, "cuda")

    def test_base_backend_init_parameters(self):
        """Test BaseBackend initialization with parameters."""
        class ConcreteBackend(BaseBackend):
            def init(self):
                pass

            def parse_batch(self, input_data, prompt, batch_size=1, **kwargs):
                return []

        backend = ConcreteBackend(
            model_name="custom/model",
            device="cpu",
            custom_arg="value"
        )
        self.assertEqual(backend.model_name, "custom/model")
        self.assertEqual(backend.device, "cpu")
        self.assertEqual(backend.kwargs.get("custom_arg"), "value")


class TestTransformersBackend(unittest.TestCase):
    """Tests for TransformersBackend class."""

    def test_transformers_backend_initialization_params(self):
        """Test TransformersBackend initialization parameters."""
        with patch("infinity_parser2.backends.transformers.AutoModelForImageTextToText") as mock_model:
            with patch("infinity_parser2.backends.transformers.AutoProcessor") as mock_processor:
                mock_model.from_pretrained.return_value = MagicMock()
                mock_processor.from_pretrained.return_value = MagicMock()

                backend = TransformersBackend(
                    model_name="test/model",
                    device="cuda",
                    torch_dtype="float16",
                    min_pixels=1024,
                    max_pixels=4096,
                )

                self.assertEqual(backend.model_name, "test/model")
                self.assertEqual(backend.device, "cuda")
                self.assertEqual(backend.min_pixels, 1024)
                self.assertEqual(backend.max_pixels, 4096)

    def test_transformers_backend_min_max_pixels_defaults(self):
        """Test TransformersBackend default min_pixels and max_pixels."""
        with patch("infinity_parser2.backends.transformers.AutoModelForImageTextToText") as mock_model:
            with patch("infinity_parser2.backends.transformers.AutoProcessor") as mock_processor:
                mock_model.from_pretrained.return_value = MagicMock()
                mock_processor.from_pretrained.return_value = MagicMock()

                backend = TransformersBackend(model_name="test/model")
                self.assertEqual(backend.min_pixels, 2048)
                self.assertEqual(backend.max_pixels, 16777216)

    def test_transformers_backend_process_inputs(self):
        """Test _process_inputs method."""
        with patch("infinity_parser2.backends.transformers.AutoModelForImageTextToText") as mock_model:
            with patch("infinity_parser2.backends.transformers.AutoProcessor") as mock_processor:
                with patch("infinity_parser2.backends.transformers.process_vision_info") as mock_process_vision:
                    mock_model.from_pretrained.return_value = MagicMock()
                    mock_processor_instance = MagicMock()
                    mock_processor.from_pretrained.return_value = mock_processor_instance
                    mock_processor_instance.apply_chat_template.return_value = "processed"
                    mock_processor_instance.batch_decode.return_value = ["decoded"]
                    mock_processor_instance.return_value = {"input_ids": MagicMock()}
                    mock_process_vision.return_value = ([MagicMock()], None)

                    backend = TransformersBackend(model_name="test/model")

                    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    img = Image.new("RGB", (100, 100), color="red")
                    img.save(temp_file.name)
                    temp_file.close()

                    try:
                        result = backend._process_inputs(
                            [temp_file.name], "Test prompt"
                        )
                        self.assertIsInstance(result, dict)
                        self.assertIn("input_ids", result)
                    finally:
                        import os
                        os.unlink(temp_file.name)

    def test_transformers_backend_process_multiple_inputs(self):
        """Test processing multiple inputs."""
        with patch("infinity_parser2.backends.transformers.AutoModelForImageTextToText") as mock_model:
            with patch("infinity_parser2.backends.transformers.AutoProcessor") as mock_processor:
                with patch("infinity_parser2.backends.transformers.process_vision_info") as mock_process_vision:
                    mock_model.from_pretrained.return_value = MagicMock()
                    mock_processor_instance = MagicMock()
                    mock_processor.from_pretrained.return_value = mock_processor_instance
                    mock_processor_instance.apply_chat_template.return_value = "processed"
                    mock_processor_instance.batch_decode.return_value = ["decoded"]
                    mock_processor_instance.return_value = {"input_ids": MagicMock()}
                    mock_process_vision.return_value = ([MagicMock()], None)

                    backend = TransformersBackend(model_name="test/model")

                    temp_files = []
                    for i in range(3):
                        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        img = Image.new("RGB", (100, 100), color="blue")
                        img.save(f.name)
                        temp_files.append(f.name)

                    try:
                        result = backend._process_inputs(
                            temp_files, "Test prompt"
                        )
                        self.assertIsInstance(result, dict)
                        self.assertIn("input_ids", result)
                    finally:
                        import os
                        for f in temp_files:
                            os.unlink(f)

    def test_transformers_backend_generate_output_format(self):
        """Test _generate method output format."""
        with patch("infinity_parser2.backends.transformers.AutoModelForImageTextToText") as mock_model:
            with patch("infinity_parser2.backends.transformers.AutoProcessor") as mock_processor:
                mock_model_instance = MagicMock()
                mock_model.from_pretrained.return_value = mock_model_instance
                mock_model_instance.device = "cuda"

                mock_processor_instance = MagicMock()
                mock_processor.from_pretrained.return_value = mock_processor_instance
                mock_processor_instance.apply_chat_template.return_value = "processed text"
                mock_processor_instance.batch_decode.return_value = ["Generated text"]

                backend = TransformersBackend(model_name="test/model")
                backend._model = mock_model_instance

                with patch.object(backend._processor, '__call__') as mock_call:
                    mock_input_ids = MagicMock()
                    mock_call.return_value = {
                        "input_ids": mock_input_ids
                    }

                    mock_output_ids = MagicMock()
                    mock_model_instance.generate.return_value = [mock_output_ids]
                    mock_output_ids.__getitem__ = MagicMock(return_value=[1, 2, 3])

                    results = backend._generate({"input_ids": mock_input_ids})
                    self.assertIsInstance(results, list)

    def test_transformers_backend_parse_batch(self):
        """Test parse_batch basic functionality."""
        with patch("infinity_parser2.backends.transformers.AutoModelForImageTextToText") as mock_model:
            with patch("infinity_parser2.backends.transformers.AutoProcessor") as mock_processor:
                with patch("infinity_parser2.backends.transformers.process_vision_info") as mock_process_vision:
                    mock_model_instance = MagicMock()
                    mock_model.from_pretrained.return_value = mock_model_instance
                    mock_model_instance.device = "cuda"

                    mock_processor_instance = MagicMock()
                    mock_processor.from_pretrained.return_value = mock_processor_instance
                    mock_processor_instance.apply_chat_template.return_value = "processed"
                    mock_processor_instance.batch_decode.return_value = ["Result"]
                    mock_processor_instance.return_value = {"input_ids": MagicMock()}
                    mock_process_vision.return_value = ([MagicMock()], None)

                    backend = TransformersBackend(model_name="test/model")
                    backend._model = mock_model_instance

                    with patch.object(backend._processor, '__call__') as mock_call:
                        mock_input_ids = MagicMock()
                        mock_call.return_value = {
                            "input_ids": mock_input_ids
                        }

                        mock_output_ids = MagicMock()
                        mock_model_instance.generate.return_value = [mock_output_ids]
                        mock_output_ids.__getitem__ = MagicMock(return_value=[1, 2, 3])

                        results = backend.parse_batch(
                            [Image.new("RGB", (100, 100))],
                            "Test prompt"
                        )
                        self.assertIsInstance(results, list)
                        self.assertEqual(len(results), 1)


class TestVLLMEngineBackend(unittest.TestCase):
    """Tests for VLLMEngineBackend class."""

    def test_vllm_engine_backend_initialization(self):
        """Test VLLMEngineBackend initialization parameters."""
        with patch("infinity_parser2.backends.vllm_engine.LLM") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance

            backend = VLLMEngineBackend(
                model_name="test/model",
                device="cuda",
                tensor_parallel_size=2,
                min_pixels=1024,
                max_pixels=4096,
            )

            self.assertEqual(backend.model_name, "test/model")
            self.assertEqual(backend.device, "cuda")
            self.assertEqual(backend.tensor_parallel_size, 2)
            self.assertEqual(backend.min_pixels, 1024)
            self.assertEqual(backend.max_pixels, 4096)

    def test_vllm_engine_backend_min_max_pixels_defaults(self):
        """Test VLLMEngineBackend default min_pixels and max_pixels."""
        with patch("infinity_parser2.backends.vllm_engine.LLM") as mock_llm:
            mock_llm.return_value = MagicMock()

            backend = VLLMEngineBackend(model_name="test/model")
            self.assertEqual(backend.min_pixels, 2048)
            self.assertEqual(backend.max_pixels, 16777216)

    def test_vllm_engine_build_messages(self):
        """Test _build_messages method."""
        with patch("infinity_parser2.backends.vllm_engine.LLM") as mock_llm:
            mock_llm.return_value = MagicMock()

            backend = VLLMEngineBackend(model_name="test/model")
            messages = backend._build_messages("base64data", "image/png", "Test prompt")

            self.assertIsInstance(messages, list)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["role"], "user")
            self.assertIsInstance(messages[0]["content"], list)
            self.assertEqual(len(messages[0]["content"]), 2)

            image_content = messages[0]["content"][0]
            self.assertEqual(image_content["type"], "image_url")
            self.assertIn("data:image/png;base64,base64data", image_content["image_url"]["url"])

            text_content = messages[0]["content"][1]
            self.assertEqual(text_content["type"], "text")
            self.assertEqual(text_content["text"], "Test prompt")

    def test_vllm_engine_parse_batch_returns_list(self):
        """Test parse_batch returns a list."""
        with patch("infinity_parser2.backends.vllm_engine.LLM") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance

            mock_output = MagicMock()
            mock_output.outputs = [MagicMock(text="Parsed result")]
            mock_llm_instance.chat.return_value = [mock_output]

            backend = VLLMEngineBackend(model_name="test/model")

            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img = Image.new("RGB", (100, 100), color="green")
            img.save(temp_file.name)
            temp_file.close()

            try:
                results = backend.parse_batch([temp_file.name], "Test prompt")
                self.assertIsInstance(results, list)
                self.assertEqual(len(results), 1)
            finally:
                import os
                os.unlink(temp_file.name)

    def test_vllm_engine_parse_batch(self):
        """Test parse_batch basic functionality."""
        with patch("infinity_parser2.backends.vllm_engine.LLM") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance

            mock_output = MagicMock()
            mock_output.outputs = [MagicMock(text="Parsed result")]
            mock_llm_instance.chat.return_value = [mock_output]

            backend = VLLMEngineBackend(model_name="test/model")

            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img = Image.new("RGB", (100, 100), color="green")
            img.save(temp_file.name)
            temp_file.close()

            try:
                results = backend.parse_batch(
                    [temp_file.name],
                    "Test prompt"
                )
                self.assertIsInstance(results, list)
                self.assertEqual(len(results), 1)
                mock_llm_instance.chat.assert_called()
            finally:
                import os
                os.unlink(temp_file.name)


class TestVLLMServerBackend(unittest.TestCase):
    """Tests for VLLMServerBackend class."""

    def test_vllm_server_backend_initialization(self):
        """Test VLLMServerBackend initialization parameters."""
        with patch("infinity_parser2.backends.vllm_server.OpenAI") as mock_openai:
            mock_client_instance = MagicMock()
            mock_openai.return_value = mock_client_instance

            backend = VLLMServerBackend(
                model_name="test/model",
                api_url="http://localhost:8000/v1/chat/completions",
                api_key="test-key",
                timeout=60,
                min_pixels=1024,
                max_pixels=4096,
            )

            self.assertEqual(backend.model_name, "test/model")
            self.assertEqual(backend.api_url, "http://localhost:8000/v1/chat/completions")
            self.assertEqual(backend.api_key, "test-key")
            self.assertEqual(backend.timeout, 60)
            self.assertEqual(backend.min_pixels, 1024)
            self.assertEqual(backend.max_pixels, 4096)
            self.assertIsNotNone(backend.client)

    def test_vllm_server_backend_min_max_pixels_defaults(self):
        """Test VLLMServerBackend default min_pixels and max_pixels."""
        with patch("infinity_parser2.backends.vllm_server.OpenAI") as mock_openai:
            mock_client_instance = MagicMock()
            mock_openai.return_value = mock_client_instance

            backend = VLLMServerBackend(api_url="http://localhost:8000/v1/chat/completions")
            self.assertEqual(backend.min_pixels, 2048)
            self.assertEqual(backend.max_pixels, 16777216)

    def test_vllm_server_connection_check(self):
        """Test server connection validation on init."""
        with patch("infinity_parser2.backends.vllm_server.OpenAI") as mock_openai:
            mock_client_instance = MagicMock()
            mock_openai.return_value = mock_client_instance

            backend = VLLMServerBackend(api_url="http://localhost:8000/v1/chat/completions")
            mock_client_instance.chat.completions.create.assert_called_once()

    def test_vllm_server_connection_failure(self):
        """Test RuntimeError on connection failure."""
        with patch("infinity_parser2.backends.vllm_server.OpenAI") as mock_openai:
            mock_client_instance = MagicMock()
            mock_client_instance.chat.completions.create.side_effect = Exception("Connection refused")
            mock_openai.return_value = mock_client_instance

            with self.assertRaises(RuntimeError) as context:
                VLLMServerBackend(api_url="http://localhost:8000/v1/chat/completions")

            self.assertIn("Cannot connect to vLLM server", str(context.exception))

    def test_vllm_server_parse_batch_empty_input(self):
        """Test parse_batch with empty input returns empty list."""
        with patch("infinity_parser2.backends.vllm_server.OpenAI") as mock_openai:
            mock_client_instance = MagicMock()
            mock_openai.return_value = mock_client_instance

            backend = VLLMServerBackend(api_url="http://localhost:8000/v1/chat/completions")
            results = backend.parse_batch([], "Test prompt")
            self.assertEqual(results, [])

    def test_vllm_server_parse_batch_success(self):
        """Test successful parse_batch call."""
        with patch("infinity_parser2.backends.vllm_server.OpenAI") as mock_openai:
            mock_client_instance = MagicMock()
            mock_openai.return_value = mock_client_instance

            mock_chat_response = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Parsed content"
            mock_chat_response.choices = [MagicMock(message=mock_message)]
            mock_client_instance.chat.completions.create.return_value = mock_chat_response

            backend = VLLMServerBackend(api_url="http://localhost:8000/v1/chat/completions")

            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img = Image.new("RGB", (100, 100), color="yellow")
            img.save(temp_file.name)
            temp_file.close()

            try:
                results = backend.parse_batch([temp_file.name], "Test prompt")
                self.assertIsInstance(results, list)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0], "Parsed content")
            finally:
                import os
                os.unlink(temp_file.name)

    def test_vllm_server_extra_body(self):
        """Test that OpenAI client is called with correct parameters."""
        with patch("infinity_parser2.backends.vllm_server.OpenAI") as mock_openai:
            mock_client_instance = MagicMock()
            mock_openai.return_value = mock_client_instance

            mock_chat_response = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Result"
            mock_chat_response.choices = [MagicMock(message=mock_message)]
            mock_client_instance.chat.completions.create.return_value = mock_chat_response

            backend = VLLMServerBackend(
                api_url="http://localhost:8000/v1/chat/completions",
                api_key="my-secret-key"
            )

            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img = Image.new("RGB", (100, 100), color="cyan")
            img.save(temp_file.name)
            temp_file.close()

            try:
                backend.parse_batch([temp_file.name], "Test prompt")
                call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
                self.assertEqual(call_kwargs["model"], "infly/Infinity-Parser2-Pro")
                self.assertIn("messages", call_kwargs)
                self.assertEqual(call_kwargs["max_tokens"], 32768)
                self.assertEqual(call_kwargs["temperature"], 0.0)
                self.assertEqual(call_kwargs["top_p"], 1.0)
                self.assertEqual(
                    call_kwargs["extra_body"],
                    {"chat_template_kwargs": {"enable_thinking": False}}
                )
            finally:
                import os
                os.unlink(temp_file.name)


class TestBackendRegistry(unittest.TestCase):
    """Tests for backend registry mapping."""

    def test_backend_registry_keys(self):
        """Test that BACKEND_REGISTRY contains expected keys."""
        from infinity_parser2.backends import TransformersBackend, VLLMEngineBackend, VLLMServerBackend
        from infinity_parser2.parser import BACKEND_REGISTRY

        self.assertIn("transformers", BACKEND_REGISTRY)
        self.assertIn("vllm-engine", BACKEND_REGISTRY)
        self.assertIn("vllm-server", BACKEND_REGISTRY)

    def test_backend_registry_values(self):
        """Test that BACKEND_REGISTRY contains correct backend classes."""
        from infinity_parser2.parser import BACKEND_REGISTRY

        self.assertEqual(BACKEND_REGISTRY["transformers"], TransformersBackend)
        self.assertEqual(BACKEND_REGISTRY["vllm-engine"], VLLMEngineBackend)
        self.assertEqual(BACKEND_REGISTRY["vllm-server"], VLLMServerBackend)


if __name__ == "__main__":
    unittest.main()
