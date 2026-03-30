import tempfile
import unittest
from requests.exceptions import SSLError
from unittest.mock import patch

from model_provider_utils import (
    RetryingEmbedding,
    create_multimodal_llm,
    create_text_llm,
    should_use_dashscope_embedding,
    should_use_dashscope_llm,
)


class ModelProviderUtilsTests(unittest.TestCase):
    class _FakeDelta:
        def __init__(self, content=None, reasoning_content=None):
            self.content = content
            self.reasoning_content = reasoning_content

    class _FakeChoice:
        def __init__(self, content=None, reasoning_content=None):
            self.delta = ModelProviderUtilsTests._FakeDelta(
                content=content,
                reasoning_content=reasoning_content,
            )

    class _FakeChunk:
        def __init__(self, content=None, reasoning_content=None):
            self.choices = [ModelProviderUtilsTests._FakeChoice(content=content, reasoning_content=reasoning_content)]

    def test_retrying_embedding_retries_transient_ssl_failures(self):
        class _FlakyDelegate:
            model_name = "text-embedding-v3"
            embed_batch_size = 10
            callback_manager = None
            num_workers = None
            embeddings_cache = None
            rate_limiter = None

            def __init__(self):
                self.query_calls = 0

            def _get_query_embedding(self, query):
                self.query_calls += 1
                if self.query_calls == 1:
                    raise SSLError("UNEXPECTED_EOF_WHILE_READING")
                return [0.1, 0.2, 0.3]

            def _get_text_embedding(self, text):
                return [0.4, 0.5, 0.6]

            async def _aget_query_embedding(self, query):
                return self._get_query_embedding(query)

        delegate = _FlakyDelegate()
        wrapper = RetryingEmbedding(delegate=delegate, max_retries=2, retry_delay_seconds=0.0)

        embedding = wrapper.get_query_embedding("MRB trigger criteria")

        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        self.assertEqual(delegate.query_calls, 2)

    def test_retrying_embedding_does_not_retry_non_transient_errors(self):
        class _BadDelegate:
            model_name = "text-embedding-v3"
            embed_batch_size = 10
            callback_manager = None
            num_workers = None
            embeddings_cache = None
            rate_limiter = None

            def _get_query_embedding(self, query):
                raise ValueError("bad request")

            def _get_text_embedding(self, text):
                return [0.4, 0.5, 0.6]

            async def _aget_query_embedding(self, query):
                return self._get_query_embedding(query)

        wrapper = RetryingEmbedding(delegate=_BadDelegate(), max_retries=2, retry_delay_seconds=0.0)

        with self.assertRaises(ValueError):
            wrapper.get_query_embedding("MRB trigger criteria")

    def test_detects_dashscope_from_base_url(self):
        self.assertTrue(
            should_use_dashscope_embedding(
                model_name="text-embedding-3-small",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        )
        self.assertTrue(
            should_use_dashscope_llm(
                model_name="gpt-4o-mini",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        )

    def test_detects_dashscope_from_model_name(self):
        self.assertTrue(
            should_use_dashscope_embedding(
                model_name="text-embedding-v3",
                api_base="https://api.openai.com/v1",
            )
        )
        self.assertTrue(
            should_use_dashscope_llm(
                model_name="qwen3.5-plus",
                api_base="https://api.openai.com/v1",
            )
        )

    def test_keeps_openai_for_openai_models(self):
        self.assertFalse(
            should_use_dashscope_embedding(
                model_name="text-embedding-3-small",
                api_base="https://api.openai.com/v1",
            )
        )
        self.assertFalse(
            should_use_dashscope_llm(
                model_name="gpt-4o-mini",
                api_base="https://api.openai.com/v1",
            )
        )

    def test_dashscope_text_llm_uses_openai_compatible_wrapper(self):
        llm = create_text_llm(
            model_name="qwen3.5-plus",
            api_key="dummy",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(type(llm).__name__, "OpenAICompatibleLLM")

    def test_dashscope_text_llm_can_disable_thinking(self):
        llm = create_text_llm(
            model_name="qwen3.5-plus",
            api_key="dummy",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            disable_thinking=True,
        )

        self.assertEqual(
            llm.additional_request_kwargs,
            {"extra_body": {"enable_thinking": False}},
        )

    def test_stream_complete_preserves_reasoning_deltas(self):
        class _FakeOpenAIClient:
            def __init__(self, *args, **kwargs):
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                return iter(
                    [
                        ModelProviderUtilsTests._FakeChunk(reasoning_content="Thinking"),
                        ModelProviderUtilsTests._FakeChunk(content="答案"),
                    ]
                )

        with patch("model_provider_utils.OpenAIClient", _FakeOpenAIClient):
            llm = create_text_llm(
                model_name="qwen3.5-plus",
                api_key="dummy",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                disable_thinking=False,
            )

        chunks = list(llm.stream_complete("test"))

        self.assertEqual(chunks[0].additional_kwargs["reasoning_delta"], "Thinking")
        self.assertIsNone(chunks[0].delta)
        self.assertEqual(chunks[1].delta, "答案")

    def test_multimodal_stream_complete_embeds_local_images(self):
        captured_kwargs = {}

        class _FakeOpenAIClient:
            def __init__(self, *args, **kwargs):
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                captured_kwargs.update(kwargs)
                return iter([ModelProviderUtilsTests._FakeChunk(content="图像回答")])

        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            image_file.write(b"fake-image")
            image_file.flush()

            with patch("model_provider_utils.OpenAIClient", _FakeOpenAIClient):
                llm = create_multimodal_llm(
                    model_name="gpt-4o",
                    api_key="dummy",
                    api_base="https://api.openai.com/v1",
                )

                chunks = list(llm.stream_complete("请解释这张图", image_paths=[image_file.name]))

        self.assertEqual(chunks[0].delta, "图像回答")
        self.assertEqual(captured_kwargs["messages"][0]["role"], "user")
        self.assertEqual(captured_kwargs["messages"][0]["content"][0]["type"], "text")
        self.assertEqual(captured_kwargs["messages"][0]["content"][1]["type"], "image_url")
        self.assertTrue(
            captured_kwargs["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
        )


if __name__ == "__main__":
    unittest.main()
