import io
import os
import unittest
import urllib.error
from unittest.mock import patch

from icml_ai_ac.scoring.providers import (
    ChatCompletionClient,
    ChatResponseContentError,
    effective_openrouter_reasoning_effort,
    extract_content,
    is_batch_blocking_provider_error,
    post_json_with_retries,
    redact_provider_error_detail,
)


class ProviderTests(unittest.TestCase):
    def test_effective_reasoning_is_model_specific(self) -> None:
        self.assertEqual(
            effective_openrouter_reasoning_effort(
                model="google/gemini-3.5-flash-lite",
                requested_effort="none",
            ),
            "minimal",
        )
        self.assertEqual(
            effective_openrouter_reasoning_effort(
                model="qwen/qwen3.7-plus",
                requested_effort="none",
            ),
            "none",
        )
        self.assertEqual(
            effective_openrouter_reasoning_effort(
                model="google/gemini-3.5-flash-lite",
                requested_effort="",
            ),
            "minimal",
        )

    def test_batch_blocking_provider_errors_cover_auth_and_rate_limits(self) -> None:
        for status in (400, 401, 402, 403, 404, 422, 429):
            self.assertTrue(is_batch_blocking_provider_error(RuntimeError(f"HTTP {status} from provider")))
        self.assertTrue(is_batch_blocking_provider_error(RuntimeError("HTTP Error 429: rate limited")))
        self.assertFalse(is_batch_blocking_provider_error(TimeoutError("request timed out")))

    def test_content_error_retains_usage_without_dumping_response(self) -> None:
        response = {"model": "served", "usage": {"cost": 0.25}, "secret": "not in message"}
        error = ChatResponseContentError("missing text", response=response)

        self.assertEqual(str(error), "missing text")
        self.assertEqual(error.usage, {"cost": 0.25})
        self.assertEqual(error.served_model, "served")

    def test_extract_content_accepts_text_blocks(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "{\"status\":"},
                            {"type": "text", "text": "\"ok\"}"},
                        ]
                    }
                }
            ]
        }

        self.assertEqual(extract_content(response), '{"status":\n"ok"}')

    def test_extract_content_identifies_message_refusal(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "refusal": "I cannot help with that request.",
                    }
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "refusal"):
            extract_content(response)

    def test_openrouter_structured_calls_disable_returned_reasoning_by_default(self) -> None:
        captured: dict[str, object] = {}

        def fake_post_json_with_retries(url, payload, *, headers, timeout_seconds, retries, backoff_seconds):
            captured.update(payload)
            return {
                "model": "provider/served-model",
                "choices": [{"message": {"content": "{}"}}],
                "usage": {},
            }

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            with patch("icml_ai_ac.scoring.providers.post_json_with_retries", fake_post_json_with_retries):
                client = ChatCompletionClient(provider="openrouter", model="qwen/qwen3.6-35b-a3b")
                result = client.complete(
                    messages=[{"role": "user", "content": "Return JSON."}],
                    temperature=0.1,
                    max_output_tokens=100,
                    response_format={"type": "json_object"},
                )

        self.assertEqual(captured["reasoning"], {"effort": "none", "exclude": True})
        self.assertEqual(result.served_model, "provider/served-model")

    def test_gemini_35_flash_lite_uses_minimal_reasoning(self) -> None:
        captured: dict[str, object] = {}

        def fake_post_json_with_retries(url, payload, *, headers, timeout_seconds, retries, backoff_seconds):
            captured.update(payload)
            return {
                "model": "google/gemini-3.5-flash-lite",
                "choices": [{"message": {"content": "{}"}}],
                "usage": {},
            }

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            with patch("icml_ai_ac.scoring.providers.post_json_with_retries", fake_post_json_with_retries):
                client = ChatCompletionClient(
                    provider="openrouter",
                    model="google/gemini-3.5-flash-lite",
                    reasoning_effort="none",
                )
                client.complete(
                    messages=[{"role": "user", "content": "Return JSON."}],
                    temperature=0.0,
                    max_output_tokens=100,
                    response_format={"type": "json_object"},
                )

        self.assertEqual(captured["reasoning"], {"effort": "minimal", "exclude": True})

    def test_openrouter_provider_preferences_are_sent(self) -> None:
        captured: dict[str, object] = {}

        def fake_post_json_with_retries(url, payload, *, headers, timeout_seconds, retries, backoff_seconds):
            captured.update(payload)
            return {
                "model": "anthropic/claude-sonnet-5",
                "choices": [{"message": {"content": "{}"}}],
                "usage": {},
            }

        preferences = {"only": ["anthropic"], "allow_fallbacks": False}
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            with patch("icml_ai_ac.scoring.providers.post_json_with_retries", fake_post_json_with_retries):
                client = ChatCompletionClient(
                    provider="openrouter",
                    model="anthropic/claude-sonnet-5",
                    openrouter_provider_preferences=preferences,
                )
                result = client.complete(
                    messages=[{"role": "user", "content": "Return JSON."}],
                    temperature=0.0,
                    max_output_tokens=100,
                    response_format={"type": "json_object"},
                )

        self.assertEqual(captured["provider"], preferences)
        self.assertEqual(result.request["provider"], preferences)

    def test_provider_error_detail_redacts_account_identifiers(self) -> None:
        detail = '{"error":{"message":"bad request","metadata":{"user_id":"owner-123"}}}'

        redacted = redact_provider_error_detail(detail)

        self.assertNotIn("owner-123", redacted)
        self.assertIn('"user_id":"[redacted]"', redacted)

    def test_http_error_persistence_uses_redacted_detail(self) -> None:
        error = urllib.error.HTTPError(
            "https://example.test",
            400,
            "bad request",
            {},
            io.BytesIO(b'{"user_id":"owner-123","message":"bad"}'),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, r'"user_id":"\[redacted\]"') as caught:
                post_json_with_retries(
                    "https://example.test",
                    {"x": 1},
                    headers={},
                    timeout_seconds=1,
                    retries=0,
                    backoff_seconds=0,
                )

        self.assertNotIn("owner-123", str(caught.exception))

    def test_post_json_retries_invalid_json_response(self) -> None:
        calls = {"count": 0}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                calls["count"] += 1
                if calls["count"] == 1:
                    return b"<html>temporary provider edge error</html>"
                return b'{"ok": true}'

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = post_json_with_retries(
                "https://example.test",
                {"x": 1},
                headers={},
                timeout_seconds=1,
                retries=1,
                backoff_seconds=0,
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls["count"], 2)

    def test_openai_reasoning_uses_responses_file_input(self) -> None:
        captured: dict[str, object] = {}

        def fake_post_json_with_retries(url, payload, *, headers, timeout_seconds, retries, backoff_seconds):
            captured["url"] = url
            captured.update(payload)
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "{}"}],
                    }
                ],
                "usage": {},
            }

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            with patch("icml_ai_ac.scoring.providers.post_json_with_retries", fake_post_json_with_retries):
                client = ChatCompletionClient(provider="openai", model="gpt-5.5", reasoning_effort="xhigh")
                client.complete(
                    messages=[
                        {"role": "system", "content": "Return JSON."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Read this."},
                                {
                                    "type": "file",
                                    "file": {
                                        "filename": "paper.pdf",
                                        "file_data": "data:application/pdf;base64,abc",
                                    },
                                },
                            ],
                        },
                    ],
                    temperature=0.0,
                    max_output_tokens=100,
                    seed=20260725,
                    response_format={"type": "json_object"},
                )

        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(captured["reasoning"], {"effort": "xhigh"})
        self.assertNotIn("seed", captured)
        self.assertEqual(captured["text"], {"format": {"type": "json_object"}})
        input_items = captured["input"]
        self.assertIsInstance(input_items, list)
        content = input_items[0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "Read this."})
        self.assertEqual(content[1]["type"], "input_file")


if __name__ == "__main__":
    unittest.main()
