from __future__ import annotations

import http.client
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_FILES_URL = "https://api.openai.com/v1/files"
OPENROUTER_MODELS_REQUIRING_MINIMAL_REASONING = frozenset(
    {
        "google/gemini-3.5-flash-lite",
    }
)


@dataclass(slots=True)
class ChatResult:
    provider: str
    model: str
    served_model: str | None
    request: dict[str, Any]
    response: dict[str, Any]
    content: str
    usage: dict[str, Any]
    elapsed_seconds: float


class ChatResponseContentError(ValueError):
    """A provider returned a response object without usable assistant text."""

    def __init__(self, message: str, *, response: dict[str, Any]) -> None:
        super().__init__(message)
        self.response = response
        self.usage = response.get("usage", {}) if isinstance(response.get("usage"), dict) else {}
        self.served_model = response_model(response)


def effective_openrouter_reasoning_effort(
    *,
    model: str,
    requested_effort: str | None,
) -> str | None:
    effort = requested_effort
    if effort is None:
        effort = os.environ.get("OPENROUTER_REASONING_EFFORT", "none")
    effort = effort.strip().lower() or "none"
    if effort == "none" and model in OPENROUTER_MODELS_REQUIRING_MINIMAL_REASONING:
        return "minimal"
    return effort or None


class ChatCompletionClient:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        reasoning_effort: str | None = None,
        openrouter_pdf_engine: str | None = None,
        openrouter_provider_preferences: dict[str, Any] | None = None,
        timeout_seconds: float = 120.0,
        retries: int = 3,
        backoff_seconds: float = 5.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.openrouter_pdf_engine = openrouter_pdf_engine
        self.openrouter_provider_preferences = (
            dict(openrouter_provider_preferences) if openrouter_provider_preferences else None
        )
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_output_tokens: int,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        if self.provider == "openai" and self.reasoning_effort:
            return self.complete_openai_responses(
                messages=messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                seed=seed,
                response_format=response_format,
            )
        api_key = self.api_key()
        url = self.url()
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_output_tokens,
        }
        if self.provider == "openrouter":
            request_body.pop("max_completion_tokens")
            request_body["max_tokens"] = max_output_tokens
            if self.openrouter_provider_preferences:
                request_body["provider"] = dict(self.openrouter_provider_preferences)
        if seed is not None:
            request_body["seed"] = seed
        if response_format is not None:
            request_body["response_format"] = response_format
        if self.provider == "openrouter":
            reasoning_effort = effective_openrouter_reasoning_effort(
                model=self.model,
                requested_effort=self.reasoning_effort,
            )
            if reasoning_effort:
                request_body["reasoning"] = {"effort": reasoning_effort, "exclude": True}
            if self.openrouter_pdf_engine:
                request_body["plugins"] = [
                    {
                        "id": "file-parser",
                        "pdf": {"engine": self.openrouter_pdf_engine},
                    }
                ]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["X-Title"] = "ICML AI Executive AC Study"
        started = time.monotonic()
        response = post_json_with_retries(
            url,
            request_body,
            headers=headers,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            backoff_seconds=self.backoff_seconds,
        )
        try:
            content = extract_content(response)
        except ValueError as exc:
            raise ChatResponseContentError(str(exc), response=response) from exc
        return ChatResult(
            provider=self.provider,
            model=self.model,
            served_model=response_model(response),
            request=request_body,
            response=response,
            content=content,
            usage=response.get("usage", {}) if isinstance(response.get("usage"), dict) else {},
            elapsed_seconds=time.monotonic() - started,
        )

    def complete_openai_responses(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_output_tokens: int,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        api_key = self.api_key()
        instructions, input_items = convert_messages_to_openai_responses(
            messages,
            api_key=api_key,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            backoff_seconds=self.backoff_seconds,
        )
        request_body: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": self.reasoning_effort},
        }
        if instructions:
            request_body["instructions"] = instructions
        if response_format is not None:
            request_body["text"] = {"format": response_format}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        started = time.monotonic()
        response = post_json_with_retries(
            OPENAI_RESPONSES_URL,
            request_body,
            headers=headers,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            backoff_seconds=self.backoff_seconds,
        )
        try:
            content = extract_responses_output_text(response)
        except ValueError as exc:
            raise ChatResponseContentError(str(exc), response=response) from exc
        return ChatResult(
            provider=self.provider,
            model=self.model,
            served_model=response_model(response),
            request=request_body,
            response=response,
            content=content,
            usage=response.get("usage", {}) if isinstance(response.get("usage"), dict) else {},
            elapsed_seconds=time.monotonic() - started,
        )

    def api_key(self) -> str:
        if self.provider == "openrouter":
            name = "OPENROUTER_API_KEY"
        elif self.provider == "openai":
            name = "OPENAI_API_KEY"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"{name} is required for provider={self.provider}")
        return value

    def url(self) -> str:
        if self.provider == "openrouter":
            return OPENROUTER_URL
        if self.provider == "openai":
            return OPENAI_CHAT_URL
        raise ValueError(f"Unsupported provider: {self.provider}")


def is_batch_blocking_provider_error(exc: Exception) -> bool:
    """Return whether a request error should stop the remaining batch suite."""
    blocking_statuses = {400, 401, 402, 403, 404, 422, 429}
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in blocking_statuses
    message = str(exc).lower()
    return any(
        f"http {status} " in message or f"http error {status}" in message
        for status in blocking_statuses
    )


def post_json_with_retries(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                data = response.read().decode("utf-8")
                parsed = json.loads(data)
                if not isinstance(parsed, dict):
                    raise ValueError("chat completion response must be a JSON object")
                return parsed
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            json.JSONDecodeError,
            ssl.SSLError,
            ConnectionError,
        ) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError):
                if exc.code == 429 and attempt < retries:
                    time.sleep(parse_retry_after(exc.headers.get("retry-after")) or backoff_seconds * (2**attempt))
                    continue
                if exc.code < 500:
                    detail = redact_provider_error_detail(exc.read().decode("utf-8", errors="replace"))
                    raise RuntimeError(f"HTTP {exc.code} from provider: {detail}") from exc
            if attempt < retries:
                time.sleep(backoff_seconds * (2**attempt))
    if last_error is None:
        raise RuntimeError("provider request failed without an exception")
    raise last_error


def post_multipart_with_retries(
    url: str,
    *,
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
    file_content_type: str,
    headers: dict[str, str],
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        boundary = f"----icml-ai-ac-{uuid.uuid4().hex}"
        body = build_multipart_body(
            boundary=boundary,
            fields=fields,
            file_field=file_field,
            file_path=file_path,
            file_content_type=file_content_type,
        )
        request_headers = {
            **headers,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                data = response.read().decode("utf-8")
                parsed = json.loads(data)
                if not isinstance(parsed, dict):
                    raise ValueError("multipart response must be a JSON object")
                return parsed
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            json.JSONDecodeError,
            ssl.SSLError,
            ConnectionError,
        ) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError):
                if exc.code == 429 and attempt < retries:
                    time.sleep(parse_retry_after(exc.headers.get("retry-after")) or backoff_seconds * (2**attempt))
                    continue
                if exc.code < 500:
                    detail = redact_provider_error_detail(exc.read().decode("utf-8", errors="replace"))
                    raise RuntimeError(f"HTTP {exc.code} from provider: {detail}") from exc
            if attempt < retries:
                time.sleep(backoff_seconds * (2**attempt))
    if last_error is None:
        raise RuntimeError("multipart provider request failed without an exception")
    raise last_error


def build_multipart_body(
    *,
    boundary: str,
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
    file_content_type: str,
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {file_content_type}\r\n\r\n".encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks)


def response_model(response: dict[str, Any]) -> str | None:
    value = response.get("model")
    return str(value) if value else None


def redact_provider_error_detail(detail: str) -> str:
    """Remove account-scoped identifiers before errors are persisted."""
    sensitive_keys = {"api_key", "authorization", "token", "user", "user_id"}
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        redacted = detail
        for key in sensitive_keys:
            pattern = rf'("{re.escape(key)}"\s*:\s*")[^"]*(")'
            redacted = re.sub(pattern, rf'\1[redacted]\2', redacted, flags=re.IGNORECASE)
        return redacted

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "[redacted]" if key.lower() in sensitive_keys else scrub(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return json.dumps(scrub(payload), ensure_ascii=True, separators=(",", ":"))


def upload_openai_file(
    *,
    api_key: str,
    path: Path,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> str:
    response = post_multipart_with_retries(
        OPENAI_FILES_URL,
        fields={"purpose": "user_data"},
        file_field="file",
        file_path=path,
        file_content_type="application/pdf",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout_seconds=timeout_seconds,
        retries=retries,
        backoff_seconds=backoff_seconds,
    )
    file_id = response.get("id")
    if not isinstance(file_id, str) or not file_id:
        raise ValueError("OpenAI file upload response missing id")
    return file_id


def extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("chat response choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("chat response choice has no message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    content_blocks = [content] if isinstance(content, dict) else content
    if isinstance(content_blocks, list):
        parts: list[str] = []
        refusals: list[str] = []
        for part in content_blocks:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(text, dict) and isinstance(text.get("value"), str):
                parts.append(text["value"])
            refusal = part.get("refusal")
            if isinstance(refusal, str):
                refusals.append(refusal)
        if parts:
            return "\n".join(parts)
        if refusals:
            raise ValueError("chat response contained a refusal instead of output text")
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal:
        raise ValueError("chat response contained a refusal instead of output text")
    raise ValueError("chat response message content must be text or text blocks")


def extract_responses_output_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    output = response.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    raise ValueError("responses API response has no output text")


def convert_messages_to_openai_responses(
    messages: list[dict[str, Any]],
    *,
    api_key: str,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if role == "system":
            text = content if isinstance(content, str) else content_parts_to_text(content)
            if text:
                instructions.append(text)
            continue
        input_items.append(
            {
                "role": role if role in {"user", "assistant", "developer"} else "user",
                "content": convert_content_to_openai_responses(
                    content,
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    backoff_seconds=backoff_seconds,
                ),
            }
        )
    return "\n\n".join(instructions), input_items


def convert_content_to_openai_responses(
    content: Any,
    *,
    api_key: str,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "input_text", "text": str(content)}]
    converted: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            converted.append({"type": "input_text", "text": str(part)})
            continue
        part_type = part.get("type")
        if part_type == "text":
            converted.append({"type": "input_text", "text": str(part.get("text") or "")})
        elif part_type == "file":
            file_payload = part.get("file") if isinstance(part.get("file"), dict) else {}
            file_path = file_payload.get("path")
            if isinstance(file_path, str) and file_path:
                file_id = upload_openai_file(
                    api_key=api_key,
                    path=Path(file_path),
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    backoff_seconds=backoff_seconds,
                )
                converted.append({"type": "input_file", "file_id": file_id})
                continue
            converted.append(
                {
                    "type": "input_file",
                    "filename": str(file_payload.get("filename") or "input.pdf"),
                    "file_data": str(file_payload.get("file_data") or ""),
                }
            )
        else:
            converted.append({"type": "input_text", "text": json.dumps(part, ensure_ascii=False)})
    return converted


def content_parts_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None
