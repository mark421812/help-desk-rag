"""Groq Chat Completions adapter（OpenAI 相容 API）。

非串流呼叫，一律使用 strict `json_schema` 的 response_format。並行數與等待佇列都有上限，佇列滿
或等太久直接回報 rate limit，不形成無界佇列。429、498 與 5xx 依 `Retry-After` 或指數退避重試，
總嘗試次數有上限，而且不會超過總時限。

Groq 的 strict 模式仍會偶發 400 `json_validate_failed`，這種
錯誤另外重試一次，而且只重試一次。錯誤回應可能回顯 prompt 或模型輸出，所以只取 allowlist 內
的錯誤代碼，其他內容都不離開這個模組。
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Final

import httpx
import structlog

from helpdesk.application.llm import (
    LlmContractError,
    LlmRateLimitedError,
    LlmRequest,
    LlmResponse,
    LlmTimeoutError,
    LlmUnavailableError,
    TokenUsage,
)

logger = structlog.get_logger(__name__)

KNOWN_PROVIDER_ERROR_CODES: Final = frozenset(
    {
        "context_length_exceeded",
        "invalid_request_error",
        "json_validate_failed",
        "model_decommissioned",
        "model_not_found",
        "output_parse_failed",
        "rate_limit_exceeded",
        "request_too_large",
        "tool_use_failed",
    }
)
SCHEMA_RETRY_CODE: Final = "json_validate_failed"
# Groq 文件把 498 定義為 flex tier 容量不足。
RETRYABLE_STATUSES: Final = frozenset({498, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class _Reply:
    status: int
    headers: httpx.Headers
    payload: Any = None
    error_code: str | None = None


class GroqChatProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        total_timeout_seconds: float,
        max_concurrency: int,
        total_attempts: int,
        max_queue_size: int = 8,
        queue_wait_seconds: float = 10.0,
        retry_backoff_seconds: float = 0.5,
        retry_max_backoff_seconds: float = 4.0,
        max_response_bytes: int = 262_144,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._total_timeout_seconds = total_timeout_seconds
        self._total_attempts = total_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._retry_max_backoff_seconds = retry_max_backoff_seconds
        self._max_response_bytes = max_response_bytes
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # admitted 包含呼叫中與等待 semaphore 的請求；檢查與遞增發生在第一個 await 之前，
        # 所以同一個 event loop 內不會超收。
        self._max_admitted = max_concurrency + max_queue_size
        self._admitted = 0
        self._queue_wait_seconds = queue_wait_seconds
        self._client = httpx.AsyncClient(
            headers={
                "authorization": f"Bearer {api_key}",
                "accept": "application/json",
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(
                connect=connect_timeout_seconds,
                read=read_timeout_seconds,
                write=read_timeout_seconds,
                pool=connect_timeout_seconds,
            ),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    async def complete(self, request: LlmRequest) -> LlmResponse:
        if self._admitted >= self._max_admitted:
            raise LlmRateLimitedError(
                "the provider queue is full", retry_after_seconds=self._queue_retry_after()
            )
        self._admitted += 1
        try:
            return await self._complete_admitted(request)
        finally:
            self._admitted -= 1

    async def close(self) -> None:
        await self._client.aclose()

    async def _complete_admitted(self, request: LlmRequest) -> LlmResponse:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "max_completion_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "stream": False,
            **({"reasoning_effort": request.reasoning_effort} if request.reasoning_effort else {}),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_schema.name,
                    "strict": True,
                    "schema": dict(request.output_schema.schema),
                },
            },
        }
        deadline = time.monotonic() + self._total_timeout_seconds
        wait = min(self._queue_wait_seconds, deadline - time.monotonic())
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=wait)
        except TimeoutError as exc:
            raise LlmRateLimitedError(
                "timed out waiting for a provider slot",
                retry_after_seconds=self._queue_retry_after(),
            ) from exc
        try:
            body = await self._post_with_retry(payload, deadline)
        finally:
            self._semaphore.release()
        return self._response(body)

    async def _post_with_retry(self, payload: dict[str, object], deadline: float) -> Any:
        attempt = 0
        schema_retry_available = True
        while True:
            last_attempt = attempt + 1 >= self._total_attempts
            try:
                reply = await self._post(payload, deadline)
            except httpx.TimeoutException as exc:
                if last_attempt:
                    raise LlmTimeoutError("the provider timed out", error_code="timeout") from exc
                await self._backoff(attempt, None, deadline)
                attempt += 1
                continue
            except httpx.TransportError as exc:
                if last_attempt:
                    raise LlmUnavailableError(
                        "the provider is unreachable", error_code="transport_error"
                    ) from exc
                await self._backoff(attempt, None, deadline)
                attempt += 1
                continue

            status = reply.status
            if status == 200:
                return reply.payload
            if status == 400 and reply.error_code == SCHEMA_RETRY_CODE and schema_retry_available:
                schema_retry_available = False
                logger.warning("llm_schema_retry", status=status, error_code=reply.error_code)
                continue
            if status == 429:
                retry_after = _retry_after_seconds(reply.headers)
                # 等不到 retry-after 就回報限流與等待秒數，不當成逾時：結果是確定的，使用者
                # 只需要稍候再問。
                if last_attempt or not self._fits(self._delay(attempt, retry_after), deadline):
                    raise LlmRateLimitedError(
                        "the provider rate limited the request", retry_after_seconds=retry_after
                    )
                logger.warning("llm_retry", status=status, error_code=reply.error_code)
                await self._backoff(attempt, retry_after, deadline)
                attempt += 1
                continue
            if status in RETRYABLE_STATUSES:
                if last_attempt:
                    raise LlmUnavailableError(
                        "the provider is unavailable", error_code=reply.error_code, status=status
                    )
                logger.warning("llm_retry", status=status, error_code=reply.error_code)
                await self._backoff(attempt, _retry_after_seconds(reply.headers), deadline)
                attempt += 1
                continue
            if status in {401, 403, 404}:
                raise LlmUnavailableError(
                    "the provider refused the credentials or model",
                    error_code=reply.error_code,
                    status=status,
                )
            if status in {400, 413, 422}:
                raise LlmContractError(
                    "the provider rejected the request", error_code=reply.error_code, status=status
                )
            raise LlmUnavailableError(
                "the provider returned an unsupported status",
                error_code=reply.error_code,
                status=status,
            )

    async def _post(self, payload: dict[str, object], deadline: float) -> _Reply:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LlmTimeoutError("the provider deadline passed", error_code="timeout")
        try:
            async with (
                asyncio.timeout(remaining),
                self._client.stream("POST", self._url, json=payload) as response,
            ):
                declared = response.headers.get("content-length")
                if declared is not None and declared.isdigit():
                    if int(declared) > self._max_response_bytes:
                        raise LlmContractError(
                            "the provider response is too large", error_code="response_too_large"
                        )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self._max_response_bytes:
                        raise LlmContractError(
                            "the provider response is too large", error_code="response_too_large"
                        )
                if response.status_code != 200:
                    return _Reply(
                        response.status_code,
                        response.headers,
                        error_code=safe_provider_error_code(bytes(body)),
                    )
                if "application/json" not in response.headers.get("content-type", "").casefold():
                    raise LlmContractError(
                        "the provider returned a non-JSON response", error_code="invalid_response"
                    )
                try:
                    parsed: Any = json.loads(bytes(body))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise LlmContractError(
                        "the provider returned invalid JSON", error_code="invalid_response"
                    ) from exc
                return _Reply(response.status_code, response.headers, payload=parsed)
        except TimeoutError as exc:
            raise LlmTimeoutError("the provider deadline passed", error_code="timeout") from exc

    def _delay(self, attempt: int, retry_after: int | None) -> float:
        if retry_after is not None:
            return float(retry_after)
        return min(self._retry_max_backoff_seconds, self._retry_backoff_seconds * 2.0**attempt)

    @staticmethod
    def _fits(delay: float, deadline: float) -> bool:
        return deadline - time.monotonic() > delay

    async def _backoff(self, attempt: int, retry_after: int | None, deadline: float) -> None:
        delay = self._delay(attempt, retry_after)
        if not self._fits(delay, deadline):
            raise LlmTimeoutError("the provider deadline passed", error_code="timeout")
        await asyncio.sleep(delay)

    def _queue_retry_after(self) -> int:
        return max(1, math.ceil(self._queue_wait_seconds))

    def _response(self, payload: Any) -> LlmResponse:
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise LlmContractError(
                "the provider returned invalid choices", error_code="invalid_response"
            )
        choice = choices[0]
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        finish_reason = choice.get("finish_reason")
        if not isinstance(content, str) or not isinstance(finish_reason, str):
            raise LlmContractError(
                "the provider returned no message", error_code="invalid_response"
            )
        if not content.strip() or not finish_reason.strip():
            raise LlmContractError(
                "the provider returned a blank message", error_code="invalid_response"
            )
        usage = payload.get("usage")
        return LlmResponse(
            content=content,
            finish_reason=finish_reason,
            model=self._model,
            usage=TokenUsage(
                prompt=_usage(usage, "prompt_tokens"),
                completion=_usage(usage, "completion_tokens"),
                cached_prompt=_usage(
                    usage.get("prompt_tokens_details") if isinstance(usage, dict) else None,
                    "cached_tokens",
                ),
            ),
        )


def safe_provider_error_code(raw_body: bytes) -> str:
    """回傳 allowlist 內的錯誤代碼；其他任何值都不離開這個函式。"""

    try:
        payload: Any = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "unparsable"
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return "unparsable"
    for field in ("code", "type"):
        value = error.get(field)
        if isinstance(value, str) and value in KNOWN_PROVIDER_ERROR_CODES:
            return value
    return "other"


def _usage(usage: Any, field: str) -> int | None:
    if not isinstance(usage, dict) or field not in usage:
        return None
    value = usage[field]
    if type(value) is not int or value < 0:
        raise LlmContractError("the provider returned invalid usage", error_code="invalid_response")
    return value


def _retry_after_seconds(headers: httpx.Headers) -> int | None:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            return max(0, math.ceil(parsedate_to_datetime(raw).timestamp() - time.time()))
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0, math.ceil(seconds)) if math.isfinite(seconds) else None
