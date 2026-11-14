"""Groq adapter：strict schema 請求、重試規則、錯誤代碼 allowlist 與並行上限。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from dataclasses import replace
from typing import Any

import httpx
import pytest

from helpdesk.application.llm import (
    LlmContractError,
    LlmRateLimitedError,
    LlmRequest,
    LlmUnavailableError,
    OutputSchema,
    PromptMessage,
    TokenUsage,
)
from helpdesk.infrastructure.llm.groq import GroqChatProvider

REQUEST = LlmRequest(
    messages=(PromptMessage("system", "rules"), PromptMessage("user", "機密的問題內容")),
    output_schema=OutputSchema("standalone_question", {"type": "object"}),
    max_output_tokens=300,
    temperature=0.0,
    reasoning_effort="low",
)
Handler = Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]


def ok(content: str = '{"standalone_question":"q"}') -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 8},
            },
        },
    )


def error(status: int, code: str, **extra: object) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, **extra}})


class Script:
    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responses.pop(0)


def provider(
    handler: Handler, *, attempts: int = 2, concurrency: int = 2, queue: int = 8
) -> GroqChatProvider:
    return GroqChatProvider(
        base_url="https://api.groq.com/openai/v1",
        api_key="gsk_synthetic",
        model="openai/gpt-oss-20b",
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        total_timeout_seconds=5,
        max_concurrency=concurrency,
        total_attempts=attempts,
        max_queue_size=queue,
        queue_wait_seconds=0.05,
        retry_backoff_seconds=0,
        retry_max_backoff_seconds=0,
        transport=httpx.MockTransport(handler),
    )


async def test_the_request_uses_strict_json_schema_and_the_reply_is_mapped() -> None:
    script = Script(ok())

    response = await provider(script).complete(REQUEST)

    [sent] = script.requests
    body = json.loads(sent.content)
    assert str(sent.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert sent.headers["authorization"] == "Bearer gsk_synthetic"
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "standalone_question",
            "strict": True,
            "schema": {"type": "object"},
        },
    }
    assert body["max_completion_tokens"] == 300
    assert body["stream"] is False
    assert body["reasoning_effort"] == "low"
    assert (response.content, response.finish_reason) == ('{"standalone_question":"q"}', "stop")
    assert response.usage == TokenUsage(prompt=12, completion=3, cached_prompt=8)


async def test_a_request_without_reasoning_effort_leaves_the_provider_default() -> None:
    script = Script(
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]},
        )
    )

    response = await provider(script).complete(replace(REQUEST, reasoning_effort=None))

    [sent] = script.requests
    assert "reasoning_effort" not in json.loads(sent.content)
    assert response.usage == TokenUsage()


@pytest.mark.parametrize("attempts", [1, 2])
async def test_json_validate_failed_is_retried_once_on_top_of_other_attempts(attempts: int) -> None:
    script = Script(error(400, "json_validate_failed"), ok())

    response = await provider(script, attempts=attempts).complete(REQUEST)

    assert response.finish_reason == "stop"
    assert len(script.requests) == 2


async def test_json_validate_failed_is_retried_only_once() -> None:
    script = Script(error(400, "json_validate_failed"), error(400, "json_validate_failed"), ok())

    with pytest.raises(LlmContractError) as raised:
        await provider(script, attempts=3).complete(REQUEST)

    assert (raised.value.error_code, raised.value.status) == ("json_validate_failed", 400)
    assert len(script.requests) == 2


async def test_other_bad_requests_are_not_retried() -> None:
    script = Script(error(400, "invalid_request_error"), ok())

    with pytest.raises(LlmContractError) as raised:
        await provider(script).complete(REQUEST)

    assert raised.value.error_code == "invalid_request_error"
    assert len(script.requests) == 1


async def test_an_error_body_that_echoes_the_prompt_never_leaves_the_adapter() -> None:
    script = Script(
        error(400, "something_new", message="機密的問題內容", failed_generation="機密的問題內容")
    )

    with pytest.raises(LlmContractError) as raised:
        await provider(script).complete(REQUEST)

    assert raised.value.error_code == "other"
    assert "機密" not in str(raised.value) and "機密" not in repr(raised.value.__dict__)


async def test_rate_limits_and_server_errors_are_retried_within_the_attempt_budget() -> None:
    limited = httpx.Response(429, headers={"retry-after": "0"}, json={"error": {}})
    script = Script(limited, ok())
    assert (await provider(script).complete(REQUEST)).finish_reason == "stop"

    script = Script(error(503, "other"), ok())
    assert (await provider(script).complete(REQUEST)).finish_reason == "stop"


async def test_exhausted_attempts_raise_typed_errors() -> None:
    limited = httpx.Response(429, headers={"retry-after": "0"}, json={"error": {}})
    with pytest.raises(LlmRateLimitedError):
        await provider(Script(limited, limited)).complete(REQUEST)

    script = Script(error(502, "other"), error(502, "other"))
    with pytest.raises(LlmUnavailableError):
        await provider(script).complete(REQUEST)
    assert len(script.requests) == 2


async def test_a_rate_limit_longer_than_the_deadline_is_reported_with_its_wait() -> None:
    limited = httpx.Response(429, headers={"retry-after": "60"}, json={"error": {}})
    script = Script(limited, ok())

    with pytest.raises(LlmRateLimitedError) as raised:
        await provider(script).complete(REQUEST)

    assert raised.value.retry_after_seconds == 60
    assert len(script.requests) == 1


async def test_an_authorization_failure_is_not_retried() -> None:
    script = Script(error(401, "invalid_api_key"), ok())

    with pytest.raises(LlmUnavailableError):
        await provider(script).complete(REQUEST)

    assert len(script.requests) == 1


async def test_requests_beyond_the_concurrency_and_queue_limits_are_refused() -> None:
    release = asyncio.Event()

    async def hold(request: httpx.Request) -> httpx.Response:
        await release.wait()
        return ok()

    groq = provider(hold, concurrency=1, queue=0)
    first = asyncio.create_task(groq.complete(REQUEST))
    await asyncio.sleep(0.01)

    with pytest.raises(LlmRateLimitedError):
        await groq.complete(REQUEST)

    release.set()
    assert (await first).finish_reason == "stop"


async def test_a_reply_without_a_message_is_a_contract_error() -> None:
    script = Script(httpx.Response(200, json={"choices": []}))

    with pytest.raises(LlmContractError):
        await provider(script).complete(REQUEST)
