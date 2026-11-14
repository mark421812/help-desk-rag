"""離線的 Fake provider，給 `LLM_PROVIDER=fake` 與測試使用。

依序回傳事先排好的回覆（文字、完整回應或例外）並記下收到的請求；沒有回覆可給時回報 provider
無法使用，呼叫端因此走失敗路徑，例如改寫退回原問題。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from helpdesk.application.llm import LlmRequest, LlmResponse, LlmUnavailableError

Reply = str | LlmResponse | Exception


class FakeLlmProvider:
    def __init__(self, replies: Iterable[Reply] = ()) -> None:
        self._replies: deque[Reply] = deque(replies)
        self.requests: list[LlmRequest] = []

    async def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if not self._replies:
            raise LlmUnavailableError(
                "the fake provider has no scripted reply", error_code="fake_exhausted"
            )
        reply = self._replies.popleft()
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, LlmResponse):
            return reply
        return LlmResponse(content=reply, finish_reason="stop", model="fake")

    async def close(self) -> None:
        return None
