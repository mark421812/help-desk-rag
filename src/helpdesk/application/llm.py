"""外部 LLM 的 port 與請求、回應資料。

追問改寫與產生回答共用這個 port。provider 一律以 strict JSON schema
模式呼叫，所以每個請求都帶 `OutputSchema`。例外只帶 allowlist 內的錯誤代碼與 HTTP 狀態，
不帶 prompt 或模型輸出，可以直接寫進 log。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

ReasoningEffort = Literal["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class PromptMessage:
    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True, slots=True)
class OutputSchema:
    name: str
    schema: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class LlmRequest:
    messages: tuple[PromptMessage, ...]
    output_schema: OutputSchema
    max_output_tokens: int
    temperature: float
    reasoning_effort: ReasoningEffort | None = None


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """provider 回報的用量。`cached_prompt` 是命中 prompt cache 的部分，已含在 `prompt` 裡。"""

    prompt: int | None = None
    completion: int | None = None
    cached_prompt: int | None = None


@dataclass(frozen=True, slots=True)
class LlmResponse:
    content: str
    finish_reason: str
    model: str
    usage: TokenUsage = TokenUsage()


class LlmProvider(Protocol):
    async def complete(self, request: LlmRequest) -> LlmResponse: ...

    async def close(self) -> None: ...


class LlmError(Exception):
    """provider 呼叫失敗。`error_code` 只會是 provider 文件列出的代碼或 adapter 自己的代碼。"""

    def __init__(
        self, message: str, *, error_code: str | None = None, status: int | None = None
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status = status


class LlmTimeoutError(LlmError):
    """超過總時限，或排隊等不到空位。"""


class LlmRateLimitedError(LlmError):
    """provider 回 429，或本機的等待佇列已滿。"""

    def __init__(self, message: str, *, retry_after_seconds: int | None = None) -> None:
        super().__init__(message, error_code="rate_limited", status=429)
        self.retry_after_seconds = retry_after_seconds


class LlmUnavailableError(LlmError):
    """連線失敗、5xx、授權失敗等暫時或設定上的問題。"""


class LlmContractError(LlmError):
    """provider 拒絕了請求，或回應不符合約定的格式。"""


def prompt_bytes(messages: tuple[PromptMessage, ...]) -> int:
    """prompt 大小的保守上界：每則訊息 4 個固定成本加上 UTF-8 bytes。

    一個 token 至少承載一個 byte，所以 byte 數不會低估 token 數，對 Base64 或程式碼也一樣。
    """

    return sum(4 + len(message.content.encode("utf-8")) for message in messages)
