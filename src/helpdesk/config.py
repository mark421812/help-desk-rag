"""唯一的 Settings：兩個程序與 migration 共用，啟動時一次驗證完畢。

欄位名稱對應同名的大寫環境變數，例如 `access_control_enabled` 對應
`ACCESS_CONTROL_ENABLED`；環境變數優先於 `.env`。這裡刻意不提供全域快取的
`get_settings()`：composition root 建立一次 `Settings` 並把選好的實作注入下去，兩個開關
因此只有一個決策點。`GROQ_API_KEY` 只有 helpdesk-api 用得到，所以由它的 composition root
檢查，ingestion-worker 與 alembic 不需要它。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal, Self
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# prompt 上限以 UTF-8 bytes 保守估算；中文每個 E5 token 實測約 4.5 bytes，再加上 system
# prompt、問題與 JSON 欄位的固定開銷，確保滿額證據不會被自己的 prompt 上限擋下。
PROMPT_BYTES_PER_CONTEXT_TOKEN: Final = 6
PROMPT_FIXED_OVERHEAD_BYTES: Final = 8_192
PROMPT_BYTES_PER_HISTORY_CHAR: Final = 6
PROMPT_OVERHEAD_BYTES_PER_HISTORY_MESSAGE: Final = 64


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    log_level: str = "INFO"
    app_env: Literal["development", "production"] = "development"

    access_control_enabled: bool = True
    sensitive_data_scan_enabled: bool = True

    llm_provider: Literal["groq", "fake"] = "groq"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_allowed_hosts: list[str] = Field(default_factory=lambda: ["api.groq.com"])
    groq_connect_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    groq_read_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    groq_total_timeout_seconds: float = Field(default=75.0, gt=0, le=900)
    groq_max_output_tokens: int = Field(default=3_000, ge=1, le=16_384)
    groq_temperature: float = Field(default=0.3, ge=0, le=2)
    answer_reasoning_effort: Literal["low", "medium", "high"] = "low"
    groq_max_concurrency: int = Field(default=2, ge=1, le=100)
    groq_total_attempts: int = Field(default=2, ge=1, le=3)

    database_url: str = "postgresql+asyncpg://helpdesk:helpdesk@postgres:5432/helpdesk"

    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "helpdesk_chunks_v1"

    embedding_model_id: str = "intfloat/multilingual-e5-small"
    embedding_model_revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    embedding_max_sequence_length: int = Field(default=512, ge=1)
    embedding_batch_size: int = Field(default=32, ge=1, le=1_024)

    chunk_target_tokens: int = Field(default=350, ge=1)
    chunk_max_tokens: int = Field(default=450, ge=1)
    chunk_overlap_tokens: int = Field(default=50, ge=0)

    retrieval_limit: int = Field(default=8, ge=1, le=50)
    candidate_multiplier: int = Field(default=3, ge=1, le=10)
    keyword_min_score: int = Field(default=3, ge=1)
    max_chunks_per_document: int = Field(default=6, ge=1)
    max_context_tokens: int = Field(default=6_000, ge=1, le=100_000)
    max_prompt_tokens: int = Field(default=64_000, ge=1, le=1_000_000)

    rewrite_history_turns: int = Field(default=3, ge=1, le=10)
    rewrite_turn_max_chars: int = Field(default=500, ge=1, le=2_000)
    rewrite_max_chars: int = Field(default=500, ge=1, le=2_000)
    rewrite_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    rewrite_reasoning_effort: Literal["low", "medium", "high"] = "low"

    turn_max_concurrency: int = Field(default=4, ge=1, le=100)
    turn_stale_after_seconds: int = Field(default=300, ge=1, le=86_400)

    documents_root: Path = Path("/data/documents")
    max_file_size_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    missing_scans_before_delete: int = Field(default=2, ge=2)
    max_missing_ratio: float = Field(default=0.25, gt=0, le=1)

    worker_max_attempts: int = Field(default=5, ge=1, le=100)
    worker_lease_seconds: int = Field(default=3_600, ge=60)
    worker_retry_base_seconds: float = Field(default=5.0, gt=0)
    worker_poll_interval_seconds: float = Field(default=2.0, gt=0)

    schedule_hour: int = Field(default=2, ge=0, le=23)
    schedule_minute: int = Field(default=0, ge=0, le=59)
    schedule_timezone: str = "Asia/Taipei"

    @field_validator("groq_api_key", "qdrant_api_key", mode="before")
    @classmethod
    def blank_secret_is_unset(cls, value: object) -> object:
        """`.env` 裡留空的金鑰等同沒設定。"""

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            if any(character in value for character in ("\r", "\n", "\x00")):
                raise ValueError("API keys must not contain control characters")
        return value

    @field_validator(
        "groq_model",
        "database_url",
        "qdrant_collection",
        "embedding_model_id",
        "embedding_model_revision",
    )
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text settings must not be blank")
        return normalized

    @field_validator("groq_base_url", "qdrant_url")
    @classmethod
    def base_url_has_no_trailing_slash(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("base URLs must not be blank")
        return normalized

    @field_validator("schedule_timezone")
    @classmethod
    def timezone_must_exist(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("SCHEDULE_TIMEZONE is not a known IANA timezone") from exc
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("CHUNK_TARGET_TOKENS must not exceed CHUNK_MAX_TOKENS")
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TARGET_TOKENS")
        if self.chunk_max_tokens >= self.embedding_max_sequence_length:
            raise ValueError("CHUNK_MAX_TOKENS must leave room below the embedding sequence limit")
        history_bytes = (
            self.rewrite_history_turns
            * 2
            * (
                self.rewrite_turn_max_chars * PROMPT_BYTES_PER_HISTORY_CHAR
                + PROMPT_OVERHEAD_BYTES_PER_HISTORY_MESSAGE
            )
        )
        minimum_prompt_tokens = (
            self.max_context_tokens * PROMPT_BYTES_PER_CONTEXT_TOKEN
            + history_bytes
            + PROMPT_FIXED_OVERHEAD_BYTES
        )
        if self.max_prompt_tokens < minimum_prompt_tokens:
            raise ValueError(
                f"MAX_PROMPT_TOKENS must be at least {minimum_prompt_tokens} so a full "
                "MAX_CONTEXT_TOKENS evidence budget and the conversation history fit the "
                "byte-based prompt estimate"
            )
        if self.groq_connect_timeout_seconds > self.groq_total_timeout_seconds:
            raise ValueError("GROQ_CONNECT_TIMEOUT_SECONDS must not exceed the total timeout")
        if self.groq_read_timeout_seconds > self.groq_total_timeout_seconds:
            raise ValueError("GROQ_READ_TIMEOUT_SECONDS must not exceed the total timeout")
        if self.llm_provider == "groq":
            _require_allowed_https_host(self.groq_base_url, self.groq_allowed_hosts)
        longest_turn = self.rewrite_timeout_seconds + self.groq_total_timeout_seconds
        if self.turn_stale_after_seconds < 2 * longest_turn:
            raise ValueError(
                "TURN_STALE_AFTER_SECONDS must be at least twice REWRITE_TIMEOUT_SECONDS plus "
                "GROQ_TOTAL_TIMEOUT_SECONDS, or recovery would fail turns that are still running"
            )
        return self


def _require_allowed_https_host(url: str, allowed_hosts: list[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("GROQ_BASE_URL must use HTTPS")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("GROQ_BASE_URL must not contain credentials, query or fragment")
    if parsed.hostname.casefold() not in {host.strip().casefold() for host in allowed_hosts}:
        raise ValueError("GROQ_BASE_URL host is not in GROQ_ALLOWED_HOSTS")
