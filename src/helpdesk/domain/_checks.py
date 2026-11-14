"""Domain 共用的欄位檢查與長度上限。

錯誤訊息只寫欄位名稱，不帶輸入值，問題文字或文件內容因此不會跟著例外進到 log。
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Hashable
from typing import Final

from helpdesk.domain.errors import DomainValidationError

MAX_LANGUAGE_LENGTH: Final = 20
MAX_QUESTION_LENGTH: Final = 20_000
MAX_PRINCIPAL_ID_LENGTH: Final = 512

_SHA256: Final = re.compile(r"[0-9a-f]{64}")


def require_text(value: str, name: str, *, max_length: int | None = None) -> None:
    if not value.strip():
        raise DomainValidationError(f"{name} must not be blank")
    if max_length is not None and len(value) > max_length:
        raise DomainValidationError(f"{name} must not exceed {max_length} characters")


def optional_text(value: str | None, name: str, *, max_length: int | None = None) -> None:
    if value is not None:
        require_text(value, name, max_length=max_length)


def require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise DomainValidationError(f"{name} must be positive")


def optional_positive(value: int | None, name: str) -> None:
    if value is not None:
        require_positive(value, name)


def require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise DomainValidationError(f"{name} must be finite")


def require_sha256(value: str, name: str) -> None:
    if not _SHA256.fullmatch(value):
        raise DomainValidationError(f"{name} must be a lowercase SHA-256 hex digest")


def require_unique(values: Collection[Hashable], name: str) -> None:
    if len(set(values)) != len(values):
        raise DomainValidationError(f"{name} must not contain duplicates")
