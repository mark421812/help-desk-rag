"""呼叫者身分的 port。

presentation 從 HTTP header 與 cookie 取出未驗證的值交給 `IdentityResolver`，得到驗證過的
`Principal`。ACL 關閉時是瀏覽器 cookie 裡的匿名 ID，開啟時是開發用 header；身分永遠不從
request body 來。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from helpdesk.domain.access import Principal
from helpdesk.domain.errors import DomainError


@dataclass(frozen=True, slots=True)
class IdentityInput:
    bearer_token: str | None = None
    development_user_id: str | None = None
    development_group_ids: tuple[str, ...] = ()
    browser_id: str | None = None


class UnauthorizedError(DomainError):
    """沒有可接受的身分。訊息只描述哪個欄位不合格，不帶 header 或 cookie 的值。"""


class IdentityResolver(Protocol):
    def resolve(self, identity: IdentityInput) -> Principal: ...
