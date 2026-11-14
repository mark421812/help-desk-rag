"""存取規則與呼叫者身分。

ACL 開關關閉時這些資料照常保存，只是不拿來比對；之後打開開關不需要重建資料。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from helpdesk.domain._checks import MAX_PRINCIPAL_ID_LENGTH, require_text, require_unique


class PrincipalType(StrEnum):
    USER = "user"
    GROUP = "group"


class PermissionEffect(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class AccessRule:
    """文件上的一條規則：某個使用者或群組被允許或拒絕讀取。"""

    principal_type: PrincipalType
    principal_id: str
    effect: PermissionEffect = PermissionEffect.ALLOW

    def __post_init__(self) -> None:
        require_text(self.principal_id, "principal_id", max_length=MAX_PRINCIPAL_ID_LENGTH)


@dataclass(frozen=True, slots=True)
class Principal:
    """身分 adapter 驗證過的呼叫者。

    ACL 開啟時 `user_id` 來自開發用身分 header；關閉時是瀏覽器的匿名 ID，`group_ids` 為空。
    兩種情況都用 `user_id` 當對話擁有者。
    """

    user_id: str
    group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.user_id, "user_id", max_length=MAX_PRINCIPAL_ID_LENGTH)
        for group_id in self.group_ids:
            require_text(group_id, "group_id", max_length=MAX_PRINCIPAL_ID_LENGTH)
        require_unique(self.group_ids, "group_ids")
