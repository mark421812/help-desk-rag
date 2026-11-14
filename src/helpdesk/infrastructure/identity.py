"""身分 adapter。

`AnonymousIdentity` 給 ACL 關閉時使用：每個瀏覽器以 cookie 裡的隨機 UUID 當匿名身分，只用來
區分對話的擁有者，檢索不比對 ACL。cookie 由 presentation 在第一次造訪時發給。
`DevelopmentHeaderIdentity` 只接受 `x-dev-user-id` 與 `x-dev-group-ids`
header，拒絕 Bearer token、空白、過長與控制字元。header 可以偽造，所以 composition root 在
`APP_ENV=production` 時拒絕使用它。
"""

from __future__ import annotations

from typing import Final
from uuid import UUID

from helpdesk.application.identity import IdentityInput, UnauthorizedError
from helpdesk.domain.access import Principal

MAX_ID_LENGTH: Final = 512
MAX_GROUPS: Final = 100


class AnonymousIdentity:
    def resolve(self, identity: IdentityInput) -> Principal:
        if identity.browser_id is None:
            raise UnauthorizedError("the browser id cookie is required")
        try:
            browser_id = UUID(identity.browser_id)
        except ValueError:
            raise UnauthorizedError("the browser id cookie must be a UUID") from None
        return Principal(user_id=f"browser:{browser_id}")


class DevelopmentHeaderIdentity:
    def resolve(self, identity: IdentityInput) -> Principal:
        if identity.bearer_token is not None:
            raise UnauthorizedError("bearer tokens are not accepted by the development identity")
        if identity.development_user_id is None:
            raise UnauthorizedError("the x-dev-user-id header is required")
        if len(identity.development_group_ids) > MAX_GROUPS:
            raise UnauthorizedError(f"at most {MAX_GROUPS} development groups are allowed")
        user_id = _validated(identity.development_user_id, "user id")
        groups = tuple(_validated(value, "group id") for value in identity.development_group_ids)
        if len(set(groups)) != len(groups):
            raise UnauthorizedError("development group ids must be unique")
        return Principal(user_id=user_id, group_ids=groups)


def _validated(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise UnauthorizedError(f"the development {label} must not be blank")
    if len(normalized) > MAX_ID_LENGTH:
        raise UnauthorizedError(f"the development {label} exceeds {MAX_ID_LENGTH} characters")
    if any(ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F for character in normalized):
        raise UnauthorizedError(f"the development {label} contains a control character")
    return normalized
