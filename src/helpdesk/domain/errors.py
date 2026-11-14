"""Domain 規則被違反時的例外。呼叫端依型別分流，不解析訊息文字。"""


class DomainError(Exception):
    """所有可預期的 domain 失敗。"""


class DomainValidationError(DomainError):
    """資料不符合 domain invariant。"""


class InvalidTransitionError(DomainError):
    """回合狀態轉移不在允許的轉移表內。"""


class ArchivedConversationError(DomainError):
    """已封存的對話不接受新回合。"""


class IdempotencyConflictError(DomainError):
    """同一個 Idempotency-Key 帶了內容不同的請求。"""
