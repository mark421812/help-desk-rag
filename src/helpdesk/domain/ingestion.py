"""Ingestion 的兩個狀態機：來源檔案 `SourceItem` 與工作佇列 `IngestionJob`。

兩者都依轉移表前進，表外的移動一律拒絕。檔案「消失」不是處理狀態，而是
`missing_scan_count` 計數：檔案一時沒掃到時保留原本的處理狀態，重新出現就能回到原狀。
文件被拒絕的原因用例外型別表達，每個型別自帶落點狀態與錯誤代碼，呼叫端只需要一個
except 分支。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Final, NewType, TypeVar
from uuid import NAMESPACE_URL, UUID, uuid5

from helpdesk.domain._checks import require_sha256, require_text
from helpdesk.domain.documents import SourceItemId
from helpdesk.domain.errors import DomainError, DomainValidationError, InvalidTransitionError

JobId = NewType("JobId", UUID)
_State = TypeVar("_State", bound=StrEnum)

MAX_ERROR_MESSAGE_LENGTH: Final = 2_000


class SourceItemStatus(StrEnum):
    """一個來源檔案最近一次處理的結果。

    `metadata_missing`、`ocr_required`、`quarantined` 是永久拒絕，只有檔案或 metadata
    改變才會重試。`failed` 是可重試的失敗，job 會依退避時間再跑。
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    METADATA_MISSING = "metadata_missing"
    OCR_REQUIRED = "ocr_required"
    QUARANTINED = "quarantined"
    FAILED = "failed"
    DELETED = "deleted"


_REJECTED: Final = frozenset(
    {
        SourceItemStatus.METADATA_MISSING,
        SourceItemStatus.OCR_REQUIRED,
        SourceItemStatus.QUARANTINED,
    }
)
_SETTLED: Final = frozenset({SourceItemStatus.COMPLETED, SourceItemStatus.FAILED, *_REJECTED})
_PROCESSING_OUTCOMES: Final = frozenset(
    {SourceItemStatus.PENDING, SourceItemStatus.PROCESSING, SourceItemStatus.DELETED, *_SETTLED}
)

# 同一個檔案可能排了好幾個 job（每次內容改變一個），所以任何還存在的狀態都能再進入
# processing；deleted 只能因為檔案重新出現而回到 pending。
_ITEM_TRANSITIONS: Final[Mapping[SourceItemStatus, frozenset[SourceItemStatus]]] = MappingProxyType(
    {
        SourceItemStatus.PENDING: frozenset(
            {SourceItemStatus.PENDING, SourceItemStatus.PROCESSING, SourceItemStatus.DELETED}
        ),
        SourceItemStatus.PROCESSING: _PROCESSING_OUTCOMES,
        **{
            status: frozenset(
                {
                    SourceItemStatus.PENDING,
                    SourceItemStatus.PROCESSING,
                    SourceItemStatus.DELETED,
                }
            )
            for status in _SETTLED
        },
        SourceItemStatus.DELETED: frozenset({SourceItemStatus.PENDING}),
    }
)


class IngestionJobType(StrEnum):
    PROCESS_ITEM = "process_item"
    DELETE_ITEM = "delete_item"


class IngestionJobStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# processing → processing 是 lease 過期後由另一個 worker 接手。
_JOB_TRANSITIONS: Final[Mapping[IngestionJobStatus, frozenset[IngestionJobStatus]]] = (
    MappingProxyType(
        {
            IngestionJobStatus.PENDING: frozenset({IngestionJobStatus.PROCESSING}),
            IngestionJobStatus.PROCESSING: frozenset(
                {
                    IngestionJobStatus.PROCESSING,
                    IngestionJobStatus.PENDING,
                    IngestionJobStatus.COMPLETED,
                    IngestionJobStatus.FAILED,
                }
            ),
            IngestionJobStatus.COMPLETED: frozenset(),
            IngestionJobStatus.FAILED: frozenset({IngestionJobStatus.PENDING}),
        }
    )
)


class LeaseLostError(DomainError):
    """worker 已經不再持有這個 job 的 lease，不能寫回結果。"""


class SourceChangedError(DomainError):
    """處理期間檔案被改動或移除；稍後重試會讀到新的內容。"""


class ScanRejectedError(DomainError):
    """掃描結果看起來像來源掛載失敗，拒絕據此刪除文件。"""


class ScanInProgressError(DomainError):
    """另一次掃描正在進行。"""


class SourceFileNotFoundError(DomainError):
    """指定的相對路徑在文件根目錄下找不到支援的檔案。"""


class JobNotFoundError(DomainError):
    pass


class DocumentRejectedError(DomainError):
    """文件不能被索引的永久原因。`status` 與 `code` 決定來源檔案的落點。"""

    status: ClassVar[SourceItemStatus] = SourceItemStatus.QUARANTINED
    code: ClassVar[str] = "invalid_document"


class InvalidDocumentError(DocumentRejectedError):
    """格式支援，但檔案無法安全解析，例如加密、損毀或沒有可索引的內容。"""


class UnsupportedDocumentError(DocumentRejectedError):
    """沒有任何 parser 支援這個副檔名或 MIME type。"""


class FileTooLargeError(DocumentRejectedError):
    code = "file_too_large"


class OcrRequiredError(DocumentRejectedError):
    """PDF 沒有文字層，需要 OCR。"""

    status = SourceItemStatus.OCR_REQUIRED
    code = "ocr_required"


class InvalidMetadataError(DocumentRejectedError):
    """metadata 檔案存在但內容不合法，例如 YAML 錯誤、未知欄位或違反 domain 規則。"""

    code = "invalid_metadata"


class MetadataMissingError(DocumentRejectedError):
    """ACL 開啟時缺少必填的 metadata 欄位。"""

    status = SourceItemStatus.METADATA_MISSING
    code = "metadata_missing"

    def __init__(self, missing_fields: tuple[str, ...]) -> None:
        self.missing_fields = tuple(sorted(set(missing_fields)))
        super().__init__(f"Missing required metadata: {', '.join(self.missing_fields)}")


def _move(
    table: Mapping[_State, frozenset[_State]], current: _State, target: _State, name: str
) -> _State:
    if target not in table[current]:
        raise InvalidTransitionError(f"{name} cannot move from {current} to {target}")
    return target


def _bounded_message(message: str) -> str:
    return message[:MAX_ERROR_MESSAGE_LENGTH]


@dataclass(frozen=True, slots=True)
class SourceItem:
    """文件根目錄下的一個檔案，以相對路徑識別。

    `source_version` 是修改時間加大小，`metadata_source_hash` 是生效 metadata 檔案與
    metadata 規則的指紋，兩者合起來是掃描時判斷要不要重做的依據。
    `change_generation` 在掃描每看到一次變化（內容、metadata、ACL 開關，或消失後又出現）時加一，
    處理 job 的 key 由它組成。內容或 metadata 改回先前的樣子也會得到新的 key，不會因為舊 job
    還在而漏做。
    """

    id: SourceItemId
    relative_path: str
    source_version: str
    metadata_source_hash: str
    file_size: int
    modified_at: datetime
    status: SourceItemStatus = SourceItemStatus.PENDING
    mime_type: str | None = None
    missing_scan_count: int = 0
    change_generation: int = 0
    last_seen_at: datetime | None = None
    last_successful_content_hash: str | None = None
    last_successful_metadata_hash: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    completed_at: datetime | None = None
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        require_text(self.relative_path, "relative_path")
        require_text(self.source_version, "source_version", max_length=255)
        require_sha256(self.metadata_source_hash, "metadata_source_hash")
        if self.file_size < 0 or self.missing_scan_count < 0 or self.change_generation < 0:
            raise DomainValidationError("source item sizes and counters must not be negative")
        if (self.status is SourceItemStatus.DELETED) != (self.deleted_at is not None):
            raise DomainValidationError("exactly the deleted items carry deleted_at")
        if (self.error_code is None) != (self.error_message is None):
            raise DomainValidationError("error_code and error_message go together")

    @staticmethod
    def id_for(relative_path: str) -> SourceItemId:
        return SourceItemId(uuid5(NAMESPACE_URL, f"helpdesk:source-item:{relative_path}"))

    @classmethod
    def discovered(
        cls,
        relative_path: str,
        *,
        source_version: str,
        metadata_source_hash: str,
        file_size: int,
        modified_at: datetime,
        at: datetime,
    ) -> SourceItem:
        return cls(
            id=cls.id_for(relative_path),
            relative_path=relative_path,
            source_version=source_version,
            metadata_source_hash=metadata_source_hash,
            file_size=file_size,
            modified_at=modified_at,
            last_seen_at=at,
        )

    @property
    def discovery_fingerprint(self) -> str:
        return f"{self.source_version}:{self.metadata_source_hash}"

    @property
    def is_missing(self) -> bool:
        return self.missing_scan_count > 0 and self.status is not SourceItemStatus.DELETED

    def observe(
        self,
        *,
        source_version: str,
        metadata_source_hash: str,
        file_size: int,
        modified_at: datetime,
        at: datetime,
    ) -> tuple[SourceItem, bool]:
        """套用一次掃描到的檔案；回傳更新後的 item 與是否需要排處理 job。"""

        returned = self.missing_scan_count > 0 or self.status is SourceItemStatus.DELETED
        changed = returned or (source_version, metadata_source_hash) != (
            self.source_version,
            self.metadata_source_hash,
        )
        observed = replace(
            self,
            source_version=source_version,
            metadata_source_hash=metadata_source_hash,
            file_size=file_size,
            modified_at=modified_at,
            missing_scan_count=0,
            change_generation=self.change_generation + int(changed),
            last_seen_at=at,
        )
        if changed:
            observed = observed._moved(SourceItemStatus.PENDING)
        needs_processing = changed or self.last_successful_content_hash is None
        return observed, needs_processing

    def mark_missing(self) -> SourceItem:
        return replace(self, missing_scan_count=self.missing_scan_count + 1)

    def begin_processing(self) -> SourceItem:
        return self._moved(SourceItemStatus.PROCESSING)

    def complete(self, *, content_hash: str, mime_type: str, at: datetime) -> SourceItem:
        require_sha256(content_hash, "content_hash")
        completed = self._moved(SourceItemStatus.COMPLETED)
        return replace(
            completed,
            mime_type=mime_type,
            last_successful_content_hash=content_hash,
            last_successful_metadata_hash=self.metadata_source_hash,
            completed_at=at,
        )

    def reject(self, error: DocumentRejectedError) -> SourceItem:
        rejected = self._moved(error.status, error=(error.code, str(error)))
        return replace(
            rejected, last_successful_content_hash=None, last_successful_metadata_hash=None
        )

    def fail(self, code: str, message: str) -> SourceItem:
        return self._moved(SourceItemStatus.FAILED, error=(code, message))

    def requeue(self) -> SourceItem:
        return self._moved(SourceItemStatus.PENDING)

    def mark_deleted(self, at: datetime) -> SourceItem:
        deleted = self._moved(SourceItemStatus.DELETED, deleted_at=at)
        return replace(
            deleted, last_successful_content_hash=None, last_successful_metadata_hash=None
        )

    def _moved(
        self,
        target: SourceItemStatus,
        *,
        error: tuple[str, str] | None = None,
        deleted_at: datetime | None = None,
    ) -> SourceItem:
        status = _move(_ITEM_TRANSITIONS, self.status, target, "source item")
        code, message = error if error is not None else (None, None)
        return replace(
            self,
            status=status,
            error_code=code,
            error_message=None if message is None else _bounded_message(message),
            deleted_at=deleted_at,
        )


@dataclass(frozen=True, slots=True)
class IngestionJob:
    """PostgreSQL 工作佇列的一筆 job，帶 lease、重試次數與退避時間。

    `idempotency_key` 描述「為了哪一次變化而做」，同一把 key 只會排一次 job；job 的 ID
    由 key 推導，重複排程因此不會產生第二筆。
    """

    id: JobId
    source_item_id: SourceItemId
    job_type: IngestionJobType
    idempotency_key: str
    available_at: datetime
    max_attempts: int
    status: IngestionJobStatus = IngestionJobStatus.PENDING
    attempt_count: int = 0
    leased_by: str | None = None
    lease_expires_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        require_text(self.idempotency_key, "idempotency_key", max_length=512)
        if self.attempt_count < 0 or self.max_attempts <= 0:
            raise DomainValidationError("job attempt counters are invalid")
        if (self.status is IngestionJobStatus.PROCESSING) != (self.leased_by is not None):
            raise DomainValidationError("exactly the processing jobs hold a lease")
        if (self.leased_by is None) != (self.lease_expires_at is None):
            raise DomainValidationError("a lease needs both an owner and an expiry")

    @classmethod
    def process(cls, item: SourceItem, *, max_attempts: int, now: datetime) -> IngestionJob:
        key = f"process:{item.id}:{item.change_generation}"
        return cls._new(item, IngestionJobType.PROCESS_ITEM, key, max_attempts, now)

    @classmethod
    def delete(cls, item: SourceItem, *, max_attempts: int, now: datetime) -> IngestionJob:
        key = f"delete:{item.id}:{item.change_generation}"
        return cls._new(item, IngestionJobType.DELETE_ITEM, key, max_attempts, now)

    @classmethod
    def _new(
        cls,
        item: SourceItem,
        job_type: IngestionJobType,
        key: str,
        max_attempts: int,
        now: datetime,
    ) -> IngestionJob:
        return cls(
            id=JobId(uuid5(NAMESPACE_URL, f"helpdesk:ingestion-job:{key}")),
            source_item_id=item.id,
            job_type=job_type,
            idempotency_key=key,
            available_at=now,
            max_attempts=max_attempts,
        )

    def is_claimable(self, now: datetime) -> bool:
        if self.status is IngestionJobStatus.PENDING:
            return self.available_at <= now
        return (
            self.status is IngestionJobStatus.PROCESSING
            and self.lease_expires_at is not None
            and self.lease_expires_at <= now
        )

    def claim(self, worker_id: str, now: datetime, lease_for: timedelta) -> IngestionJob:
        require_text(worker_id, "worker_id", max_length=255)
        if not self.is_claimable(now):
            raise LeaseLostError("job is not claimable")
        return replace(
            self,
            status=self._next(IngestionJobStatus.PROCESSING),
            leased_by=worker_id,
            lease_expires_at=now + lease_for,
        )

    def assert_lease(self, worker_id: str, now: datetime) -> None:
        if (
            self.status is not IngestionJobStatus.PROCESSING
            or self.leased_by != worker_id
            or self.lease_expires_at is None
            or self.lease_expires_at <= now
        ):
            raise LeaseLostError("worker does not own an active job lease")

    def complete(self, worker_id: str, now: datetime) -> IngestionJob:
        self.assert_lease(worker_id, now)
        return replace(
            self,
            status=self._next(IngestionJobStatus.COMPLETED),
            leased_by=None,
            lease_expires_at=None,
            error_code=None,
            error_message=None,
            completed_at=now,
        )

    def fail(
        self,
        worker_id: str,
        now: datetime,
        *,
        code: str,
        message: str,
        retryable: bool,
        base_delay: timedelta,
    ) -> IngestionJob:
        """記一次失敗。可重試時第 n 次失敗後等待 base_delay × 2^(n-1)。"""

        self.assert_lease(worker_id, now)
        attempts = self.attempt_count + 1
        retry = retryable and attempts < self.max_attempts
        return replace(
            self,
            status=self._next(IngestionJobStatus.PENDING if retry else IngestionJobStatus.FAILED),
            attempt_count=attempts,
            available_at=now + base_delay * 2 ** (attempts - 1) if retry else self.available_at,
            leased_by=None,
            lease_expires_at=None,
            error_code=code,
            error_message=_bounded_message(message),
            completed_at=None if retry else now,
        )

    def retry_manually(self, now: datetime) -> IngestionJob:
        return replace(
            self,
            status=self._next(IngestionJobStatus.PENDING),
            attempt_count=0,
            available_at=now,
            error_code=None,
            error_message=None,
            completed_at=None,
        )

    def _next(self, target: IngestionJobStatus) -> IngestionJobStatus:
        return _move(_JOB_TRANSITIONS, self.status, target, "ingestion job")
