"""掃描文件根目錄，依變化排處理 job；連續消失的檔案排刪除 job。

掃描只做發現與排程，不解析文件。檔案沒變、也已經成功處理過時不排 job；同一次變化的
job 由 idempotency key 去重，所以重跑掃描不會重複排程。
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime

from helpdesk.application.ingestion.ports import (
    Clock,
    DiscoveredFile,
    DocumentSource,
    MetadataSource,
    ScanLock,
    UnitOfWork,
    UnitOfWorkFactory,
)
from helpdesk.domain.ingestion import (
    IngestionJob,
    JobId,
    ScanRejectedError,
    SourceFileNotFoundError,
    SourceItem,
    SourceItemStatus,
)


@dataclass(frozen=True, slots=True)
class ScanReport:
    discovered: int
    queued: int
    missing: int
    deletions_queued: int


@dataclass(frozen=True, slots=True)
class IngestReport:
    item: SourceItem
    job_id: JobId | None
    queued: bool


@dataclass(frozen=True, slots=True)
class DeletionRules:
    """檔案要連續 `missing_scans_before_delete` 次掃不到才刪除。單次掃描消失的比例超過
    `max_missing_ratio`，或整個來源變成空的，就當成掛載失敗而拒絕整次掃描。"""

    missing_scans_before_delete: int = 2
    max_missing_ratio: float = 0.25

    def check(self, *, active: int, absent: int, discovered: int) -> None:
        if active and (not discovered or absent / active > self.max_missing_ratio):
            raise ScanRejectedError(
                f"deletion guard rejected the scan: {absent} of {active} documents are missing"
            )


def metadata_source_hash(policy_name: str, fingerprint: str) -> str:
    """metadata 規則也算進指紋，切換 ACL 開關後的下一次掃描會重新判斷每份文件。"""

    return hashlib.sha256(f"{policy_name}\0{fingerprint}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Discovery:
    """掃描與單檔匯入共用的依賴：來源、metadata 指紋與 job 重試上限。"""

    source: DocumentSource
    metadata_source: MetadataSource
    policy_name: str
    max_attempts: int

    def metadata_hash(self, file: DiscoveredFile) -> str:
        return metadata_source_hash(
            self.policy_name, self.metadata_source.fingerprint(file.relative_path)
        )

    async def record(
        self,
        uow: UnitOfWork,
        existing: SourceItem | None,
        file: DiscoveredFile,
        metadata_hash: str,
        now: datetime,
    ) -> tuple[SourceItem, JobId | None, bool]:
        """保存觀察結果；需要處理時排 job。回傳 item、相關 job 與這次是否新排了 job。"""

        if existing is None:
            item = SourceItem.discovered(
                file.relative_path,
                source_version=file.source_version,
                metadata_source_hash=metadata_hash,
                file_size=file.file_size,
                modified_at=file.modified_at,
                at=now,
            )
            needs_processing = True
        else:
            item, needs_processing = existing.observe(
                source_version=file.source_version,
                metadata_source_hash=metadata_hash,
                file_size=file.file_size,
                modified_at=file.modified_at,
                at=now,
            )
        await uow.items.save(item)
        if not needs_processing:
            return item, None, False
        job = IngestionJob.process(item, max_attempts=self.max_attempts, now=now)
        return item, job.id, await uow.jobs.add_if_absent(job)


class ScanDocuments:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        discovery: Discovery,
        scan_lock: ScanLock,
        rules: DeletionRules,
        clock: Clock,
    ) -> None:
        self._uow_factory = uow_factory
        self._discovery = discovery
        self._scan_lock = scan_lock
        self._rules = rules
        self._clock = clock

    async def last_completed_at(self) -> datetime | None:
        """最後一次完成的完整掃描。被刪除護欄拒絕的掃描與單檔 `ingest` 都不算。"""

        async with self._uow_factory() as uow:
            return await uow.full_scans.last_completed_at()

    async def run(self, *, allow_mass_deletion: bool = False) -> ScanReport:
        async with self._scan_lock.hold():
            files = await asyncio.to_thread(self._discovery.source.scan)
            hashes = await asyncio.to_thread(
                lambda: {file.relative_path: self._discovery.metadata_hash(file) for file in files}
            )
            return await self._record(files, hashes, allow_mass_deletion=allow_mass_deletion)

    async def _record(
        self,
        files: tuple[DiscoveredFile, ...],
        hashes: dict[str, str],
        *,
        allow_mass_deletion: bool,
    ) -> ScanReport:
        now = self._clock()
        async with self._uow_factory() as uow:
            items = {item.relative_path: item for item in await uow.items.all()}
            active = {
                path for path, item in items.items() if item.status is not SourceItemStatus.DELETED
            }
            absent = sorted(active - {file.relative_path for file in files})
            if not allow_mass_deletion:
                self._rules.check(active=len(active), absent=len(absent), discovered=len(files))

            queued = 0
            for file in files:
                _, _, created = await self._discovery.record(
                    uow, items.get(file.relative_path), file, hashes[file.relative_path], now
                )
                queued += created

            deletions = 0
            for path in absent:
                item = items[path].mark_missing()
                await uow.items.save(item)
                if item.missing_scan_count >= self._rules.missing_scans_before_delete:
                    job = IngestionJob.delete(
                        item, max_attempts=self._discovery.max_attempts, now=now
                    )
                    deletions += await uow.jobs.add_if_absent(job)
            await uow.full_scans.record_completed(now)
            await uow.commit()
        return ScanReport(
            discovered=len(files), queued=queued, missing=len(absent), deletions_queued=deletions
        )


class IngestDocument:
    """只發現並排程一個檔案，其他檔案不會因此被當成消失。"""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        discovery: Discovery,
        scan_lock: ScanLock,
        clock: Clock,
    ) -> None:
        self._uow_factory = uow_factory
        self._discovery = discovery
        self._scan_lock = scan_lock
        self._clock = clock

    async def run(self, relative_path: str) -> IngestReport:
        async with self._scan_lock.hold():
            file = await asyncio.to_thread(self._discovery.source.discover, relative_path)
            if file is None:
                raise SourceFileNotFoundError(f"no supported document at {relative_path!r}")
            metadata_hash = await asyncio.to_thread(self._discovery.metadata_hash, file)
            async with self._uow_factory() as uow:
                existing = await uow.items.by_path(file.relative_path)
                item, job_id, queued = await self._discovery.record(
                    uow, existing, file, metadata_hash, self._clock()
                )
                await uow.commit()
        return IngestReport(item=item, job_id=job_id, queued=queued)
