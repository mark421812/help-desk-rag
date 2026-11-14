"""CLI 用的查詢與手動重試。"""

from __future__ import annotations

from dataclasses import dataclass

from helpdesk.application.ingestion.ports import Clock, UnitOfWorkFactory
from helpdesk.domain.ingestion import (
    IngestionJob,
    JobId,
    JobNotFoundError,
    SourceItem,
    SourceItemStatus,
)


@dataclass(frozen=True, slots=True)
class ItemView:
    item: SourceItem
    latest_job: IngestionJob | None


class ListItems:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def run(
        self, *, status: SourceItemStatus | None = None, offset: int = 0, limit: int = 100
    ) -> list[ItemView]:
        async with self._uow_factory() as uow:
            items = await uow.items.page(status=status, offset=offset, limit=limit)
            jobs = await uow.jobs.latest_by_item([item.id for item in items])
        return [ItemView(item=item, latest_job=jobs.get(item.id)) for item in items]


class RetryJob:
    """把最終失敗的 job 重設為待處理；檔案也回到 pending。"""

    def __init__(self, uow_factory: UnitOfWorkFactory, clock: Clock) -> None:
        self._uow_factory = uow_factory
        self._clock = clock

    async def run(self, job_id: JobId) -> IngestionJob:
        async with self._uow_factory() as uow:
            job = await uow.jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(f"job {job_id} was not found")
            job = job.retry_manually(self._clock())
            item = await uow.items.get(job.source_item_id)
            if item is not None and item.status not in {
                SourceItemStatus.DELETED,
                SourceItemStatus.PROCESSING,
            }:
                await uow.items.save(item.requeue())
            await uow.jobs.save(job)
            await uow.commit()
        return job
