"""ingestion-worker 的長駐迴圈：一個 job 迴圈加上每日掃描排程，收到 SIGTERM 後收尾結束。"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import structlog

from helpdesk.domain.ingestion import LeaseLostError
from helpdesk.presentation.worker.composition import Worker, utc_now

logger = structlog.get_logger(__name__)


def worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class DailySchedule:
    hour: int
    minute: int
    timezone: str

    def _today(self, now: datetime) -> tuple[datetime, datetime]:
        local_now = now.astimezone(ZoneInfo(self.timezone))
        scheduled = datetime.combine(
            local_now.date(), time(self.hour, self.minute), tzinfo=local_now.tzinfo
        )
        return local_now, scheduled

    def next_run(self, now: datetime) -> datetime:
        local_now, scheduled = self._today(now)
        return scheduled if scheduled > local_now else scheduled + timedelta(days=1)

    def missed(self, last_scan: datetime | None, now: datetime) -> bool:
        """今天的排程時間已過，而最後一次成功掃描早於它，啟動時就補掃一次。"""

        local_now, scheduled = self._today(now)
        return local_now >= scheduled and (last_scan is None or last_scan < scheduled)


async def wait_or_stop(stop: asyncio.Event, seconds: float) -> bool:
    """等待指定秒數；期間收到停止訊號就提早回傳 True。"""

    try:
        await asyncio.wait_for(stop.wait(), timeout=max(seconds, 0))
    except TimeoutError:
        return False
    return True


async def drain_jobs(worker: Worker, owner: str, stop: asyncio.Event, *, until_idle: bool) -> int:
    """一次處理一個 job。`until_idle` 時佇列沒有可執行的 job 就結束，回傳處理了幾個。"""

    idle_wait = worker.settings.worker_poll_interval_seconds
    handled = 0
    while not stop.is_set():
        try:
            job = await worker.claim.run(owner)
            if job is None:
                if until_idle or await wait_or_stop(stop, idle_wait):
                    break
                continue
            result = await worker.process.run(job.id, owner)
        except LeaseLostError:
            logger.warning("ingestion_job_lease_lost")
            continue
        except Exception:
            if until_idle:
                raise
            logger.exception("ingestion_worker_iteration_failed")
            await wait_or_stop(stop, idle_wait)
            continue
        handled += 1
        logger.info(
            "ingestion_job_finished",
            job_id=str(result.job.id),
            job_type=result.job.job_type.value,
            job_status=result.job.status.value,
            item=result.item.relative_path,
            item_status=result.item.status.value,
            error_code=result.job.error_code,
        )
    return handled


async def scheduled_scans(worker: Worker, stop: asyncio.Event) -> None:
    settings = worker.settings
    schedule = DailySchedule(
        settings.schedule_hour, settings.schedule_minute, settings.schedule_timezone
    )
    if schedule.missed(await worker.scan.last_completed_at(), utc_now()):
        await _scan(worker)
    while not stop.is_set():
        due = schedule.next_run(utc_now())
        logger.info("next_scan_scheduled", at=due.isoformat())
        if await wait_or_stop(stop, (due - utc_now()).total_seconds()):
            return
        await _scan(worker)


async def _scan(worker: Worker) -> None:
    try:
        report = await worker.scan.run()
    except Exception:
        logger.exception("scheduled_scan_failed")
        return
    logger.info(
        "scheduled_scan_finished",
        discovered=report.discovered,
        queued=report.queued,
        missing=report.missing,
        deletions_queued=report.deletions_queued,
    )


async def run_worker(worker: Worker, *, until_idle: bool) -> int:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    owner = worker_id()
    logger.info("ingestion_worker_started", worker_id=owner, until_idle=until_idle)
    if until_idle:
        return await drain_jobs(worker, owner, stop, until_idle=True)
    scheduler = asyncio.create_task(scheduled_scans(worker, stop))
    try:
        return await drain_jobs(worker, owner, stop, until_idle=False)
    finally:
        stop.set()
        await scheduler
        logger.info("ingestion_worker_stopped", worker_id=owner)
