"""helpdesk-api 的背景回合：有並行上限的 in-process task，以及定期回收中斷的回合。

提問的 HTTP 請求只負責接受回合，處理交給這裡。同時處理的回合數由 semaphore 限制，其餘的在
accepted 狀態排隊。回收在啟動時跑一次，之後每分鐘一次，把超過門檻沒有進展的回合標成
`failed_unknown`；回收以資料庫狀態為準，所以重跑或多個程序同時跑都只會處理一次。關閉時取消
處理中的回合，`ProcessTurn` 會把它們標成 `failed_unknown`；還在排隊的回合留在 accepted，超過
門檻後由回收處理。log 只記代碼、數量與耗時。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import Any, Final

import structlog

from helpdesk.application.conversation.ask import ProcessTurn, RecoverStaleTurns
from helpdesk.application.llm import TokenUsage
from helpdesk.domain.access import Principal
from helpdesk.domain.conversations import Turn

logger = structlog.get_logger(__name__)

RECOVERY_INTERVAL_SECONDS: Final = 60.0
SHUTDOWN_GRACE_SECONDS: Final = 5.0


class TurnRunner:
    def __init__(
        self,
        process: ProcessTurn,
        recover: RecoverStaleTurns,
        *,
        max_concurrency: int,
        recovery_interval_seconds: float = RECOVERY_INTERVAL_SECONDS,
    ) -> None:
        self._process = process
        self._recover = recover
        self._slots = asyncio.Semaphore(max_concurrency)
        self._recovery_interval_seconds = recovery_interval_seconds
        self._tasks: set[asyncio.Task[None]] = set()

    def start(self) -> None:
        self._spawn(self._recover_forever())

    def submit(self, turn: Turn, principal: Principal) -> None:
        self._spawn(self._process_turn(turn, principal))

    async def close(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=SHUTDOWN_GRACE_SECONDS)

    def _spawn(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _process_turn(self, turn: Turn, principal: Principal) -> None:
        async with self._slots:
            started = time.perf_counter()
            try:
                report = await self._process.run(turn, principal)
            except Exception as error:
                logger.error(
                    "turn_processing_failed", turn_id=str(turn.id), error_type=type(error).__name__
                )
                return
        final = report.turn
        logger.info(
            "turn_processed",
            turn_id=str(final.id),
            conversation_id=str(final.conversation_id),
            state=final.state.value,
            outcome=final.result.outcome.value if final.result is not None else None,
            error_code=final.error_code,
            error_type=report.error_type,
            stopped=report.stopped,
            rewrite=report.rewrite.value if report.rewrite is not None else None,
            rewrite_detail=report.rewrite_detail,
            sendable=report.sendable,
            restricted=report.restricted,
            keyword_failure=report.keyword_failure,
            scan_rules=list(report.scan_rules),
            **_usage_fields("rewrite", report.rewrite_usage),
            **_usage_fields("answer", report.answer_usage),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    async def _recover_forever(self) -> None:
        while True:
            try:
                recovered = await self._recover.run()
            except Exception as error:
                logger.error("stale_turn_recovery_failed", error_type=type(error).__name__)
            else:
                if recovered:
                    logger.warning("stale_turns_recovered", count=recovered)
            await asyncio.sleep(self._recovery_interval_seconds)


def _usage_fields(step: str, usage: TokenUsage | None) -> dict[str, int | None]:
    usage = usage or TokenUsage()
    return {
        f"{step}_prompt_tokens": usage.prompt,
        f"{step}_completion_tokens": usage.completion,
        f"{step}_cached_prompt_tokens": usage.cached_prompt,
    }
