"""`helpdesk-worker` 指令列。

`run` 是長駐的 worker（job 迴圈加每日掃描），`run --until-idle` 處理完佇列就結束。
`scan`、`ingest`、`items`、`retry` 只排程或查詢，不載入 embedding model。輸出是 JSON，
domain 錯誤以結束碼 2 回報。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from helpdesk.config import Settings
from helpdesk.domain.errors import DomainError
from helpdesk.domain.ingestion import JobId, SourceItemStatus
from helpdesk.infrastructure.logging import configure_logging
from helpdesk.presentation.worker.composition import Worker, build_worker
from helpdesk.presentation.worker.runner import run_worker


def _limit(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 1_000:
        raise argparse.ArgumentTypeError("limit must be between 1 and 1000")
    return number


def _offset(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("offset must not be negative")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helpdesk-worker", description="Help Desk ingestion")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="process jobs and run the daily scan")
    run.add_argument(
        "--until-idle", action="store_true", help="exit once no job is ready instead of waiting"
    )

    scan = commands.add_parser("scan", help="scan DOCUMENTS_ROOT and queue changed files")
    scan.add_argument(
        "--allow-mass-deletion",
        action="store_true",
        help="bypass the missing-ratio guard for this scan",
    )

    ingest = commands.add_parser("ingest", help="queue one file by its path under DOCUMENTS_ROOT")
    ingest.add_argument("relative_path")

    items = commands.add_parser("items", help="list tracked files and their latest job")
    items.add_argument("--status", choices=[status.value for status in SourceItemStatus])
    items.add_argument("--offset", type=_offset, default=0)
    items.add_argument("--limit", type=_limit, default=100)

    retry = commands.add_parser("retry", help="requeue a failed job")
    retry.add_argument("job_id", type=UUID)
    return parser


async def _execute(worker: Worker, args: argparse.Namespace) -> object:
    match args.command:
        case "run":
            return {"jobs_handled": await run_worker(worker, until_idle=args.until_idle)}
        case "scan":
            return await worker.scan.run(allow_mass_deletion=args.allow_mass_deletion)
        case "ingest":
            return await worker.ingest.run(args.relative_path)
        case "items":
            status = SourceItemStatus(args.status) if args.status else None
            return await worker.items.run(status=status, offset=args.offset, limit=args.limit)
        case "retry":
            return await worker.retry.run(JobId(args.job_id))
    raise AssertionError(f"unhandled command {args.command}")


async def _main(argv: Sequence[str] | None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level)
    worker = build_worker(settings)
    try:
        result = await _execute(worker, args)
    except DomainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        await worker.close()
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID | date | datetime):
        return str(value)
    return value


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(asyncio.run(_main(argv)))
