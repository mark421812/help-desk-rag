"""完整的 ingestion：真的 PostgreSQL（alembic migration）與 Qdrant 容器，正式的 composition root。

只換掉 E5 model：Intel Mac 沒有 torch wheel，這裡用固定的雜湊向量與以空白分詞的 token 計數。
真正的 E5 在 Docker 映像裡驗證。
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import text
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.qdrant import QdrantContainer

from helpdesk.config import Settings
from helpdesk.domain.ingestion import SourceItemStatus
from helpdesk.infrastructure.vector_index.qdrant import PAYLOAD_INDEXES
from helpdesk.presentation.worker.composition import Worker, build_worker
from helpdesk.presentation.worker.runner import drain_jobs
from tests.ingestion_fakes import FakeEmbedder, WordCounter

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY = """# 請假規則

員工每年有七天特休，應事先提出申請並經主管核准後才能休假，未經核准視為曠職處理。

## 病假

病假需檢附醫院證明，連續三天以上需另附診斷書，並於復職後三日內補辦請假手續。
"""
SIDECAR = """title: 請假管理辦法
document_type: policy
owner_department: HR
document_owner: hr@example.com
status: published
external_llm_allowed: true
acl:
  - principal_type: group
    principal_id: staff
  - principal_type: user
    principal_id: intern@example.com
    effect: deny
"""


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="asyncpg") as postgres:
        url = postgres.get_connection_url()
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("DATABASE_URL", url)
            patch.setenv("LLM_PROVIDER", "fake")
            command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")
        yield url


@pytest.fixture(scope="module")
def qdrant_url() -> Iterator[str]:
    with QdrantContainer("qdrant/qdrant:v1.18.3-unprivileged") as qdrant:
        yield f"http://{qdrant.get_container_host_ip()}:{qdrant.get_exposed_port(6333)}"


WorkerFactory = Callable[..., Worker]


@pytest.fixture
async def make_worker(
    database_url: str, qdrant_url: str, tmp_path: Path
) -> AsyncIterator[WorkerFactory]:
    collection = f"test_{uuid4().hex[:12]}"
    workers: list[Worker] = []

    def make(*, access_control: bool = False) -> Worker:
        settings = Settings(
            _env_file=None,
            llm_provider="fake",
            database_url=database_url,
            qdrant_url=qdrant_url,
            qdrant_collection=collection,
            documents_root=tmp_path,
            access_control_enabled=access_control,
            worker_retry_base_seconds=0.01,
        )
        worker = build_worker(settings, embedder=FakeEmbedder(), token_counter=WordCounter())
        workers.append(worker)
        return worker

    yield make
    cleaner = workers[0] if workers else make()
    async with cleaner.engine.begin() as connection:
        await connection.execute(
            text("TRUNCATE helpdesk.source_items, helpdesk.last_full_scan CASCADE")
        )
    for worker in workers:
        await worker.close()


async def ingest(worker: Worker) -> int:
    await worker.scan.run()
    return await drain_jobs(worker, "integration", asyncio.Event(), until_idle=True)


async def rows(worker: Worker, sql: str, **params: object) -> list[Any]:
    async with worker.engine.connect() as connection:
        return list((await connection.execute(text(sql), params)).all())


async def points(worker: Worker) -> list[models.Record]:
    client = AsyncQdrantClient(url=worker.settings.qdrant_url)
    try:
        records, _ = await client.scroll(
            worker.settings.qdrant_collection, limit=1_000, with_payload=True
        )
        return records
    finally:
        await client.close()


def write(root: Path, name: str, content: str, *, mtime: int) -> None:
    path = root / name
    path.write_text(content, encoding="utf-8")
    os.utime(path, (mtime, mtime))


async def test_a_document_is_written_to_postgres_and_qdrant_with_its_access_rules(
    make_worker: WorkerFactory, tmp_path: Path
) -> None:
    worker = make_worker()
    write(tmp_path, "leave.md", POLICY, mtime=1_700_000_000)
    (tmp_path / "leave.md.metadata.yaml").write_text(SIDECAR, encoding="utf-8")

    assert await ingest(worker) == 1

    [(status, title, version_id)] = await rows(
        worker,
        "SELECT i.status, d.title, d.current_version_id FROM helpdesk.source_items i "
        "JOIN helpdesk.documents d ON d.source_item_id = i.id",
    )
    rules = await rows(
        worker,
        "SELECT principal_type, principal_id, effect FROM helpdesk.document_access_rules "
        "ORDER BY principal_type",
    )
    chunks = await rows(
        worker,
        "SELECT id, section_path, indexing_status, qdrant_point_id FROM helpdesk.chunks "
        "WHERE version_id = :version ORDER BY chunk_index",
        version=version_id,
    )
    stored = await points(worker)
    client = AsyncQdrantClient(url=worker.settings.qdrant_url)
    info = await client.get_collection(worker.settings.qdrant_collection)
    await client.close()

    assert (status, title) == (SourceItemStatus.COMPLETED.value, "請假管理辦法")
    assert rules == [("group", "staff", "allow"), ("user", "intern@example.com", "deny")]
    assert chunks and all(state == "indexed" and point == cid for cid, _, state, point in chunks)
    assert {section for _, section, _, _ in chunks} == {"請假規則", "請假規則 > 病假"}
    assert sorted(str(record.id) for record in stored) == sorted(str(cid) for cid, *_ in chunks)
    payload = stored[0].payload or {}
    assert {key: payload[key] for key in ("is_current", "status", "external_llm_allowed")} == {
        "is_current": True,
        "status": "published",
        "external_llm_allowed": True,
    }
    assert (payload["acl_public"], payload["allowed_group_ids"], payload["denied_user_ids"]) == (
        False,
        ["staff"],
        ["intern@example.com"],
    )
    assert payload["version_id"] == str(version_id)
    assert set(info.payload_schema) == set(PAYLOAD_INDEXES)


async def test_running_again_converges_without_duplicate_points(
    make_worker: WorkerFactory, tmp_path: Path
) -> None:
    worker = make_worker()
    write(tmp_path, "leave.md", POLICY, mtime=1_700_000_000)
    await ingest(worker)
    before = sorted(str(record.id) for record in await points(worker))
    [(version_id,)] = await rows(worker, "SELECT id FROM helpdesk.document_versions")

    assert await ingest(worker) == 0
    os.utime(tmp_path / "leave.md", (1_800_000_000, 1_800_000_000))
    assert await ingest(worker) == 1
    second_worker = make_worker()
    assert await ingest(second_worker) == 0

    assert sorted(str(record.id) for record in await points(worker)) == before
    assert await rows(worker, "SELECT id FROM helpdesk.document_versions") == [(version_id,)]
    assert await rows(worker, "SELECT count(*) FROM helpdesk.chunks") == [(len(before),)]


async def test_a_changed_file_replaces_the_version_and_its_points(
    make_worker: WorkerFactory, tmp_path: Path
) -> None:
    worker = make_worker()
    write(tmp_path, "leave.md", POLICY, mtime=1_700_000_000)
    await ingest(worker)
    [(old_version,)] = await rows(worker, "SELECT id FROM helpdesk.document_versions")

    revised = POLICY.replace("七天", "十天") + "\n補充條款。\n"
    write(tmp_path, "leave.md", revised, mtime=1_800_000_000)
    await ingest(worker)

    [(new_version, chunk_count)] = await rows(
        worker, "SELECT id, chunk_count FROM helpdesk.document_versions"
    )
    [(current,)] = await rows(worker, "SELECT current_version_id FROM helpdesk.documents")
    stored = await points(worker)
    assert new_version != old_version and current == new_version
    assert len(stored) == chunk_count
    assert {(record.payload or {})["version_id"] for record in stored} == {str(new_version)}
    assert all((record.payload or {})["is_current"] for record in stored)


async def test_acl_on_keeps_documents_without_metadata_out_of_the_index(
    make_worker: WorkerFactory, tmp_path: Path
) -> None:
    write(tmp_path, "BookRAG.md", POLICY, mtime=1_700_000_000)
    open_worker = make_worker(access_control=False)
    await ingest(open_worker)
    assert len(await points(open_worker)) > 0

    acl_worker = make_worker(access_control=True)
    await ingest(acl_worker)

    assert await rows(acl_worker, "SELECT status, error_code FROM helpdesk.source_items") == [
        ("metadata_missing", "metadata_missing")
    ]
    assert await rows(acl_worker, "SELECT count(*) FROM helpdesk.documents") == [(0,)]
    assert await points(acl_worker) == []
    items = await acl_worker.items.run()
    assert items[0].latest_job is not None
    assert items[0].latest_job.error_code == "metadata_missing"


async def test_a_file_missing_from_two_scans_is_deleted_everywhere(
    make_worker: WorkerFactory, tmp_path: Path
) -> None:
    worker = make_worker()
    for index, name in enumerate(("a.md", "b.md", "c.md", "d.md", "e.md")):
        write(tmp_path, name, POLICY + f"\n第 {index} 份。\n", mtime=1_700_000_000)
    await ingest(worker)
    (tmp_path / "a.md").unlink()

    await ingest(worker)
    after_one = await rows(worker, "SELECT count(*) FROM helpdesk.documents")
    await ingest(worker)

    assert after_one == [(5,)]
    assert await rows(
        worker,
        "SELECT status, missing_scan_count FROM helpdesk.source_items WHERE relative_path = 'a.md'",
    ) == [("deleted", 2)]
    assert await rows(worker, "SELECT count(*) FROM helpdesk.documents") == [(4,)]
    [(chunks,)] = await rows(worker, "SELECT count(*) FROM helpdesk.chunks")
    assert len(await points(worker)) == chunks


async def test_only_a_full_scan_moves_the_last_full_scan_time(
    make_worker: WorkerFactory, tmp_path: Path
) -> None:
    worker = make_worker()
    write(tmp_path, "a.md", POLICY, mtime=1_700_000_000)

    await worker.ingest.run("a.md")
    after_ingest = await worker.scan.last_completed_at()
    await worker.scan.run()
    first = await worker.scan.last_completed_at()
    await worker.scan.run()
    second = await worker.scan.last_completed_at()
    await worker.ingest.run("a.md")

    assert after_ingest is None
    assert first is not None and second is not None and first < second
    assert await worker.scan.last_completed_at() == second
    assert await rows(worker, "SELECT count(*) FROM helpdesk.last_full_scan") == [(1,)]
