"""retrieval 的真實儲存：PostgreSQL 關鍵字 SQL、Qdrant 預先過濾、PostgreSQL 複驗。

文件經由正式的 ingestion-worker 寫入（只換掉 E5 model），查詢走正式的 adapter 與 use case。
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.qdrant import QdrantContainer

from helpdesk.application.retrieval.access import AccessPolicy, AclAccess, OpenAccess
from helpdesk.application.retrieval.candidates import CandidatePool, FindCandidates
from helpdesk.application.retrieval.keywords import keyword_query
from helpdesk.application.retrieval.ports import ReadableBy
from helpdesk.config import Settings
from helpdesk.domain.access import Principal
from helpdesk.domain.documents import ChunkId
from helpdesk.infrastructure.persistence.ingestion import session_factory
from helpdesk.infrastructure.persistence.retrieval import PostgresKeywordSearch, SqlChunkCatalog
from helpdesk.infrastructure.vector_index.qdrant import QdrantVectorSearch
from helpdesk.presentation.worker.composition import Worker, build_worker
from helpdesk.presentation.worker.runner import drain_jobs
from tests.ingestion_fakes import FakeEmbedder, WordCounter
from tests.retrieval_fakes import TODAY

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALICE = Principal(user_id="alice@example.com", group_ids=("staff",))
INTERN = Principal(user_id="intern@example.com", group_ids=("staff",))
OUTSIDER = Principal(user_id="guest@example.com")

DOCUMENTS = {
    "public": "status: published\n",
    "staff": (
        "status: published\nacl:\n"
        "  - {principal_type: group, principal_id: staff}\n"
        "  - {principal_type: user, principal_id: intern@example.com, effect: deny}\n"
    ),
    "draft": "status: draft\n",
    "expired": "status: published\nexpires_at: 2020-01-01\n",
    "future": "status: published\neffective_at: 2999-01-01\n",
    "private": "status: published\nexternal_llm_allowed: false\n",
}


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


@dataclass
class Stores:
    worker: Worker
    vectors: QdrantVectorSearch
    keywords: PostgresKeywordSearch
    catalog: SqlChunkCatalog
    names: dict[ChunkId, str]
    positions: dict[ChunkId, tuple[UUID, int]]

    async def find(self, access: AccessPolicy, principal: Principal) -> CandidatePool:
        search = FindCandidates(
            embedder=FixedQuery(),
            vectors=self.vectors,
            keywords=self.keywords,
            catalog=self.catalog,
            keyword_min_score=3,
        )
        return await search.run(
            "報名時間",
            scope=access.vector_scope(principal),
            admit=lambda stored: access.admits(stored, principal, TODAY),
            limit=24,
        )

    def documents(self, chunk_ids: list[ChunkId]) -> set[str]:
        return {self.names[chunk_id] for chunk_id in chunk_ids}


class FixedQuery:
    def embed_query(self, text: str) -> list[float]:
        return [0.5, 0.5, 0.5, 0.5]


@pytest.fixture
async def stores(database_url: str, qdrant_url: str, tmp_path: Path) -> AsyncIterator[Stores]:
    settings = Settings(
        _env_file=None,
        llm_provider="fake",
        database_url=database_url,
        qdrant_url=qdrant_url,
        qdrant_collection=f"test_{uuid4().hex[:12]}",
        documents_root=tmp_path,
        access_control_enabled=False,
    )
    for name, sidecar in DOCUMENTS.items():
        path = tmp_path / f"{name}.md"
        path.write_text(
            f"# {name}\n\n{name} 文件說明報名時間與報名資格，代碼 HR2024。\n", encoding="utf-8"
        )
        os.utime(path, (1_700_000_000, 1_700_000_000))
        (tmp_path / f"{name}.md.metadata.yaml").write_text(
            f"title: {name}\n{sidecar}", encoding="utf-8"
        )
    worker = build_worker(settings, embedder=FakeEmbedder(), token_counter=WordCounter())
    await worker.scan.run()
    await drain_jobs(worker, "integration", asyncio.Event(), until_idle=True)
    sessions = session_factory(worker.engine)
    async with worker.engine.connect() as connection:
        rows = (
            await connection.execute(
                text(
                    "SELECT c.id, d.title, c.document_id, c.chunk_index FROM helpdesk.chunks c "
                    "JOIN helpdesk.documents d ON d.id = c.document_id"
                )
            )
        ).all()
    vectors = QdrantVectorSearch(AsyncQdrantClient(url=qdrant_url), settings.qdrant_collection)
    yield Stores(
        worker=worker,
        vectors=vectors,
        keywords=PostgresKeywordSearch(sessions),
        catalog=SqlChunkCatalog(sessions),
        names={ChunkId(row[0]): row[1] for row in rows},
        positions={ChunkId(row[0]): (row[2], row[3]) for row in rows},
    )
    async with worker.engine.begin() as connection:
        await connection.execute(text("TRUNCATE helpdesk.source_items CASCADE"))
    await vectors.close()
    await worker.close()


async def test_keyword_search_scores_by_term_weight_and_escapes_like_wildcards(
    stores: Stores,
) -> None:
    hits = await stores.keywords.search(keyword_query("報名時間", 3), limit=50, offset=0)

    assert stores.documents([hit.chunk_id for hit in hits]) == set(DOCUMENTS)
    # 報名時間(4) + 報名(2) + 名時(2) + 時間(2)
    assert {hit.score for hit in hits} == {10.0}
    order = [stores.positions[hit.chunk_id] for hit in hits]
    assert order == sorted(order)
    assert await stores.keywords.search(keyword_query("報名時間", 11), limit=50, offset=0) == []
    assert await stores.keywords.search(keyword_query("hr_024", 3), limit=50, offset=0) == []
    assert await stores.keywords.search(keyword_query("hr2024", 3), limit=50, offset=0)
    page = await stores.keywords.search(keyword_query("報名時間", 3), limit=2, offset=2)
    assert [hit.chunk_id for hit in page] == [hit.chunk_id for hit in hits[2:4]]


async def test_keyword_ties_follow_document_then_chunk_order_not_chunk_uuids(
    stores: Stores,
) -> None:
    first, second = UUID(int=0xA), UUID(int=0xB)
    # chunk UUID 的順序刻意和（文件、段落）順序相反。
    chunks = {
        (first, 0): UUID(int=0xF4),
        (first, 1): UUID(int=0xF3),
        (second, 0): UUID(int=0xF2),
        (second, 1): UUID(int=0xF1),
    }
    async with stores.worker.engine.begin() as connection:
        for document in (first, second):
            item, version = uuid4(), uuid4()
            await connection.execute(
                text(
                    "INSERT INTO helpdesk.source_items (id, relative_path, source_version, "
                    "metadata_source_hash, file_size, modified_at, status) VALUES (:id, :path, "
                    "'mtime:1;size:1', :hash, 1, now(), 'completed')"
                ),
                {"id": item, "path": f"tie-{document}.md", "hash": "a" * 64},
            )
            await connection.execute(
                text(
                    "INSERT INTO helpdesk.documents (id, source_item_id, title, document_type, "
                    "language, status, confidentiality, external_llm_allowed) VALUES (:id, :item, "
                    "'同分', 'general', 'zh-TW', 'published', 'internal', true)"
                ),
                {"id": document, "item": item},
            )
            await connection.execute(
                text(
                    "INSERT INTO helpdesk.document_versions (id, document_id, content_hash, "
                    "mime_type, file_size, parser_name, parser_version, embedding_model, "
                    "embedding_model_revision, embedding_dimension, chunk_count, indexing_status) "
                    "VALUES (:id, :document, :hash, 'text/markdown', 1, 'md', '1', 'e5', 'rev', "
                    "4, 2, 'indexed')"
                ),
                {"id": version, "document": document, "hash": "b" * 64},
            )
            for index in (0, 1):
                await connection.execute(
                    text(
                        "INSERT INTO helpdesk.chunks (id, document_id, version_id, chunk_index, "
                        "content, content_hash, token_count, qdrant_point_id, indexing_status) "
                        "VALUES (:id, :document, :version, :index, '同分詞彙', :hash, 3, :id, "
                        "'indexed')"
                    ),
                    {
                        "id": chunks[(document, index)],
                        "document": document,
                        "version": version,
                        "index": index,
                        "hash": "c" * 64,
                    },
                )
            await connection.execute(
                text("UPDATE helpdesk.documents SET current_version_id = :v WHERE id = :d"),
                {"v": version, "d": document},
            )

    hits = await stores.keywords.search(keyword_query("同分詞彙", 3), limit=50, offset=0)
    page = await stores.keywords.search(keyword_query("同分詞彙", 3), limit=2, offset=1)

    assert len({hit.score for hit in hits}) == 1
    assert [hit.chunk_id for hit in hits] == [chunks[key] for key in sorted(chunks)]
    assert [hit.chunk_id for hit in page] == [chunks[(first, 1)], chunks[(second, 0)]]


async def test_keyword_search_only_returns_the_current_version(stores: Stores) -> None:
    async with stores.worker.engine.begin() as connection:
        await connection.execute(
            text("UPDATE helpdesk.documents SET current_version_id = NULL WHERE title = 'public'")
        )

    hits = await stores.keywords.search(keyword_query("報名時間", 3), limit=50, offset=0)
    pool = await stores.find(OpenAccess(), ALICE)

    assert "public" not in stores.documents([hit.chunk_id for hit in hits])
    assert "public" not in stores.documents([c.chunk_id for c in pool.candidates])


async def test_the_qdrant_pre_filter_applies_status_and_acl(stores: Stores) -> None:
    async def visible(principal: Principal) -> set[str]:
        hits = await stores.vectors.search(
            [0.5, 0.5, 0.5, 0.5], ReadableBy(principal), limit=50, offset=0
        )
        return stores.documents([hit.chunk_id for hit in hits])

    dated = {"public", "expired", "future", "private"}
    assert await visible(ALICE) == dated | {"staff"}
    assert await visible(INTERN) == dated
    assert await visible(OUTSIDER) == dated


@pytest.mark.parametrize(
    ("access", "principal", "expected"),
    [
        (AclAccess(), ALICE, {"public", "staff", "private"}),
        (AclAccess(), INTERN, {"public", "private"}),
        (AclAccess(), OUTSIDER, {"public", "private"}),
        (OpenAccess(), INTERN, set(DOCUMENTS)),
    ],
)
async def test_both_sides_agree_after_the_postgres_post_check(
    stores: Stores, access: AccessPolicy, principal: Principal, expected: set[str]
) -> None:
    pool = await stores.find(access, principal)

    assert pool.keyword_failure is None
    assert stores.documents([c.chunk_id for c in pool.candidates]) == expected
    assert all(c.vector_score is not None and c.keyword_score is not None for c in pool.candidates)


async def test_the_catalog_returns_documents_with_their_access_rules(stores: Stores) -> None:
    staff = [chunk_id for chunk_id, name in stores.names.items() if name == "staff"]

    [loaded] = await stores.catalog.load(staff)

    assert loaded.is_current
    assert {rule.principal_id for rule in loaded.document.metadata.access_rules} == {
        "staff",
        "intern@example.com",
    }
    assert await stores.catalog.is_ready()
    assert await stores.vectors.is_ready()
