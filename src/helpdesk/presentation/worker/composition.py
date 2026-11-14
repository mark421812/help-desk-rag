"""ingestion-worker 的 composition root。

`Settings` 在這裡讀一次，選定實作後注入 use case。ACL 開關在 ingestion 只決定一件事：
metadata 是選填還是必填（`OptionalMetadata` 或 `RequiredMetadata`）。程式其他地方都不讀開關。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from helpdesk.application.ingestion.admin import ListItems, RetryJob
from helpdesk.application.ingestion.metadata import OptionalMetadata, RequiredMetadata
from helpdesk.application.ingestion.ports import MetadataPolicy, PassageEmbedder, UnitOfWork
from helpdesk.application.ingestion.processing import ClaimNextJob, ProcessingLimits, ProcessJob
from helpdesk.application.ingestion.scanning import (
    DeletionRules,
    Discovery,
    IngestDocument,
    ScanDocuments,
)
from helpdesk.config import Settings
from helpdesk.infrastructure.chunking.structure_aware import (
    ChunkingConfig,
    StructureAwareChunker,
    TokenCounter,
)
from helpdesk.infrastructure.embedding.e5 import E5Encoder
from helpdesk.infrastructure.metadata.yaml_sidecars import YamlSidecarMetadata
from helpdesk.infrastructure.parsers.registry import default_parsers
from helpdesk.infrastructure.persistence.ingestion import (
    PostgresScanLock,
    SqlUnitOfWork,
    session_factory,
)
from helpdesk.infrastructure.sources.local_folder import LocalFolderSource
from helpdesk.infrastructure.vector_index.qdrant import QdrantVectorIndex


def metadata_policy(settings: Settings) -> MetadataPolicy:
    return RequiredMetadata() if settings.access_control_enabled else OptionalMetadata()


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Worker:
    settings: Settings
    scan: ScanDocuments
    ingest: IngestDocument
    claim: ClaimNextJob
    process: ProcessJob
    retry: RetryJob
    items: ListItems
    engine: AsyncEngine
    index: QdrantVectorIndex

    async def close(self) -> None:
        await self.index.close()
        await self.engine.dispose()


def build_worker(
    settings: Settings,
    *,
    embedder: PassageEmbedder | None = None,
    token_counter: TokenCounter | None = None,
) -> Worker:
    """測試可以換掉 embedding model 與 token 計數，其餘都是正式的 adapter。"""

    encoder = E5Encoder(
        settings.embedding_model_id,
        settings.embedding_model_revision,
        max_sequence_length=settings.embedding_max_sequence_length,
        batch_size=settings.embedding_batch_size,
    )
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    sessions = session_factory(engine)

    def unit_of_work() -> UnitOfWork:
        return SqlUnitOfWork(sessions)

    parsers = default_parsers()
    source = LocalFolderSource(settings.documents_root, parsers.supported_extensions)
    metadata_source = YamlSidecarMetadata(settings.documents_root)
    policy = metadata_policy(settings)
    index = QdrantVectorIndex(
        AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=(
                settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            ),
            timeout=60,
        ),
        settings.qdrant_collection,
    )
    discovery = Discovery(
        source=source,
        metadata_source=metadata_source,
        policy_name=policy.name,
        max_attempts=settings.worker_max_attempts,
    )
    scan_lock = PostgresScanLock(engine)
    chunker = StructureAwareChunker(
        token_counter or encoder,
        ChunkingConfig(
            target_tokens=settings.chunk_target_tokens,
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            model_max_tokens=settings.embedding_max_sequence_length,
        ),
    )
    return Worker(
        settings=settings,
        scan=ScanDocuments(
            unit_of_work,
            discovery,
            scan_lock,
            DeletionRules(
                missing_scans_before_delete=settings.missing_scans_before_delete,
                max_missing_ratio=settings.max_missing_ratio,
            ),
            utc_now,
        ),
        ingest=IngestDocument(unit_of_work, discovery, scan_lock, utc_now),
        claim=ClaimNextJob(unit_of_work, timedelta(seconds=settings.worker_lease_seconds), utc_now),
        process=ProcessJob(
            unit_of_work,
            source=source,
            parser=parsers,
            chunker=chunker,
            embedder=embedder or encoder,
            metadata_source=metadata_source,
            metadata_policy=policy,
            index=index,
            limits=ProcessingLimits(
                max_file_size_bytes=settings.max_file_size_bytes,
                retry_base_delay=timedelta(seconds=settings.worker_retry_base_seconds),
            ),
            clock=utc_now,
        ),
        retry=RetryJob(unit_of_work, utc_now),
        items=ListItems(unit_of_work),
        engine=engine,
        index=index,
    )
