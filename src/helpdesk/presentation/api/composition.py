"""helpdesk-api 的 composition root。

`Settings` 在這裡讀一次。兩個開關只在 `policies` 判斷：`ACCESS_CONTROL_ENABLED` 決定
`AclAccess` 或 `OpenAccess`，以及開發用身分 header 或瀏覽器匿名 ID；
`SENSITIVE_DATA_SCAN_ENABLED` 決定 `RuleBasedOutboundScan` 或 `NoOutboundScan`，同一個掃描器
同時給追問改寫與回答。選好的實作注入 use case，程式其他地方不讀開關。`GROQ_API_KEY` 也在
這裡檢查，所以只有 helpdesk-api 需要它。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

from helpdesk.application.answering.generate import AnswerLimits, GenerateAnswer
from helpdesk.application.conversation.ask import ProcessTurn, RecoverStaleTurns, SubmitTurn
from helpdesk.application.conversation.manage import ManageConversations
from helpdesk.application.identity import IdentityResolver
from helpdesk.application.llm import LlmProvider
from helpdesk.application.outbound_scan import OutboundScan
from helpdesk.application.retrieval.access import AccessPolicy, AclAccess, OpenAccess
from helpdesk.application.retrieval.assembly import AssemblyLimits
from helpdesk.application.retrieval.candidates import FindCandidates
from helpdesk.application.retrieval.ports import QueryEmbedder
from helpdesk.application.retrieval.retrieve import RetrieveEvidence
from helpdesk.application.retrieval.understanding import RewriteLimits, UnderstandQuestion
from helpdesk.config import Settings
from helpdesk.infrastructure.embedding.e5 import E5Encoder
from helpdesk.infrastructure.identity import AnonymousIdentity, DevelopmentHeaderIdentity
from helpdesk.infrastructure.llm.fake import FakeLlmProvider
from helpdesk.infrastructure.llm.groq import GroqChatProvider
from helpdesk.infrastructure.outbound_scan.scanners import NoOutboundScan, RuleBasedOutboundScan
from helpdesk.infrastructure.persistence.conversations import SqlConversations
from helpdesk.infrastructure.persistence.ingestion import session_factory
from helpdesk.infrastructure.persistence.retrieval import PostgresKeywordSearch, SqlChunkCatalog
from helpdesk.infrastructure.vector_index.qdrant import QdrantVectorSearch
from helpdesk.presentation.api.background import TurnRunner


class ProductionIdentityMissingError(RuntimeError):
    """ACL 開啟、APP_ENV=production，但還沒有正式登入可用。"""


class MissingGroqKeyError(RuntimeError):
    """LLM_PROVIDER=groq 卻沒有 GROQ_API_KEY。"""


@dataclass(frozen=True, slots=True)
class Policies:
    """兩個開關選出的實作。"""

    access: AccessPolicy
    identity: IdentityResolver
    scan: OutboundScan


@dataclass(frozen=True, slots=True)
class ApiRuntime:
    retrieve: RetrieveEvidence
    conversations: ManageConversations
    submit: SubmitTurn
    turns: TurnRunner
    identity: IdentityResolver
    readiness: Mapping[str, Callable[[], Awaitable[bool]]]
    warm_up: Callable[[], None]
    close: Callable[[], Awaitable[None]]


def policies(settings: Settings) -> Policies:
    """兩個開關唯一的決策點。"""

    if settings.access_control_enabled:
        if settings.app_env == "production":
            raise ProductionIdentityMissingError(
                "ACCESS_CONTROL_ENABLED=true needs a production identity adapter; "
                "development headers are refused when APP_ENV=production"
            )
        access: AccessPolicy = AclAccess()
        identity: IdentityResolver = DevelopmentHeaderIdentity()
    else:
        access, identity = OpenAccess(), AnonymousIdentity()
    scan: OutboundScan = (
        RuleBasedOutboundScan.packaged()
        if settings.sensitive_data_scan_enabled
        else NoOutboundScan()
    )
    return Policies(access=access, identity=identity, scan=scan)


def llm_provider(settings: Settings) -> LlmProvider:
    if settings.llm_provider == "fake":
        return FakeLlmProvider()
    if settings.groq_api_key is None:
        raise MissingGroqKeyError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
    return GroqChatProvider(
        base_url=settings.groq_base_url,
        api_key=settings.groq_api_key.get_secret_value(),
        model=settings.groq_model,
        connect_timeout_seconds=settings.groq_connect_timeout_seconds,
        read_timeout_seconds=settings.groq_read_timeout_seconds,
        total_timeout_seconds=settings.groq_total_timeout_seconds,
        max_concurrency=settings.groq_max_concurrency,
        total_attempts=settings.groq_total_attempts,
    )


def build_runtime(settings: Settings, *, embedder: QueryEmbedder | None = None) -> ApiRuntime:
    """測試可以換掉 embedding model（主機上沒有 torch），其餘都是正式的 adapter。"""

    chosen = policies(settings)
    provider = llm_provider(settings)
    encoder = E5Encoder(
        settings.embedding_model_id,
        settings.embedding_model_revision,
        max_sequence_length=settings.embedding_max_sequence_length,
        batch_size=settings.embedding_batch_size,
    )
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    sessions = session_factory(engine)
    catalog = SqlChunkCatalog(sessions)
    conversations = SqlConversations(sessions)
    vectors = QdrantVectorSearch(
        AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=(
                settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            ),
            timeout=30,
        ),
        settings.qdrant_collection,
    )
    timezone = ZoneInfo(settings.schedule_timezone)

    def today() -> date:
        return datetime.now(timezone).date()

    understand = UnderstandQuestion(
        provider=provider,
        scan=chosen.scan,
        limits=RewriteLimits(
            history_turns=settings.rewrite_history_turns,
            turn_max_chars=settings.rewrite_turn_max_chars,
            max_chars=settings.rewrite_max_chars,
            timeout_seconds=settings.rewrite_timeout_seconds,
            max_prompt_bytes=settings.max_prompt_tokens,
            reasoning_effort=settings.rewrite_reasoning_effort,
        ),
    )
    retrieve = RetrieveEvidence(
        understand=understand,
        candidates=FindCandidates(
            embedder=embedder or encoder,
            vectors=vectors,
            keywords=PostgresKeywordSearch(sessions),
            catalog=catalog,
            keyword_min_score=settings.keyword_min_score,
        ),
        access=chosen.access,
        limits=AssemblyLimits(
            max_evidence=settings.retrieval_limit,
            max_chunks_per_document=settings.max_chunks_per_document,
            max_context_tokens=settings.max_context_tokens,
        ),
        candidate_count=settings.retrieval_limit * settings.candidate_multiplier,
        today=today,
    )
    turns = TurnRunner(
        ProcessTurn(
            repository=conversations,
            understand=understand,
            retrieve=retrieve,
            generate=GenerateAnswer(
                provider=provider,
                scan=chosen.scan,
                limits=AnswerLimits(
                    max_prompt_bytes=settings.max_prompt_tokens,
                    max_output_tokens=settings.groq_max_output_tokens,
                    temperature=settings.groq_temperature,
                    reasoning_effort=settings.answer_reasoning_effort,
                ),
            ),
            history_turns=settings.rewrite_history_turns,
        ),
        RecoverStaleTurns(
            conversations, stale_after=timedelta(seconds=settings.turn_stale_after_seconds)
        ),
        max_concurrency=settings.turn_max_concurrency,
    )

    async def model_loaded() -> bool:
        return embedder is not None or encoder.is_loaded

    def warm_up() -> None:
        if embedder is None:
            encoder.load()

    async def close() -> None:
        await turns.close()
        await provider.close()
        await vectors.close()
        await engine.dispose()

    return ApiRuntime(
        retrieve=retrieve,
        conversations=ManageConversations(conversations),
        submit=SubmitTurn(conversations),
        turns=turns,
        identity=chosen.identity,
        readiness={
            "postgres": catalog.is_ready,
            "qdrant": vectors.is_ready,
            "embedding_model": model_loaded,
        },
        warm_up=warm_up,
        close=close,
    )
