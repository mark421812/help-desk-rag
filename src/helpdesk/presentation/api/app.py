"""helpdesk-api 的 FastAPI app：網頁、對話與提問、retrieve、健康檢查。

路由只做 HTTP 與 domain 之間的轉換：解析 request、從 header 或 cookie 取得身分、呼叫 use case、
把結果投影成回應。受限證據（不可外送）只回引用，不回內容。

提問走輪詢：`POST /api/v1/conversations/{id}/turns` 接受回合後回 202 與回合網址，處理在背景
進行；`GET` 回合網址取得目前狀態。回答與引用只在回合完成後才出現，不會有半成品。同一把
Idempotency-Key 再送一次會回 200 與既有回合。

log 只記數量、狀態與代碼，從不記問題、歷史、回答、證據的文字或身分。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Final, Literal, Self, cast
from uuid import UUID

import structlog
from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from helpdesk.application.conversation.manage import ConversationPage
from helpdesk.application.conversation.ports import (
    ConversationCursor,
    ConversationNotFoundError,
    ConversationRecord,
    TurnNotFoundError,
)
from helpdesk.application.identity import IdentityInput, UnauthorizedError
from helpdesk.application.retrieval.retrieve import Retrieval
from helpdesk.application.retrieval.understanding import HistoryMessage
from helpdesk.config import Settings
from helpdesk.domain.access import Principal
from helpdesk.domain.conversations import ConversationId, Turn, TurnId
from helpdesk.domain.errors import (
    ArchivedConversationError,
    DomainError,
    DomainValidationError,
    IdempotencyConflictError,
)
from helpdesk.domain.retrieval import Citation, Evidence
from helpdesk.presentation.api.composition import ApiRuntime, build_runtime
from helpdesk.presentation.api.middleware import (
    BoundedRequestBody,
    BrowserIdCookie,
    RequestContext,
    error_body,
)

logger = structlog.get_logger(__name__)

# 兩萬字的中文問題在 ensure_ascii 的 JSON 裡約 120 KB，再留空間給歷史。
MAX_REQUEST_BYTES: Final = 262_144
MAX_TEXT_LENGTH: Final = 20_000
MAX_HISTORY_MESSAGES: Final = 50
MAX_TITLE_LENGTH: Final = 200
STATIC_DIR: Final = Path(__file__).parent / "static"

RuntimeFactory = Callable[[Settings], ApiRuntime]

# 依例外型別對應 HTTP 狀態與對外的錯誤代碼；訊息只描述規則，不帶輸入值。
DOMAIN_ERRORS: Final[tuple[tuple[type[DomainError], int, str], ...]] = (
    (ConversationNotFoundError, 404, "not_found"),
    (TurnNotFoundError, 404, "not_found"),
    (IdempotencyConflictError, 409, "idempotency_conflict"),
    (ArchivedConversationError, 409, "conversation_archived"),
    (UnauthorizedError, 401, "unauthorized"),
    (DomainValidationError, 422, "invalid_request"),
)


def _trimmed_language(value: str) -> str:
    if not value.strip():
        raise ValueError("language must not be blank")
    return value.strip()


Language = Annotated[str, Field(min_length=1, max_length=20), AfterValidator(_trimmed_language)]


class HistoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    language: Language | None = None
    history: list[HistoryItem] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)


class CreateConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=MAX_TITLE_LENGTH)

    @field_validator("title")
    @classmethod
    def blank_title_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class UpdateConversationRequest(BaseModel):
    """改名、封存或兩者都做；封存後不能再提問，也不能取消封存。"""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=MAX_TITLE_LENGTH)
    status: Literal["archived"] | None = None

    @field_validator("title")
    @classmethod
    def title_is_trimmed(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("title must not be blank")
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def changes_something(self) -> Self:
        if self.title is None and self.status is None:
            raise ValueError("title or status is required")
        return self


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    language: Language | None = None


class ConversationOut(BaseModel):
    conversation_id: UUID
    title: str | None
    status: str
    created_at: datetime
    updated_at: datetime


class ConversationPageOut(BaseModel):
    conversations: list[ConversationOut]
    next_cursor: str | None


class NumberedCitationOut(BaseModel):
    number: int
    title: str
    section_path: str | None
    page: int | None


class AnswerPartOut(BaseModel):
    text: str
    citations: list[int]


class AnswerOut(BaseModel):
    parts: list[AnswerPartOut]
    citations: list[NumberedCitationOut]


class TurnErrorOut(BaseModel):
    code: str
    retry_after_seconds: int | None


class TurnOut(BaseModel):
    """`answer` 只在 answered 時有值，`restricted_citations` 只在 external_generation_blocked。"""

    turn_id: UUID
    conversation_id: UUID
    state: str
    question: str
    standalone_question: str | None
    language: str | None
    outcome: str | None
    answer: AnswerOut | None
    restricted_citations: list[NumberedCitationOut]
    error: TurnErrorOut | None
    status_url: str


class TurnListOut(BaseModel):
    turns: list[TurnOut]


class CitationOut(BaseModel):
    title: str
    section_path: str | None
    page: int | None


class ScoresOut(BaseModel):
    fused: float
    vector: float | None
    keyword: float | None


class EvidenceOut(BaseModel):
    rank: int
    bucket: Literal["sendable", "restricted"]
    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    document_type: str
    token_count: int
    content: str | None
    citation: CitationOut
    scores: ScoresOut


class RetrieveResponse(BaseModel):
    original_question: str
    standalone_question: str
    language: str | None
    rewrite: str
    status: str
    evidence: list[EvidenceOut]
    warnings: list[str]


def create_app(settings: Settings, *, runtime_factory: RuntimeFactory = build_runtime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = runtime_factory(settings)
        app.state.runtime = runtime
        runtime.turns.start()
        warm_up = asyncio.create_task(_warm_up(runtime))
        try:
            yield
        finally:
            await asyncio.wait({warm_up})
            await runtime.close()

    app = FastAPI(
        title="helpdesk-api", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
        return JSONResponse(
            status_code=422,
            content=error_body(
                "invalid_request", f"invalid fields: {', '.join(fields)}", _request_id(request)
            ),
        )

    @app.exception_handler(DomainError)
    async def domain_error(request: Request, exc: DomainError) -> JSONResponse:
        for kind, status, code in DOMAIN_ERRORS:
            if isinstance(exc, kind):
                return JSONResponse(
                    status_code=status, content=error_body(code, str(exc), _request_id(request))
                )
        raise exc

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready(request: Request) -> JSONResponse:
        runtime = _runtime(request)
        names = list(runtime.readiness)
        results = await asyncio.gather(*(runtime.readiness[name]() for name in names))
        checks = dict(zip(names, results, strict=True))
        is_ready = all(checks.values())
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={"status": "ready" if is_ready else "not_ready", "checks": checks},
        )

    @app.post("/api/v1/retrieve")
    async def retrieve(
        body: RetrieveRequest,
        request: Request,
        principal: Annotated[Principal, Depends(_principal)],
    ) -> RetrieveResponse:
        started = time.perf_counter()
        result = await _runtime(request).retrieve.run(
            body.question,
            language=body.language,
            history=[HistoryMessage(role=item.role, text=item.text) for item in body.history],
            principal=principal,
        )
        evidence = result.evidence
        logger.info(
            "retrieval_completed",
            status=evidence.status.value,
            rewrite=result.understanding.rewrite.value,
            rewrite_detail=result.understanding.detail,
            history_messages=len(body.history),
            candidates=len(result.candidates),
            sendable=len(evidence.sendable),
            restricted=len(evidence.restricted),
            keyword_failure=result.keyword_failure,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return _response(result)

    @app.post("/api/v1/conversations", status_code=201)
    async def create_conversation(
        body: CreateConversationRequest,
        request: Request,
        principal: Annotated[Principal, Depends(_principal)],
    ) -> ConversationOut:
        record = await _runtime(request).conversations.create(principal.user_id, body.title)
        return _conversation_out(record)

    @app.get("/api/v1/conversations")
    async def list_conversations(
        request: Request,
        principal: Annotated[Principal, Depends(_principal)],
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        cursor: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    ) -> ConversationPageOut:
        page = await _runtime(request).conversations.page(
            principal.user_id,
            after=None if cursor is None else _decode_cursor(cursor),
            limit=limit,
        )
        return _page_out(page)

    @app.get("/api/v1/conversations/{conversation_id}")
    async def get_conversation(
        conversation_id: UUID,
        request: Request,
        principal: Annotated[Principal, Depends(_principal)],
    ) -> ConversationOut:
        record = await _runtime(request).conversations.get(
            principal.user_id, ConversationId(conversation_id)
        )
        return _conversation_out(record)

    @app.patch("/api/v1/conversations/{conversation_id}")
    async def update_conversation(
        conversation_id: UUID,
        body: UpdateConversationRequest,
        request: Request,
        principal: Annotated[Principal, Depends(_principal)],
    ) -> ConversationOut:
        record = await _runtime(request).conversations.update(
            principal.user_id,
            ConversationId(conversation_id),
            title=body.title,
            archive=body.status == "archived",
        )
        return _conversation_out(record)

    @app.get("/api/v1/conversations/{conversation_id}/turns")
    async def list_turns(
        conversation_id: UUID,
        request: Request,
        principal: Annotated[Principal, Depends(_principal)],
    ) -> TurnListOut:
        turns = await _runtime(request).conversations.turns(
            principal.user_id, ConversationId(conversation_id)
        )
        return TurnListOut(turns=[_turn_out(turn) for turn in turns])

    @app.post("/api/v1/conversations/{conversation_id}/turns")
    async def ask(
        conversation_id: UUID,
        body: AskRequest,
        request: Request,
        principal: Annotated[Principal, Depends(_principal)],
        idempotency_key: Annotated[UUID, Header()],
    ) -> JSONResponse:
        runtime = _runtime(request)
        submission = await runtime.submit.run(
            principal.user_id,
            ConversationId(conversation_id),
            idempotency_key=idempotency_key,
            question=body.question,
            language=body.language,
        )
        if submission.created:
            runtime.turns.submit(submission.turn, principal)
        logger.info(
            "turn_submitted",
            turn_id=str(submission.turn.id),
            replayed=not submission.created,
            state=submission.turn.state.value,
        )
        out = _turn_out(submission.turn)
        return JSONResponse(
            status_code=202 if submission.created else 200,
            content=out.model_dump(mode="json"),
            headers={"location": out.status_url},
        )

    @app.get("/api/v1/conversations/{conversation_id}/turns/{turn_id}")
    async def get_turn(
        conversation_id: UUID,
        turn_id: UUID,
        request: Request,
        principal: Annotated[Principal, Depends(_principal)],
    ) -> TurnOut:
        turn = await _runtime(request).conversations.turn(
            principal.user_id, ConversationId(conversation_id), TurnId(turn_id)
        )
        return _turn_out(turn)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.add_middleware(BrowserIdCookie, secure=settings.app_env == "production")
    app.add_middleware(BoundedRequestBody, max_bytes=MAX_REQUEST_BYTES)
    app.add_middleware(RequestContext)
    return app


async def _warm_up(runtime: ApiRuntime) -> None:
    started = time.perf_counter()
    try:
        await asyncio.to_thread(runtime.warm_up)
    except Exception as exc:
        logger.error("embedding_model_load_failed", error_type=type(exc).__name__)
        return
    logger.info("embedding_model_loaded", seconds=round(time.perf_counter() - started, 1))


def _runtime(request: Request) -> ApiRuntime:
    return cast(ApiRuntime, request.app.state.runtime)


def _request_id(request: Request) -> str | None:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else None


def _principal(request: Request) -> Principal:
    authorization = request.headers.get("authorization")
    bearer_token: str | None = None
    if authorization is not None:
        scheme, _, token = authorization.partition(" ")
        if scheme.casefold() != "bearer" or not token.strip():
            raise UnauthorizedError("authorization must use a bearer token")
        bearer_token = token.strip()
    groups = request.headers.get("x-dev-group-ids")
    return _runtime(request).identity.resolve(
        IdentityInput(
            bearer_token=bearer_token,
            development_user_id=request.headers.get("x-dev-user-id"),
            development_group_ids=(
                tuple(item.strip() for item in groups.split(","))
                if groups and groups.strip()
                else ()
            ),
            browser_id=getattr(request.state, "browser_id", None),
        )
    )


def _conversation_out(record: ConversationRecord) -> ConversationOut:
    conversation = record.conversation
    return ConversationOut(
        conversation_id=conversation.id,
        title=conversation.title,
        status=conversation.status.value,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _page_out(page: ConversationPage) -> ConversationPageOut:
    cursor = page.next_cursor
    return ConversationPageOut(
        conversations=[_conversation_out(record) for record in page.records],
        next_cursor=None if cursor is None else _encode_cursor(cursor),
    )


def _encode_cursor(cursor: ConversationCursor) -> str:
    raw = f"{cursor.updated_at.isoformat()}|{cursor.conversation_id}"
    return base64.urlsafe_b64encode(raw.encode("ascii")).decode("ascii").rstrip("=")


def _decode_cursor(value: str) -> ConversationCursor:
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode("ascii")
        stamp, separator, identifier = raw.partition("|")
        updated_at = datetime.fromisoformat(stamp)
        if not separator or updated_at.tzinfo is None:
            raise ValueError("incomplete cursor")
        return ConversationCursor(updated_at, ConversationId(UUID(identifier)))
    except (ValueError, binascii.Error, UnicodeDecodeError):
        raise DomainValidationError("cursor is not a valid page cursor") from None


def _numbered(citations: tuple[Citation, ...]) -> list[NumberedCitationOut]:
    return [
        NumberedCitationOut(
            number=number,
            title=citation.title,
            section_path=citation.section_path,
            page=citation.page_number,
        )
        for number, citation in enumerate(citations, start=1)
    ]


def _turn_out(turn: Turn) -> TurnOut:
    result = turn.result
    answer = result.answer if result is not None else None
    answer_out: AnswerOut | None = None
    if answer is not None:
        numbers = {citation.chunk_id: n for n, citation in enumerate(answer.citations, start=1)}
        answer_out = AnswerOut(
            parts=[
                AnswerPartOut(
                    text=part.text, citations=[numbers[chunk_id] for chunk_id in part.chunk_ids]
                )
                for part in answer.parts
            ],
            citations=_numbered(answer.citations),
        )
    return TurnOut(
        turn_id=turn.id,
        conversation_id=turn.conversation_id,
        state=turn.state.value,
        question=turn.original_question,
        standalone_question=turn.standalone_question,
        language=turn.language,
        outcome=result.outcome.value if result is not None else None,
        answer=answer_out,
        restricted_citations=_numbered(result.restricted_citations) if result else [],
        error=(
            None
            if turn.error_code is None
            else TurnErrorOut(code=turn.error_code, retry_after_seconds=turn.retry_after_seconds)
        ),
        status_url=f"/api/v1/conversations/{turn.conversation_id}/turns/{turn.id}",
    )


def _response(result: Retrieval) -> RetrieveResponse:
    plan = result.understanding.plan
    candidates = {candidate.chunk_id: candidate for candidate in result.candidates}
    sendable = {evidence.citation.chunk_id for evidence in result.evidence.sendable}

    def item(evidence: Evidence) -> EvidenceOut:
        citation = evidence.citation
        candidate = candidates[citation.chunk_id]
        external = citation.chunk_id in sendable
        return EvidenceOut(
            rank=evidence.rank,
            bucket="sendable" if external else "restricted",
            chunk_id=citation.chunk_id,
            document_id=citation.document_id,
            version_id=citation.version_id,
            document_type=evidence.document_type,
            token_count=evidence.token_count,
            content=evidence.content if external else None,
            citation=CitationOut(
                title=citation.title, section_path=citation.section_path, page=citation.page_number
            ),
            scores=ScoresOut(
                fused=candidate.fused_score,
                vector=candidate.vector_score,
                keyword=candidate.keyword_score,
            ),
        )

    return RetrieveResponse(
        original_question=plan.original_question,
        standalone_question=plan.standalone_question,
        language=plan.language,
        rewrite=result.understanding.rewrite.value,
        status=result.evidence.status.value,
        evidence=[item(evidence) for evidence in result.evidence.ordered],
        warnings=["keyword_search_unavailable"] if result.keyword_failure else [],
    )
