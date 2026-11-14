"""純 ASGI middleware。

`BoundedRequestBody` 在 FastAPI 解析 JSON 之前逐段累計 body bytes，超過上限回 413，不只相信
`Content-Length`。`RequestContext` 建立 request ID、加上安全 header（CSP 只允許同源），每個
請求記一筆只有 method、路由樣板、狀態與耗時的 log。沒被處理的例外只記型別與程式位置，不記例外
訊息，因為資料庫或 HTTP 例外的訊息可能帶著查詢參數，也就是使用者的問題。`BrowserIdCookie` 在
第一次造訪時發給瀏覽器一個隨機 UUID 的 cookie，ACL 關閉時用它區分對話的擁有者。
"""

from __future__ import annotations

import time
import traceback
from collections.abc import MutableMapping
from typing import Final, cast
from uuid import UUID, uuid4

import structlog
from starlette.requests import cookie_parser
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = structlog.get_logger(__name__)

CONTENT_SECURITY_POLICY: Final = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
    "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)
SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cache-control", b"no-store"),
    (b"content-security-policy", CONTENT_SECURITY_POLICY.encode("ascii")),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-resource-policy", b"same-origin"),
)
BROWSER_ID_COOKIE: Final = "helpdesk_browser_id"
BROWSER_ID_MAX_AGE_SECONDS: Final = 365 * 24 * 3600


def error_body(error: str, message: str, request_id: str | None) -> dict[str, object]:
    return {"error": error, "message": message, "request_id": request_id}


class BoundedRequestBody:
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        declared = _header(scope, b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self._max_bytes:
            await self._reject(scope, receive, send)
            return
        body = bytearray()
        disconnect: Message | None = None
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnect = message
                break
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self._max_bytes:
                await self._reject(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if disconnect is not None:
                return disconnect
            if replayed:
                # 檔案回應會一直等 http.disconnect 來判斷用戶端是否離開；body 讀完後改等真正的
                # receive，否則它會以為用戶端已離線而送出空的 body。
                return await receive()
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self._app(scope, replay, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        state = scope.get("state")
        request_id = state.get("request_id") if isinstance(state, MutableMapping) else None
        response = JSONResponse(
            status_code=413,
            content=error_body(
                "request_too_large", "the request body exceeds the size limit", request_id
            ),
        )
        await response(scope, receive, send)


class RequestContext:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        request_id = _request_id(scope)
        state = scope.setdefault("state", {})
        if isinstance(state, MutableMapping):
            state["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        status = 500
        response_started = False

        async def send_with_headers(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = int(message["status"])
                names = {name for name, _ in SECURITY_HEADERS} | {b"x-request-id"}
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() not in names
                ]
                headers.extend(SECURITY_HEADERS)
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self._app(scope, receive, send_with_headers)
        except Exception as exc:
            logger.error(
                "unhandled_exception",
                error_type=type(exc).__name__,
                frames=[
                    f"{frame.filename}:{frame.lineno}:{frame.name}"
                    for frame in traceback.extract_tb(exc.__traceback__)
                ],
            )
            if response_started:
                raise
            response = JSONResponse(
                status_code=500,
                content=error_body("internal_error", "unexpected server error", request_id),
            )
            await response(scope, receive, send_with_headers)
        finally:
            route = scope.get("route")
            logger.info(
                "http_request",
                method=str(scope.get("method", "")),
                route=str(getattr(route, "path", "__unmatched__")),
                status=status,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            structlog.contextvars.clear_contextvars()


class BrowserIdCookie:
    """把瀏覽器的匿名 ID 放進 `scope["state"]["browser_id"]`，沒有或格式不對時發一個新的。

    cookie 只有隨機 UUID，不含任何個人資料。HttpOnly 讓頁面的 JavaScript 讀不到，SameSite=Lax
    讓別的網站發出的 POST 不會帶上它。
    """

    def __init__(self, app: ASGIApp, *, secure: bool) -> None:
        self._app = app
        self._secure = secure

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        cookies = cookie_parser(_header(scope, b"cookie") or "")
        existing = _canonical_uuid(cookies.get(BROWSER_ID_COOKIE))
        browser_id = existing or str(uuid4())
        state = scope.setdefault("state", {})
        if isinstance(state, MutableMapping):
            state["browser_id"] = browser_id
        if existing is not None:
            await self._app(scope, receive, send)
            return
        attributes = f"Path=/; Max-Age={BROWSER_ID_MAX_AGE_SECONDS}; HttpOnly; SameSite=Lax"
        cookie = f"{BROWSER_ID_COOKIE}={browser_id}; {attributes}"
        if self._secure:
            cookie += "; Secure"

        async def send_with_cookie(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [*message.get("headers", []), (b"set-cookie", cookie.encode("ascii"))]
                message = {**message, "headers": headers}
            await send(message)

        await self._app(scope, receive, send_with_cookie)


def _canonical_uuid(value: str | None) -> str | None:
    if value is None or len(value) > 64:
        return None
    try:
        return str(UUID(value))
    except ValueError:
        return None


def _request_id(scope: Scope) -> str:
    """只接受 128 字以內的可列印 ASCII，否則產生新的 UUID。"""

    raw = _header(scope, b"x-request-id")
    value = (raw or "").strip()
    if 0 < len(value) <= 128 and all(0x20 <= ord(character) <= 0x7E for character in value):
        return value
    return str(uuid4())


def _header(scope: Scope, name: bytes) -> str | None:
    for raw_name, raw_value in cast(list[tuple[bytes, bytes]], scope.get("headers", [])):
        if raw_name.lower() == name:
            return raw_value.decode("latin-1")
    return None
