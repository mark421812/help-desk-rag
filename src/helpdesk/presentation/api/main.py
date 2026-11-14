"""`helpdesk-api` 指令列進入點。

預設只綁 127.0.0.1；容器裡由 compose 傳 `--host 0.0.0.0`，主機那一側再綁回 127.0.0.1。
uvicorn 的 access log 關閉，改由 middleware 記不含內容的請求 log。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from helpdesk.config import Settings
from helpdesk.infrastructure.logging import configure_logging
from helpdesk.presentation.api.app import create_app


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="helpdesk-api", description="Help Desk API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8140)
    args = parser.parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=args.port,
        access_log=False,
        server_header=False,
        proxy_headers=False,
        log_config=None,
    )
