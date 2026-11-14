"""兩個程序共用的 log 設定：structlog 輸出一行一筆 JSON 到 stderr。"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str) -> None:
    numeric_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    logging.basicConfig(stream=sys.stderr, level=numeric_level, format="%(message)s", force=True)
    # qdrant-client 透過 httpx 每個請求記一行 INFO，會淹沒 worker 自己的事件。
    logging.getLogger("httpx").setLevel(max(numeric_level, logging.WARNING))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
