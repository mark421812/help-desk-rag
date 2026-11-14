from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from helpdesk.application.ingestion.ports import ParsedDocument
from helpdesk.domain.ingestion import UnsupportedDocumentError
from helpdesk.infrastructure.parsers.base import FormatParser
from helpdesk.infrastructure.parsers.docx import DocxDocumentParser
from helpdesk.infrastructure.parsers.markdown import MarkdownDocumentParser
from helpdesk.infrastructure.parsers.pdf import PdfDocumentParser
from helpdesk.infrastructure.parsers.text import TextDocumentParser


class ParserRegistry:
    """依副檔名選 parser；同一個副檔名註冊兩次直接拒絕，選擇結果才會固定。"""

    def __init__(self, parsers: Iterable[FormatParser]) -> None:
        by_extension: dict[str, FormatParser] = {}
        for parser in parsers:
            for raw_extension in parser.extensions:
                extension = _normalize_extension(raw_extension)
                if extension in by_extension:
                    raise ValueError(f"more than one parser is registered for {extension}")
                by_extension[extension] = parser
        self._by_extension = by_extension

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(self._by_extension)

    def parse(self, path: Path) -> ParsedDocument:
        parser = self._by_extension.get(_normalize_extension(path.suffix))
        if parser is None:
            raise UnsupportedDocumentError(
                f"unsupported document extension {path.suffix or '<none>'!r}: {path.name}"
            )
        return parser.parse(path)


def default_parsers() -> ParserRegistry:
    return ParserRegistry(
        (
            TextDocumentParser(),
            MarkdownDocumentParser(),
            DocxDocumentParser(),
            PdfDocumentParser(),
        )
    )


def _normalize_extension(extension: str) -> str:
    normalized = extension.strip().casefold()
    if normalized and not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized
