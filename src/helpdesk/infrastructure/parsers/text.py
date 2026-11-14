from __future__ import annotations

from pathlib import Path

from helpdesk.application.ingestion.ports import ParsedBlock, ParsedDocument
from helpdesk.domain.ingestion import InvalidDocumentError
from helpdesk.infrastructure.parsers.base import (
    decode_text,
    normalize_text,
    split_paragraphs,
    validate_input_file,
)


class TextDocumentParser:
    """純文字依空白行切成段落。"""

    extensions = frozenset({".txt"})
    mime_types = frozenset({"text/plain"})
    parser_name = "plain-text"
    parser_version = "1"

    def parse(self, path: Path) -> ParsedDocument:
        validate_input_file(path)
        text = normalize_text(decode_text(path.read_bytes(), path))
        if not text:
            raise InvalidDocumentError(f"text document is empty: {path.name}")
        return ParsedDocument(
            blocks=tuple(
                ParsedBlock(content=paragraph, block_type="paragraph")
                for paragraph in split_paragraphs(text)
            ),
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            mime_type="text/plain",
        )
