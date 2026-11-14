"""parser 共用的介面與文字正規化。"""

from __future__ import annotations

import codecs
import re
import unicodedata
from pathlib import Path
from typing import Protocol

from helpdesk.application.ingestion.ports import ParsedDocument
from helpdesk.domain.ingestion import InvalidDocumentError


class FormatParser(Protocol):
    extensions: frozenset[str]
    mime_types: frozenset[str]

    def parse(self, path: Path) -> ParsedDocument: ...


def validate_input_file(path: Path) -> None:
    if not path.exists():
        raise InvalidDocumentError(f"document does not exist: {path.name}")
    if not path.is_file():
        raise InvalidDocumentError(f"document is not a regular file: {path.name}")


def normalize_text(value: str) -> str:
    """正規化 Unicode 與換行，但不破壞有意義的段落與表格結構。"""

    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\x00", "")
    lines = [re.sub(r"[\t\f\v ]+$", "", line) for line in value.split("\n")]
    return "\n".join(lines).strip()


def split_paragraphs(value: str) -> tuple[str, ...]:
    normalized = normalize_text(value)
    if not normalized:
        return ()
    return tuple(
        paragraph.strip() for paragraph in re.split(r"\n[\t ]*\n+", normalized) if paragraph.strip()
    )


def decode_text(raw: bytes, path: Path) -> str:
    """先看 BOM，再試 UTF-8，最後才用 charset-normalizer 推測編碼。"""

    if raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return raw.decode("utf-32")
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as decode_error:
        from charset_normalizer import from_bytes

        match = from_bytes(raw).best()
        if match is None:
            raise InvalidDocumentError(
                f"unable to determine text encoding: {path.name}"
            ) from decode_error
        return str(match)
