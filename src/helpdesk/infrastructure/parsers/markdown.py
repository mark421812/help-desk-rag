from __future__ import annotations

import re
from pathlib import Path

from helpdesk.application.ingestion.ports import ParsedBlock, ParsedDocument
from helpdesk.domain.ingestion import InvalidDocumentError
from helpdesk.infrastructure.parsers.base import decode_text, normalize_text, validate_input_file

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_LIST_RE = re.compile(r"^[ \t]*(?:[-+*]|\d+[.)])[ \t]+")
_TABLE_SEPARATOR_RE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t]*$"
)


class MarkdownDocumentParser:
    """用小型狀態機保留標題階層、code fence、表格與清單。"""

    extensions = frozenset({".md", ".markdown"})
    mime_types = frozenset({"text/markdown", "text/x-markdown"})
    parser_name = "markdown"
    parser_version = "1"

    def parse(self, path: Path) -> ParsedDocument:
        validate_input_file(path)
        text = normalize_text(decode_text(path.read_bytes(), path))
        if not text:
            raise InvalidDocumentError(f"markdown document is empty: {path.name}")

        blocks: list[ParsedBlock] = []
        headings: list[str] = []
        buffer: list[str] = []
        buffer_type = "paragraph"
        in_fence = False
        fence_marker = ""

        def section_path() -> str | None:
            populated = [heading for heading in headings if heading]
            return " > ".join(populated) if populated else None

        def flush() -> None:
            nonlocal buffer, buffer_type
            content = normalize_text("\n".join(buffer))
            if content:
                blocks.append(
                    ParsedBlock(
                        content=content, section_path=section_path(), block_type=buffer_type
                    )
                )
            buffer = []
            buffer_type = "paragraph"

        lines = text.splitlines()
        index = 0
        while index < len(lines):
            line = lines[index]
            fence_match = _FENCE_RE.match(line)
            if in_fence:
                buffer.append(line)
                if fence_match and line.lstrip().startswith(fence_marker):
                    flush()
                    in_fence = False
                    fence_marker = ""
                index += 1
                continue

            if fence_match:
                flush()
                in_fence = True
                fence_marker = fence_match.group(1)[0] * len(fence_match.group(1))
                buffer_type = "code"
                buffer.append(line)
                index += 1
                continue

            heading_match = _HEADING_RE.match(line)
            if heading_match:
                flush()
                level = len(heading_match.group(1))
                del headings[level - 1 :]
                while len(headings) < level - 1:
                    headings.append("")
                headings.append(heading_match.group(2).strip())
                index += 1
                continue

            if not line.strip():
                flush()
                index += 1
                continue

            is_table_start = (
                "|" in line
                and index + 1 < len(lines)
                and _TABLE_SEPARATOR_RE.match(lines[index + 1]) is not None
            )
            if is_table_start:
                flush()
                buffer_type = "table"
                buffer.extend((line, lines[index + 1]))
                index += 2
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    buffer.append(lines[index])
                    index += 1
                flush()
                continue

            if _LIST_RE.match(line):
                if buffer and buffer_type != "list":
                    flush()
                buffer_type = "list"
                buffer.append(line)
                index += 1
                continue

            if buffer and buffer_type == "list":
                flush()
            buffer.append(line)
            index += 1

        flush()
        if not blocks:
            raise InvalidDocumentError(f"markdown document has no indexable content: {path.name}")
        return ParsedDocument(
            blocks=tuple(blocks),
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            mime_type="text/markdown",
        )
