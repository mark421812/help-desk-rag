from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from helpdesk.application.ingestion.ports import ParsedBlock, ParsedDocument
from helpdesk.domain.ingestion import InvalidDocumentError
from helpdesk.infrastructure.parsers.base import normalize_text, validate_input_file

_HEADING_STYLE_RE = re.compile(r"^heading[ \t]+([1-9])$", re.IGNORECASE)
_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class DocxDocumentParser:
    """依原文件順序讀取段落與表格，並維護 Heading 階層。"""

    extensions = frozenset({".docx"})
    mime_types = frozenset({_MIME_TYPE})
    parser_name = "python-docx"
    parser_version = "1"

    def parse(self, path: Path) -> ParsedDocument:
        validate_input_file(path)
        try:
            from docx import Document

            document = Document(str(path))
        except Exception as exc:
            raise InvalidDocumentError(f"unable to open DOCX document: {path.name}") from exc

        blocks: list[ParsedBlock] = []
        headings: list[str] = []
        # 第一個 Heading 1 或 Title 段落是文件標題，在那之前出現的 Title 段落不算內文。
        title_consumed = False

        for item_kind, item in _document_items(document):
            if item_kind == "paragraph":
                content = normalize_text(item.text)
                if not content:
                    continue
                style_name = str(getattr(getattr(item, "style", None), "name", "") or "")
                heading_match = _HEADING_STYLE_RE.match(style_name)
                if heading_match:
                    level = int(heading_match.group(1))
                    del headings[level - 1 :]
                    while len(headings) < level - 1:
                        headings.append("")
                    headings.append(content)
                    title_consumed = title_consumed or level == 1
                    continue
                if style_name.casefold() == "title" and not title_consumed:
                    title_consumed = True
                    continue
                blocks.append(
                    ParsedBlock(
                        content=content,
                        section_path=_section_path(headings),
                        block_type="paragraph",
                    )
                )
                continue

            rendered_table = _render_table(item)
            if rendered_table:
                blocks.append(
                    ParsedBlock(
                        content=rendered_table,
                        section_path=_section_path(headings),
                        block_type="table",
                    )
                )

        if not blocks:
            raise InvalidDocumentError(f"DOCX document has no indexable content: {path.name}")
        return ParsedDocument(
            blocks=tuple(blocks),
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            mime_type=_MIME_TYPE,
        )


def _section_path(headings: list[str]) -> str | None:
    populated = [heading for heading in headings if heading]
    return " > ".join(populated) if populated else None


def _document_items(document: Any) -> Iterator[tuple[str, Any]]:
    """保留段落與表格在 Word 裡交錯的順序。"""

    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield "paragraph", Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield "table", Table(child, document)


def _render_table(table: Any) -> str:
    """把表格轉成類 Markdown 文字，讓 embedding 保留欄列關係。"""

    rows: list[str] = []
    for row in table.rows:
        cells = [normalize_text(cell.text).replace("|", r"\|") for cell in row.cells]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    if len(rows) == 1:
        return rows[0]
    column_count = len(table.rows[0].cells)
    separator = "| " + " | ".join("---" for _ in range(column_count)) + " |"
    return "\n".join((rows[0], separator, *rows[1:]))
