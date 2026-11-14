from __future__ import annotations

from pathlib import Path

from helpdesk.application.ingestion.ports import ParsedBlock, ParsedDocument
from helpdesk.domain.ingestion import InvalidDocumentError, OcrRequiredError
from helpdesk.infrastructure.parsers.base import normalize_text, validate_input_file
from helpdesk.infrastructure.parsers.pdf_structure import (
    Heading,
    appendix_heading,
    carry_over,
    detect_heading,
    drop_page_number_lines,
    is_toc_page,
    join_wrapped_lines,
    page_full_width,
    push_heading,
    section_path,
)


class PdfDocumentParser:
    """用 pypdf 逐頁擷取文字，頁碼留給引用使用。

    pypdf 沒有版面資訊，章節路徑由行首編號等保守規則推斷（見 `pdf_structure`），標題會
    跨頁延續。加密的 PDF 一律拒絕，即使使用者密碼是空的：作者設了密碼代表限制了內容
    的使用方式。版本 2 是加上章節偵測後的版本。
    """

    extensions = frozenset({".pdf"})
    mime_types = frozenset({"application/pdf"})
    parser_name = "pypdf"
    parser_version = "2"

    def parse(self, path: Path) -> ParsedDocument:
        validate_input_file(path)
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)
        except Exception as exc:
            raise InvalidDocumentError(f"unable to open PDF document: {path.name}") from exc
        if reader.is_encrypted:
            raise InvalidDocumentError(f"encrypted PDF is not supported: {path.name}")

        page_lines: list[list[str]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                extracted = normalize_text(page.extract_text() or "")
            except Exception as exc:
                raise InvalidDocumentError(
                    f"unable to extract text from PDF page {page_number}: {path.name}"
                ) from exc
            page_lines.append(
                drop_page_number_lines([line.strip() for line in extracted.split("\n")])
            )

        blocks: list[ParsedBlock] = []
        headings: tuple[Heading, ...] = ()
        for page_number, raw_lines in enumerate(page_lines, start=1):
            page_blocks, headings = _page_blocks(raw_lines, page_number, headings)
            blocks.extend(page_blocks)

        if not blocks:
            raise OcrRequiredError(f"PDF has no extractable text and requires OCR: {path.name}")
        return ParsedDocument(
            blocks=tuple(blocks),
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            mime_type="application/pdf",
        )


def _page_blocks(
    lines: list[str], page_number: int, headings: tuple[Heading, ...]
) -> tuple[list[ParsedBlock], tuple[Heading, ...]]:
    """把一頁切成 block；標題更新章節路徑，並從內文移除。"""

    detect = not is_toc_page(lines)
    if detect:
        headings = carry_over(headings, lines)
        appendix = appendix_heading(lines)
        if appendix is not None:
            headings = (appendix,)
    full_width = page_full_width(lines)

    blocks: list[ParsedBlock] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(
                ParsedBlock(
                    content=join_wrapped_lines(paragraph, full_width),
                    section_path=section_path(headings),
                    page_number=page_number,
                    block_type="paragraph",
                )
            )
            paragraph.clear()

    for line in lines:
        heading = detect_heading(line) if detect and line else None
        if heading is None and line:
            paragraph.append(line)
            continue
        flush()
        if heading is not None:
            headings = push_heading(headings, heading)
    flush()
    return blocks, headings
