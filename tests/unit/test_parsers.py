"""各格式的 parser，以及加密 PDF 一律拒絕。"""

from __future__ import annotations

from pathlib import Path

import pytest

from helpdesk.domain.ingestion import (
    InvalidDocumentError,
    OcrRequiredError,
    UnsupportedDocumentError,
)
from helpdesk.infrastructure.parsers.docx import DocxDocumentParser
from helpdesk.infrastructure.parsers.markdown import MarkdownDocumentParser
from helpdesk.infrastructure.parsers.registry import ParserRegistry, default_parsers
from helpdesk.infrastructure.parsers.text import TextDocumentParser


def test_text_parser_reads_utf8_and_preserves_paragraphs(tmp_path: Path) -> None:
    path = tmp_path / "remote-work-policy.txt"
    path.write_text("第一條 適用範圍\n\n員工每週可以遠端工作兩天。", encoding="utf-8")

    document = TextDocumentParser().parse(path)

    assert document.mime_type == "text/plain"
    assert [block.content for block in document.blocks] == [
        "第一條 適用範圍",
        "員工每週可以遠端工作兩天。",
    ]


def test_text_parser_falls_back_to_detected_legacy_encodings(tmp_path: Path) -> None:
    path = tmp_path / "big5.txt"
    path.write_bytes("請假規則：員工每年有七天特休，應事先提出申請。".encode("big5"))

    document = TextDocumentParser().parse(path)

    assert document.blocks[0].content == "請假規則：員工每年有七天特休，應事先提出申請。"


def test_markdown_parser_preserves_heading_hierarchy_and_block_types(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text(
        """# 人事規章

適用所有正式員工。

## 遠端工作

- 每週至多兩天
- 需主管核准

| 類型 | 天數 |
| --- | --- |
| 一般 | 2 |

```text
申請代碼: WFH
```
""",
        encoding="utf-8",
    )

    document = MarkdownDocumentParser().parse(path)

    assert [(block.section_path, block.block_type) for block in document.blocks] == [
        ("人事規章", "paragraph"),
        ("人事規章 > 遠端工作", "list"),
        ("人事規章 > 遠端工作", "table"),
        ("人事規章 > 遠端工作", "code"),
    ]
    assert "每週至多兩天" in document.blocks[1].content
    assert "| 一般 | 2 |" in document.blocks[2].content


def test_registry_selects_by_case_insensitive_extension(tmp_path: Path) -> None:
    path = tmp_path / "POLICY.MD"
    path.write_text("# 規章\n\n內容", encoding="utf-8")
    registry = default_parsers()

    assert registry.parse(path).parser_name == "markdown"
    assert registry.supported_extensions >= {".pdf", ".docx", ".txt", ".md"}


def test_registry_rejects_an_unknown_document_type(tmp_path: Path) -> None:
    path = tmp_path / "sheet.xlsx"
    path.write_bytes(b"x")

    with pytest.raises(UnsupportedDocumentError):
        default_parsers().parse(path)


def test_registry_rejects_duplicate_extensions() -> None:
    with pytest.raises(ValueError, match="more than one parser"):
        ParserRegistry((TextDocumentParser(), TextDocumentParser()))


def test_blank_pdf_is_marked_as_requiring_ocr(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    path = tmp_path / "scanned.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as output:
        writer.write(output)

    with pytest.raises(OcrRequiredError):
        default_parsers().parse(path)


def test_encrypted_pdf_is_rejected_even_with_an_empty_user_password(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    path = tmp_path / "protected.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt(user_password="", owner_password="owner-only", algorithm="RC4-128")
    with path.open("wb") as output:
        writer.write(output)

    with pytest.raises(InvalidDocumentError, match="encrypted PDF"):
        default_parsers().parse(path)


def test_docx_parser_preserves_heading_and_table_structure(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    path = tmp_path / "handbook.docx"
    source = docx.Document()
    source.add_paragraph("員工手冊", style="Title")
    source.add_heading("請假規則", level=2)
    source.add_paragraph("特休應事先提出申請。")
    table = source.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "假別"
    table.cell(0, 1).text = "天數"
    table.cell(1, 0).text = "事假"
    table.cell(1, 1).text = "14"
    source.save(path)

    document = DocxDocumentParser().parse(path)

    assert document.blocks[0].content == "特休應事先提出申請。"
    assert document.blocks[0].section_path == "請假規則"
    assert document.blocks[1].block_type == "table"
    assert "| 假別 | 天數 |" in document.blocks[1].content
