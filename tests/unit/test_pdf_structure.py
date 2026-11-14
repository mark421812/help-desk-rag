"""PDF 章節偵測（parser 版本 2）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from helpdesk.infrastructure.parsers.pdf import PdfDocumentParser
from helpdesk.infrastructure.parsers.pdf_structure import (
    Heading,
    carry_over,
    detect_heading,
    drop_page_number_lines,
    is_toc_page,
    join_wrapped_lines,
    push_heading,
    section_path,
)


@pytest.mark.parametrize(
    ("line", "level", "title"),
    [
        ("壹、總則", 1, "壹、總則"),
        (
            "十四、成績通知單與錄取通知單寄發、補發及成績複查",
            2,
            "十四、成績通知單與錄取通知單寄發、補發及成績複查",
        ),
        ("四、 注意事項：", 2, "四、注意事項"),
        ("ㄧ、修業年限", 2, "一、修業年限"),
        ("二十、報到須知", 2, "二十、報到須知"),
        ("1.2 報名資格說明", 3, "1.2 報名資格說明"),
        ("報名資格", 3, "報名資格"),
        ("報名時間：", 3, "報名時間"),
    ],
)
def test_detects_headings(line: str, level: int, title: str) -> None:
    heading = detect_heading(line)

    assert heading is not None
    assert (heading.level, heading.title) == (level, title)


@pytest.mark.parametrize(
    "line",
    [
        "一、修業年限：一至四年。",
        "二、現任公立學校教師，且依法取得合格教師證書。",
        "三、現任各機關銓敘合格之公務人員以及民選公職人員。",
        "（一）網路報名取得繳費帳號：",
        "1. 一律至本校「招生網路報名系統」選擇報名",
        "一、報名資格審查及繳交資料之相關規定與注意事項說明如下列各項所述內容",
        "二、",
        "六、報考相關注意事項‥‥‥‥‥‥‥‥‥13",
        "七、報名結果查詢 14",
        "考生應於報名時間內完成報名資格確認。",
        "報名資格及報名時間均請詳閱",
        "1.5 倍",
        "",
    ],
)
def test_body_text_is_not_a_heading(line: str) -> None:
    assert detect_heading(line) is None


def test_heading_stack_replaces_same_and_deeper_levels() -> None:
    stack: tuple[Any, ...] = ()
    for line in ("壹、總則", "三、報考資格", "報名資格", "四、費用"):
        heading = detect_heading(line)
        assert heading is not None
        stack = push_heading(stack, heading)

    assert section_path(stack) == "壹、總則 > 四、費用"
    assert section_path(()) is None


def test_numbering_restart_ends_the_previous_pages_subsection() -> None:
    stack: tuple[Heading, ...] = ()
    for line in ("貳、招生專班分則", "五、收費標準"):
        heading = detect_heading(line)
        assert heading is not None
        stack = push_heading(stack, heading)

    restarted = carry_over(stack, ["專班別 乙班", "一、現任教師，且具有證書。"])
    continued = carry_over(stack, ["承上頁內容", "六、注意事項"])

    assert section_path(restarted) == "貳、招生專班分則"
    assert section_path(continued) == "貳、招生專班分則 > 五、收費標準"


def test_toc_page_and_page_numbers_are_recognised() -> None:
    toc = ["壹、總則", "一、修業年限 ‥‥‥‥‥ 10", "二、報名方式 ‥‥‥‥‥ 10", "三、報考資格 ‥‥‥‥‥ 10"]

    assert is_toc_page(toc)
    assert not is_toc_page(["壹、總則", "一、修業年限"])
    assert drop_page_number_lines(["12", "內文", "13"]) == ["", "內文", ""]
    assert drop_page_number_lines(["12"]) == ["12"]


def test_wrapped_cjk_lines_are_joined_but_short_rows_and_items_are_not() -> None:
    full = "請於審查資料繳交截止期限前繳交報名所需相關表"
    lines = [full, "件，未依規定者不受理。", "報名費", "1,300 元", full, "（二）第二項"]

    joined = join_wrapped_lines(lines, full_width=44)

    assert joined.split("\n") == [
        full + "件，未依規定者不受理。",
        "報名費",
        "1,300 元",
        full,
        "（二）第二項",
    ]


def test_sentence_ending_and_orphan_item_markers() -> None:
    full = "請於審查資料繳交截止期限前繳交報名所需相關表件。"

    assert join_wrapped_lines([full, "下一句"], full_width=48) == f"{full}\n下一句"
    assert join_wrapped_lines(["1.", "各節考試"], full_width=48) == "1. 各節考試"


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


def parse_pages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pages: list[str]) -> Any:
    class FakeReader:
        is_encrypted = False

        def __init__(self, _path: Path) -> None:
            self.pages = [FakePage(text) for text in pages]

    monkeypatch.setattr("pypdf.PdfReader", FakeReader)
    path = tmp_path / "brochure.pdf"
    path.write_bytes(b"%PDF-1.4")
    return PdfDocumentParser().parse(path)


def test_parser_assigns_section_paths_and_carries_them_across_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = parse_pages(
        monkeypatch,
        tmp_path,
        [
            "封面文字\n\n壹、總則\n三、報考資格\n具學士學位者。\n1",
            "2\n（二）同等學力者亦可報考。\n四、費用\n報名費新臺幣1,300元。",
        ],
    )

    observed = [(b.page_number, b.section_path, b.content) for b in document.blocks]

    assert observed == [
        (1, None, "封面文字"),
        (1, "壹、總則 > 三、報考資格", "具學士學位者。"),
        (2, "壹、總則 > 三、報考資格", "（二）同等學力者亦可報考。"),
        (2, "壹、總則 > 四、費用", "報名費新臺幣1,300元。"),
    ]
    assert document.parser_version == "2"


def test_parser_does_not_turn_body_sentences_into_headings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = parse_pages(
        monkeypatch,
        tmp_path,
        ["一、修業年限：一至四年。\n二、考生須於期限內完成報名，逾期不受理。\n（一）符合資格者。"],
    )

    assert all(block.section_path is None for block in document.blocks)
    assert "一、修業年限：一至四年。" in document.blocks[0].content


def test_parser_labels_appendix_pages_and_skips_toc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    toc = "目錄\n壹、總則\n一、修業年限 ‥‥‥‥ 10\n二、報名方式 ‥‥‥‥ 10\n三、報考資格 ‥‥‥‥ 11"
    document = parse_pages(
        monkeypatch,
        tmp_path,
        [toc, "貳、分則\n內文。", "辦法條文第一條\n【附錄一】"],
    )

    paths = [(b.page_number, b.section_path) for b in document.blocks]

    assert paths == [(1, None), (2, "貳、分則"), (3, "附錄一")]
    assert "壹、總則" in document.blocks[0].content
