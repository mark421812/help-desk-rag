"""PDF 純文字的結構推斷：標題偵測、章節路徑堆疊、CJK 硬換行還原。

pypdf 只給純文字，沒有版面資訊，所以規則偏保守：寧可漏掉標題，也不要把內文誤判成標題。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import pairwise

_TOP_NUMERALS = "壹貳參肆伍陸柒捌玖拾"
# U+3127 (ㄧ) 在官方簡章 PDF 中會被用來代替「一」。
_CJK_DIGITS = {c: i for i, c in enumerate("〇一二三四五六七八九", 0)} | {"ㄧ": 1}

_TOP_RE = re.compile(rf"^([{_TOP_NUMERALS}]+)、\s*(.+)$")
_SECTION_RE = re.compile(r"^([一二三四五六七八九十ㄧ]+)、\s*(.+)$")
_DECIMAL_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$")
_LEADER_RE = re.compile(r"[‥…·]{2,}|\.{4,}")
_TRAILING_PAGE_REF_RE = re.compile(r"\s\d{1,3}$")
_ITEM_START_RE = re.compile(
    rf"^(?:[（(][一二三四五六七八九十\d]+[）)]|[①-⑳]|\d+[.、．]|[一二三四五六七八九十ㄧ{_TOP_NUMERALS}]+、"
    r"|[※◎●○■□▲•*\-]|【)"
)
_APPENDIX_RE = re.compile(r"^【+\s*((?:附錄|附表)[一二三四五六七八九十\d]+(?:-\d+)?)\s*】")
_ORPHAN_MARKER_RE = re.compile(r"^(?:\d{1,2}[.、]|[（(]\d{1,2}[）)]|[①-⑳])$")
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,3}\s*$")
_NO_JOIN_AFTER = frozenset("。！？；：.!?;:")
_HEADING_FORBIDDEN = frozenset("。，；,;！？!?")

# 只收錄整行等於這些詞才視為標題的招生簡章常見章節名；刻意保持很短。
_MARKER_HEADINGS = frozenset(
    {"報名資格", "報名時間", "報名方式", "報名費用", "繳費方式", "考試科目", "考試日期", "放榜日期"}
)


@dataclass(frozen=True, slots=True)
class Heading:
    level: int
    title: str
    numeral: int | None = None


def _numeral_value(text: str) -> int | None:
    """解析 一～三十九 的中文數字；不認得的組合回傳 None。"""

    if text == "十":
        return 10
    if text.startswith("十") and len(text) == 2 and text[1] in _CJK_DIGITS:
        return 10 + _CJK_DIGITS[text[1]]
    if len(text) >= 2 and text[1] == "十" and text[0] in "二三":
        rest = text[2:]
        if not rest:
            return _CJK_DIGITS[text[0]] * 10
        if len(rest) == 1 and rest in _CJK_DIGITS:
            return _CJK_DIGITS[text[0]] * 10 + _CJK_DIGITS[rest]
        return None
    if len(text) == 1 and text in _CJK_DIGITS:
        return _CJK_DIGITS[text]
    return None


def _clean_body(body: str) -> str | None:
    """回傳可當標題的內文；含句讀、過長或像目錄條目時回傳 None。"""

    cleaned = re.sub(r"\s+", "", body)
    cleaned = cleaned.rstrip(":：")
    if not 2 <= len(cleaned) <= 30:
        return None
    if _LEADER_RE.search(body) or _TRAILING_PAGE_REF_RE.search(body.strip()):
        return None
    if any(ch in _HEADING_FORBIDDEN for ch in cleaned) or ":" in cleaned or "：" in cleaned:
        return None
    return cleaned


def section_numeral(line: str) -> int | None:
    """行首是「一、」類章節編號時回傳其數值，不論後面內容是否像標題。"""

    match = _SECTION_RE.match(line.strip())
    return _numeral_value(match.group(1).replace("ㄧ", "一")) if match else None


def detect_heading(line: str) -> Heading | None:
    """判斷單行是否為標題；不確定就回傳 None。

    層級：壹、貳、(1) → 一、二、(2) → 1.2 小數編號 (3 起) 與固定章節詞 (3)。
    （一）與 1. 在簡章裡多半是條列內文，所以不視為標題。
    """

    text = unicodedata.normalize("NFC", line).strip()
    if not text:
        return None

    plain = re.sub(r"\s+", "", text).rstrip(":：")
    if plain in _MARKER_HEADINGS:
        return Heading(level=3, title=plain)

    if match := _TOP_RE.match(text):
        body = _clean_body(match.group(2))
        if body:
            return Heading(level=1, title=f"{match.group(1)}、{body}")
        return None
    if match := _SECTION_RE.match(text):
        value = _numeral_value(match.group(1).replace("ㄧ", "一"))
        body = _clean_body(match.group(2))
        if value is not None and body:
            marker = match.group(1).replace("ㄧ", "一")
            return Heading(level=2, title=f"{marker}、{body}", numeral=value)
        return None
    if match := _DECIMAL_RE.match(text):
        body = _clean_body(match.group(2))
        if body and not body[0].isdigit():
            return Heading(level=2 + match.group(1).count("."), title=f"{match.group(1)} {body}")
    return None


def appendix_heading(lines: list[str]) -> Heading | None:
    """簡章附錄頁以「【附錄一】」標記頁面歸屬，位置不固定（頁首或頁尾），所以整頁套用。"""

    for line in lines:
        if match := _APPENDIX_RE.match(line):
            return Heading(level=1, title=match.group(1))
    return None


def push_heading(stack: tuple[Heading, ...], heading: Heading) -> tuple[Heading, ...]:
    """新標題取代同層與更深層的舊標題，保留上層。"""

    return (*(h for h in stack if h.level < heading.level), heading)


def section_path(stack: tuple[Heading, ...]) -> str | None:
    return " > ".join(h.title for h in stack) or None


def is_toc_page(lines: list[str]) -> bool:
    """目錄頁（點線引導符）不含真正章節，整頁不偵測標題。"""

    return sum(1 for line in lines if _LEADER_RE.search(line)) >= 3


def carry_over(stack: tuple[Heading, ...], page_lines: list[str]) -> tuple[Heading, ...]:
    """換頁時決定上一頁的子標題是否延續。

    分則表格頁每頁都從「一、」重新編號；若本頁第一個章節編號不大於上頁最後的編號，
    代表上一頁的子標題已結束，延續下去會把新頁開頭標錯章節。
    """

    last = next((h.numeral for h in reversed(stack) if h.level == 2), None)
    if last is None:
        return stack
    first = next((n for n in (section_numeral(line) for line in page_lines) if n is not None), None)
    if first is not None and first <= last:
        return tuple(h for h in stack if h.level < 2)
    return stack


def drop_page_number_lines(lines: list[str]) -> list[str]:
    """移除頁首或頁尾單獨一個頁碼的行。"""

    result = list(lines)
    content_indexes = [i for i, line in enumerate(result) if line.strip()]
    if len(content_indexes) < 2:
        return result
    for index in (content_indexes[-1], content_indexes[0]):
        if _PAGE_NUMBER_RE.match(result[index]):
            result[index] = ""
    return result


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def join_wrapped_lines(lines: list[str], full_width: int) -> str:
    """把被版面硬換行截斷的句子接回去。

    只有「前一行幾乎滿版、沒有以句號或冒號結尾、下一行不是條列或標題開頭」才接，
    表格儲存格這類短行會原樣保留換行。
    """

    if not lines:
        return ""
    parts = [lines[0]]
    for previous, line in pairwise(lines):
        stripped_previous = previous.rstrip()
        if _ORPHAN_MARKER_RE.match(stripped_previous):
            # pypdf 常把條列編號與內文拆成兩行。
            parts.append(" " + line.lstrip())
            continue
        joinable = (
            full_width > 0
            and _display_width(stripped_previous) >= full_width * 0.9
            and stripped_previous[-1] not in _NO_JOIN_AFTER
            and not _ITEM_START_RE.match(line.lstrip())
            and detect_heading(line) is None
        )
        if not joinable:
            parts.append("\n" + line)
        elif stripped_previous[-1].isascii() and line.lstrip()[:1].isascii():
            parts.append(" " + line.lstrip())
        else:
            parts.append(line.lstrip())
    return "".join(parts)


def page_full_width(lines: list[str]) -> int:
    """以第 90 百分位行寬估計版面滿版寬度，避免單一超長行（網址）拉高門檻。"""

    widths = sorted(_display_width(line.rstrip()) for line in lines if line.strip())
    if len(widths) < 4:
        return 0
    return widths[int(0.9 * (len(widths) - 1))]
