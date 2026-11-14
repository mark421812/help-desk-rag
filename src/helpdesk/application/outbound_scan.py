"""送往外部 LLM 前的敏感資料掃描 port。

`SENSITIVE_DATA_SCAN_ENABLED` 在 composition root 決定注入規則掃描或不掃描的實作，呼叫端只看
`ScanVerdict.allowed`。判定只帶規則代號與類別，不帶命中的原文。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ScanVerdict:
    """沒有命中任何規則就是放行，所以不會出現「放行但有命中」的狀態。"""

    policy_version: str
    rule_ids: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return not self.rule_ids


class OutboundScan(Protocol):
    def scan(self, texts: Sequence[str]) -> ScanVerdict:
        """把所有即將外送的文字當成一份內容掃描，跨欄位的組合規則才不會被拆開繞過。"""
        ...
