"""敏感資料掃描：沿用 policy.yaml 的規則，壞掉的規則檔讓程序無法啟動。"""

from __future__ import annotations

import pytest

from helpdesk.infrastructure.outbound_scan.scanners import (
    InvalidPolicyError,
    NoOutboundScan,
    RuleBasedOutboundScan,
)

SCAN = RuleBasedOutboundScan.packaged()


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("api_key = sk-abcdef123456", "secret-assignment"),
        ("身分證 A123456789", "taiwan-id"),
        ("卡號 4111 1111 1111 1111", "payment-card"),
        ("-----BEGIN RSA PRIVATE KEY-----", "private-key"),
        ("pass​word=hunter2hunter2", "secret-assignment"),
    ],
)
def test_credentials_ids_and_cards_are_blocked(text: str, rule: str) -> None:
    verdict = SCAN.scan(["問題", text])

    assert not verdict.allowed
    assert rule in verdict.rule_ids
    assert text not in repr(verdict)


@pytest.mark.parametrize(
    "text",
    ["碩士在職專班的報名時間是什麼時候？", "身分證 A123456788", "卡號 4111 1111 1111 1112"],
)
def test_ordinary_text_and_failed_checksums_pass(text: str) -> None:
    assert SCAN.scan([text]).allowed


def test_texts_are_scanned_together_so_combinations_cannot_be_split() -> None:
    assert SCAN.scan(["customer"]).allowed
    assert SCAN.scan(["quotation"]).allowed
    assert "customer-quotation" in SCAN.scan(["customer", "quotation"]).rule_ids


def test_no_outbound_scan_allows_everything() -> None:
    assert NoOutboundScan().scan(["api_key = sk-abcdef123456", "A123456789"]).allowed


@pytest.mark.parametrize(
    "policy",
    [
        "rules: []",
        "version: v1\nenabled: true\nrules:\n  - {id: a, category: c, type: contains, value: x}",
        "version: v1\nrules:\n  - {id: a, category: c, type: unknown, value: x}",
        "version: v1\nrules:\n  - {id: a, category: c, type: regex, pattern: '('}",
        "version: v1\nrules:\n"
        "  - {id: a, category: c, type: contains, value: x}\n"
        "  - {id: a, category: c, type: contains, value: y}",
        "version: v1\nrules: [",
    ],
)
def test_an_invalid_policy_is_refused_at_load(policy: str) -> None:
    with pytest.raises(InvalidPolicyError):
        RuleBasedOutboundScan.from_yaml(policy)
