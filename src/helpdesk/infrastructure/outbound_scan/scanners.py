"""由 `policy.yaml` 宣告規則的 CPU 掃描器。

規則檔在建立掃描器時解析，格式不對就讓程序啟動失敗，不會帶著壞掉的規則上線。掃描時把所有
文字合成一份、做 NFKC 與去除零寬字元後比對；掃描本身出錯一律視為擋下。判定只帶規則代號，
不帶命中的原文。
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Final, Literal, Self

import yaml

from helpdesk.application.outbound_scan import ScanVerdict

ZERO_WIDTH: Final = re.compile(r"[​-‏⁠﻿]")
SUPPORTED_VALIDATORS: Final = frozenset({"taiwan_id_checksum", "luhn_checksum", "private_ip"})
MAX_RULES: Final = 500
MAX_PATTERN_LENGTH: Final = 4_096
MAX_LITERAL_LENGTH: Final = 4_096

RuleType = Literal["regex", "all_regex", "contains", "all_contains"]


class InvalidPolicyError(ValueError):
    """規則檔不符合格式。"""


class NoOutboundScan:
    def scan(self, texts: Sequence[str]) -> ScanVerdict:
        return ScanVerdict(policy_version="disabled")


@dataclass(frozen=True, slots=True)
class ConfiguredRule:
    rule_id: str
    category: str
    rule_type: RuleType
    patterns: tuple[re.Pattern[str], ...] = ()
    values: tuple[str, ...] = ()
    validator: str | None = None

    def matches(self, normalized: str) -> bool:
        if self.rule_type == "regex":
            pattern = self.patterns[0]
            if self.validator is None:
                return pattern.search(normalized) is not None
            return any(
                _validator_matches(self.validator, match.group())
                for match in pattern.finditer(normalized)
            )
        if self.rule_type == "all_regex":
            return all(pattern.search(normalized) is not None for pattern in self.patterns)
        if self.rule_type == "contains":
            return self.values[0] in normalized
        return all(value in normalized for value in self.values)


class RuleBasedOutboundScan:
    def __init__(self, version: str, rules: tuple[ConfiguredRule, ...]) -> None:
        self._version = version
        self._rules = rules

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        try:
            payload: Any = yaml.safe_load(text)
            if not isinstance(payload, dict) or set(payload) != {"version", "rules"}:
                raise InvalidPolicyError("the policy needs exactly version and rules")
            raw_rules = payload["rules"]
            if not isinstance(raw_rules, list) or not 1 <= len(raw_rules) <= MAX_RULES:
                raise InvalidPolicyError(f"the policy needs 1 to {MAX_RULES} rules")
            rules = tuple(_parse_rule(item) for item in raw_rules)
            version = _required_text(payload["version"], "version")
        except (yaml.YAMLError, TypeError, KeyError, re.error) as exc:
            raise InvalidPolicyError("the outbound scan policy is not valid") from exc
        rule_ids = [rule.rule_id for rule in rules]
        if len(set(rule_ids)) != len(rule_ids):
            raise InvalidPolicyError("rule ids must be unique")
        return cls(version, rules)

    @classmethod
    def packaged(cls) -> Self:
        policy = files("helpdesk.infrastructure.outbound_scan").joinpath("policy.yaml")
        return cls.from_yaml(policy.read_text(encoding="utf-8"))

    def scan(self, texts: Sequence[str]) -> ScanVerdict:
        try:
            normalized = _normalize("\n".join(texts))
            hits = sorted(
                {(rule.rule_id, rule.category) for rule in self._rules if rule.matches(normalized)}
            )
        except Exception:
            return ScanVerdict(self._version, ("scanner_failure",), ("policy_error",))
        return ScanVerdict(
            policy_version=self._version,
            rule_ids=tuple(rule_id for rule_id, _ in hits),
            categories=tuple(sorted({category for _, category in hits})),
        )


def _parse_rule(value: Any) -> ConfiguredRule:
    if not isinstance(value, dict):
        raise InvalidPolicyError("a rule must be a mapping")
    rule_type = value.get("type")
    rule_id = _required_text(value.get("id"), "rule id")
    category = _required_text(value.get("category"), "rule category")
    if rule_type == "regex":
        allowed = {"id", "category", "type", "pattern", "ignore_case", "validator"}
        if not set(value) <= allowed or "pattern" not in value:
            raise InvalidPolicyError("a regex rule has invalid fields")
        validator = value.get("validator")
        if validator is not None and validator not in SUPPORTED_VALIDATORS:
            raise InvalidPolicyError("the regex rule validator is not supported")
        return ConfiguredRule(
            rule_id,
            category,
            "regex",
            patterns=(_compile(value["pattern"], value.get("ignore_case", False)),),
            validator=validator,
        )
    if rule_type == "all_regex":
        if set(value) != {"id", "category", "type", "patterns", "ignore_case"}:
            raise InvalidPolicyError("an all_regex rule has invalid fields")
        raw = value["patterns"]
        if not isinstance(raw, list) or len(raw) < 2:
            raise InvalidPolicyError("all_regex needs at least two patterns")
        patterns = tuple(_compile(pattern, value["ignore_case"]) for pattern in raw)
        return ConfiguredRule(rule_id, category, "all_regex", patterns=patterns)
    if rule_type == "contains":
        if set(value) != {"id", "category", "type", "value"}:
            raise InvalidPolicyError("a contains rule has invalid fields")
        return ConfiguredRule(
            rule_id, category, "contains", values=(_normalize(_literal(value["value"])),)
        )
    if rule_type == "all_contains":
        if set(value) != {"id", "category", "type", "values"}:
            raise InvalidPolicyError("an all_contains rule has invalid fields")
        raw = value["values"]
        if not isinstance(raw, list) or len(raw) < 2:
            raise InvalidPolicyError("all_contains needs at least two values")
        literals = tuple(_normalize(_literal(item)) for item in raw)
        return ConfiguredRule(rule_id, category, "all_contains", values=literals)
    raise InvalidPolicyError("the rule type is not supported")


def _compile(value: Any, ignore_case: Any) -> re.Pattern[str]:
    pattern = _required_text(value, "regex pattern")
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise InvalidPolicyError("the regex pattern is too long")
    if not isinstance(ignore_case, bool):
        raise InvalidPolicyError("ignore_case must be true or false")
    return re.compile(pattern, re.IGNORECASE if ignore_case else 0)


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidPolicyError(f"{field} must be non-blank text")
    return value.strip()


def _literal(value: Any) -> str:
    literal = _required_text(value, "literal value")
    if len(literal) > MAX_LITERAL_LENGTH:
        raise InvalidPolicyError("the literal value is too long")
    return literal


def _normalize(value: str) -> str:
    return ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", value)).casefold()


def _validator_matches(validator: str, value: str) -> bool:
    if validator == "taiwan_id_checksum":
        return _valid_taiwan_id(value)
    if validator == "luhn_checksum":
        return _valid_luhn(value)
    try:
        return ipaddress.ip_address(value).is_private
    except ValueError:
        return False


TAIWAN_ID_LETTER_CODES: Final = dict(zip("ABCDEFGHJKLMNPQRSTUVXYWZIO", range(10, 36), strict=True))


def _valid_luhn(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _valid_taiwan_id(value: str) -> bool:
    normalized = value.upper()
    if len(normalized) != 10 or normalized[0] not in TAIWAN_ID_LETTER_CODES:
        return False
    code = TAIWAN_ID_LETTER_CODES[normalized[0]]
    digits = [code // 10, code % 10, *(int(character) for character in normalized[1:])]
    weights = (1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1)
    return sum(digit * weight for digit, weight in zip(digits, weights, strict=True)) % 10 == 0
