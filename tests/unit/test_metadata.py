"""metadata YAML 的解析與兩種 ACL 模式的補值規則。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from helpdesk.application.ingestion.metadata import OptionalMetadata, RequiredMetadata
from helpdesk.application.ingestion.ports import DeclaredMetadata
from helpdesk.domain.access import AccessRule, PermissionEffect, PrincipalType
from helpdesk.domain.documents import Confidentiality, DocumentStatus
from helpdesk.domain.ingestion import InvalidMetadataError, MetadataMissingError
from helpdesk.infrastructure.metadata.yaml_sidecars import YamlSidecarMetadata

LEGACY_SIDECAR = """title: 政治大學115學年度招生簡章
document_type: policy
owner_department: HR
document_owner: hr@example.com
confidentiality: internal
status: published
language: zh-TW
external_llm_allowed: true
"""


def test_folder_defaults_and_file_sidecar_merge_with_the_closest_file_winning(
    tmp_path: Path,
) -> None:
    department = tmp_path / "hr"
    department.mkdir()
    (department / "leave.md").write_text("請假規章", encoding="utf-8")
    (tmp_path / ".ingestion.yaml").write_text(
        "metadata:\n  document_type: policy\n  status: published\n  language: en\n",
        encoding="utf-8",
    )
    (department / ".ingestion.yaml").write_text("metadata:\n  language: zh-TW\n", encoding="utf-8")
    (department / "leave.md.metadata.yaml").write_text(
        """metadata:
  title: 請假管理辦法
  version_number: "2.1"
  effective_at: 2026-08-02
  acl:
    - principal_type: group
      principal_id: hr
    - principal_type: user
      principal_id: intern@example.com
      effect: deny
""",
        encoding="utf-8",
    )

    declared = YamlSidecarMetadata(tmp_path).declared("hr/leave.md")

    assert declared == DeclaredMetadata(
        title="請假管理辦法",
        document_type="policy",
        language="zh-TW",
        status=DocumentStatus.PUBLISHED,
        effective_at=date(2026, 8, 2),
        access_rules=(
            AccessRule(PrincipalType.GROUP, "hr"),
            AccessRule(PrincipalType.USER, "intern@example.com", PermissionEffect.DENY),
        ),
    )


def test_a_file_without_metadata_declares_nothing(tmp_path: Path) -> None:
    (tmp_path / "BookRAG.pdf").write_bytes(b"%PDF")

    assert YamlSidecarMetadata(tmp_path).declared("BookRAG.pdf") == DeclaredMetadata()


@pytest.mark.parametrize(
    ("sidecar", "message"),
    [
        ("title: Policy\nmade_up_field: value\n", "unknown metadata fields"),
        ("title: [unclosed\n", "not valid UTF-8 YAML"),
        ("status: approved\n", "status must be one of"),
        ("external_llm_allowed: 'yes'\n", "external_llm_allowed"),
        ("acl:\n  - principal_type: team\n    principal_id: x\n", "principal_type"),
        ("metadata:\n  title: a\nextra: b\n", "single 'metadata' key"),
    ],
)
def test_invalid_metadata_files_are_reported(tmp_path: Path, sidecar: str, message: str) -> None:
    (tmp_path / "policy.txt").write_text("內容", encoding="utf-8")
    (tmp_path / "policy.txt.metadata.yaml").write_text(sidecar, encoding="utf-8")

    with pytest.raises(InvalidMetadataError, match=message):
        YamlSidecarMetadata(tmp_path).declared("policy.txt")


def test_the_fingerprint_follows_every_metadata_file_that_applies(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    metadata = YamlSidecarMetadata(tmp_path)
    without = metadata.fingerprint("a.txt")
    (tmp_path / ".ingestion.yaml").write_text("language: zh-TW\n", encoding="utf-8")
    with_folder = metadata.fingerprint("a.txt")
    (tmp_path / "a.txt.metadata.yaml").write_text("title: [broken\n", encoding="utf-8")

    assert len({without, with_folder, metadata.fingerprint("a.txt")}) == 3


def test_acl_off_needs_no_metadata_and_publishes_under_the_file_name() -> None:
    path = "manuals/BookRAG_繁體中文完整翻譯.pdf"

    metadata = OptionalMetadata().resolve(DeclaredMetadata(), path)

    assert metadata.title == "BookRAG_繁體中文完整翻譯"
    assert metadata.status is DocumentStatus.PUBLISHED
    assert metadata.external_llm_allowed is True
    assert (metadata.document_type, metadata.language) == ("general", "zh-TW")
    assert metadata.confidentiality is Confidentiality.INTERNAL


def test_acl_off_keeps_declared_values_and_access_rules() -> None:
    rules = (AccessRule(PrincipalType.GROUP, "hr"),)
    declared = DeclaredMetadata(
        title="內部規章",
        status=DocumentStatus.DRAFT,
        external_llm_allowed=False,
        confidentiality=Confidentiality.CONFIDENTIAL,
        access_rules=rules,
    )

    metadata = OptionalMetadata().resolve(declared, "rules.pdf")

    assert (metadata.title, metadata.status, metadata.external_llm_allowed) == (
        "內部規章",
        DocumentStatus.DRAFT,
        False,
    )
    assert metadata.access_rules == rules


def test_acl_on_requires_every_field_and_never_guesses_a_title() -> None:
    with pytest.raises(MetadataMissingError) as missing:
        RequiredMetadata().resolve(DeclaredMetadata(), "BookRAG.pdf")
    with pytest.raises(MetadataMissingError) as partial:
        RequiredMetadata().resolve(
            DeclaredMetadata(document_type="policy", owner_department="HR", document_owner="hr"),
            "BookRAG.pdf",
        )

    assert missing.value.missing_fields == (
        "document_owner",
        "document_type",
        "owner_department",
        "title",
    )
    assert partial.value.missing_fields == ("title",)


def test_acl_on_fills_unwritten_governance_fields_with_the_safe_defaults(tmp_path: Path) -> None:
    (tmp_path / "brochure.pdf").write_bytes(b"%PDF")
    (tmp_path / "brochure.pdf.metadata.yaml").write_text(
        "title: 簡章\ndocument_type: policy\nowner_department: HR\ndocument_owner: hr\n",
        encoding="utf-8",
    )

    declared = YamlSidecarMetadata(tmp_path).declared("brochure.pdf")
    metadata = RequiredMetadata().resolve(declared, "brochure.pdf")

    assert (metadata.status, metadata.external_llm_allowed) == (DocumentStatus.DRAFT, False)


def test_a_complete_sidecar_resolves_the_same_in_both_modes(tmp_path: Path) -> None:
    (tmp_path / "brochure.pdf").write_bytes(b"%PDF")
    (tmp_path / "brochure.pdf.metadata.yaml").write_text(LEGACY_SIDECAR, encoding="utf-8")
    declared = YamlSidecarMetadata(tmp_path).declared("brochure.pdf")

    assert OptionalMetadata().resolve(declared, "brochure.pdf") == RequiredMetadata().resolve(
        declared, "brochure.pdf"
    )


def test_metadata_that_breaks_a_domain_rule_is_invalid_in_both_modes() -> None:
    declared = DeclaredMetadata(
        title="機密",
        document_type="policy",
        owner_department="HR",
        document_owner="hr",
        confidentiality=Confidentiality.RESTRICTED,
    )

    for policy in (OptionalMetadata(), RequiredMetadata()):
        with pytest.raises(InvalidMetadataError, match="allow rule"):
            policy.resolve(declared, "secret.pdf")
