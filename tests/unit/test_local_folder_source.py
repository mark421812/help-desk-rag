"""本機資料夾來源：列出檔案、路徑安全與符號連結。"""

from __future__ import annotations

from pathlib import Path

import pytest

from helpdesk.domain.errors import DomainValidationError
from helpdesk.infrastructure.sources.local_folder import LocalFolderSource

EXTENSIONS = (".pdf", ".docx", ".txt", ".md")


def test_scan_returns_supported_regular_files_in_stable_order(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "a.MD").write_text("a", encoding="utf-8")
    (tmp_path / "ignore.xlsx").write_text("x", encoding="utf-8")
    (tmp_path / "z.txt.metadata.yaml").write_text("title: z", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("hidden", encoding="utf-8")
    (tmp_path / "~$draft.docx").write_text("lock", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "guide.pdf").write_bytes(b"pdf")
    (tmp_path / "linked.txt").symlink_to(tmp_path / "z.txt")

    files = LocalFolderSource(tmp_path, EXTENSIONS).scan()

    assert [file.relative_path for file in files] == ["a.MD", "nested/guide.pdf", "z.txt"]
    assert files[2].source_version.endswith(";size:1")


def test_discover_resolves_exactly_one_visible_relative_file(tmp_path: Path) -> None:
    nested = tmp_path / "policies"
    nested.mkdir()
    expected = nested / "leave.md"
    expected.write_text("leave policy", encoding="utf-8")
    source = LocalFolderSource(tmp_path, EXTENSIONS)

    discovered = source.discover("policies/leave.md")

    assert discovered is not None
    assert discovered.path == expected.resolve()
    assert source.discover("policies/missing.md") is None
    for unsafe in ("../outside.md", "/etc/passwd", "policies/./leave.md", ".hidden/x.md"):
        with pytest.raises(DomainValidationError):
            source.discover(unsafe)


def test_symlinked_folders_are_neither_scanned_nor_discovered(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    source = LocalFolderSource(root, EXTENSIONS)

    assert source.scan() == ()
    assert source.discover("escape/secret.txt") is None
