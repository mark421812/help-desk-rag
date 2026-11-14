"""唯讀掃描文件根目錄。只收一般檔案，不跟隨 symbolic link，隱藏檔與暫存檔一律略過。"""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

from helpdesk.application.ingestion.ports import DiscoveredFile
from helpdesk.domain.errors import DomainValidationError

_TEMPORARY_SUFFIXES: Final = (".part", ".swp", ".tmp", "~")


class LocalFolderSource:
    def __init__(self, root: Path, extensions: Iterable[str]) -> None:
        self._root = root.resolve(strict=True)
        if not self._root.is_dir():
            raise DomainValidationError("DOCUMENTS_ROOT must be a directory")
        self._extensions = frozenset(extension.casefold() for extension in extensions)

    def scan(self) -> tuple[DiscoveredFile, ...]:
        discovered: list[DiscoveredFile] = []
        for directory, directories, filenames in os.walk(
            self._root, topdown=True, followlinks=False, onerror=_raise_scan_error
        ):
            directory_path = Path(directory)
            # 改寫 directories[:] 讓 os.walk 不進入隱藏資料夾或 symlink。
            directories[:] = sorted(
                name
                for name in directories
                if not name.startswith(".") and not (directory_path / name).is_symlink()
            )
            for filename in sorted(filenames):
                path = directory_path / filename
                if not self._accepts(filename) or path.is_symlink():
                    continue
                resolved = path.resolve(strict=True)
                self._require_inside_root(resolved)
                if resolved.is_file():
                    discovered.append(self._snapshot(resolved))
        return tuple(sorted(discovered, key=lambda file: file.relative_path))

    def discover(self, relative_path: str) -> DiscoveredFile | None:
        relative = PurePosixPath(relative_path)
        if (
            not relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_path
            or any(part.startswith(".") for part in relative.parts)
        ):
            raise DomainValidationError("the path must be a visible, normalized relative path")
        if not self._accepts(relative.name):
            return None
        # 逐層檢查 symlink，中間的資料夾也不能繞出根目錄。
        candidate = self._root
        for part in relative.parts:
            candidate /= part
            if candidate.is_symlink():
                return None
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            return None
        self._require_inside_root(resolved)
        return self._snapshot(resolved) if resolved.is_file() else None

    def _accepts(self, filename: str) -> bool:
        lowered = filename.casefold()
        return (
            not (
                filename.startswith(".")
                or filename.startswith("~$")
                or lowered.endswith(_TEMPORARY_SUFFIXES)
            )
            and PurePosixPath(lowered).suffix in self._extensions
        )

    def _snapshot(self, path: Path) -> DiscoveredFile:
        stat = path.stat()
        return DiscoveredFile(
            relative_path=path.relative_to(self._root).as_posix(),
            path=path,
            file_size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        )

    def _require_inside_root(self, path: Path) -> None:
        if not path.is_relative_to(self._root):
            raise DomainValidationError("a document escapes DOCUMENTS_ROOT")


def _raise_scan_error(error: OSError) -> None:
    raise DomainValidationError("unable to scan DOCUMENTS_ROOT") from error
