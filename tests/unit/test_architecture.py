"""Layering rules: domain imports the standard library only, application adds only domain."""

from __future__ import annotations

import ast
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "helpdesk"
STDLIB = frozenset(sys.stdlib_module_names)
ALLOWED_INTERNAL: dict[str, tuple[str, ...]] = {
    "domain": ("helpdesk.domain",),
    "application": ("helpdesk.domain", "helpdesk.application"),
}


def imported_modules(source: str, package: str) -> list[str]:
    """Absolute names of everything a module imports; relative imports are resolved."""

    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            relative = "." * node.level + (node.module or "")
            names.append(resolve_name(relative, package) if node.level else relative)
    return names


def violations(layer: str, source: str, package: str) -> list[str]:
    allowed = ALLOWED_INTERNAL[layer]
    return [
        name
        for name in imported_modules(source, package)
        if name.partition(".")[0] not in STDLIB
        and not any(name == prefix or name.startswith(f"{prefix}.") for prefix in allowed)
    ]


def package_of(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    parts = relative.parts if path.name == "__init__.py" else relative.parts[:-1]
    return ".".join(part for part in parts if part != "__init__")


@pytest.mark.parametrize("layer", sorted(ALLOWED_INTERNAL))
def test_inner_layers_import_only_what_they_may(layer: str) -> None:
    sources = sorted((PACKAGE_ROOT / layer).rglob("*.py"))
    assert sources, f"no modules found for the {layer} layer"

    found = {
        str(path.relative_to(PACKAGE_ROOT)): bad
        for path in sources
        if (bad := violations(layer, path.read_text(encoding="utf-8"), package_of(path)))
    }

    assert found == {}


@pytest.mark.parametrize(
    ("layer", "source"),
    [
        ("domain", "import sqlalchemy"),
        ("domain", "from pydantic import BaseModel"),
        ("domain", "from helpdesk.infrastructure.persistence import models"),
        ("domain", "from ..infrastructure.persistence import models"),
        ("domain", "from helpdesk.application import ports"),
        ("domain", "from helpdesk import config"),
        ("application", "from helpdesk.infrastructure import persistence"),
        ("application", "from helpdesk.config import Settings"),
        ("application", "import fastapi"),
    ],
)
def test_forbidden_imports_are_reported(layer: str, source: str) -> None:
    assert violations(layer, source, f"helpdesk.{layer}") != []


@pytest.mark.parametrize(
    ("layer", "source"),
    [
        ("domain", "from dataclasses import dataclass"),
        ("domain", "from .errors import DomainError"),
        ("domain", "from helpdesk.domain.documents import Chunk"),
        ("application", "from helpdesk.domain.conversations import Turn"),
        ("application", "from helpdesk.application.retrieval import ports"),
    ],
)
def test_permitted_imports_pass(layer: str, source: str) -> None:
    assert violations(layer, source, f"helpdesk.{layer}") == []
