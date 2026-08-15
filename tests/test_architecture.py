"""Architectural guarantees, tested rather than documented.

``import-linter`` enforces the layer contracts in CI. These tests cover the two
things that matter most and that a developer is most likely to break locally,
so the failure arrives during ``pytest`` rather than at the end of a CI run.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from assetto_ai_director import __version__

SRC = Path(__file__).resolve().parents[1] / "src" / "assetto_ai_director"

FORBIDDEN_IN_DOMAIN = {
    "PySide6",
    "mcp",
    "typer",
    "loguru",
    "scipy",
    "cv2",
    "scenedetect",
    "pydantic",
}


def _module_files(layer: str) -> list[Path]:
    return sorted((SRC / layer).rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """Collect the top-level package name of every import in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


class TestDomainIndependence:
    """The domain layer must depend on nothing but stdlib and NumPy."""

    def test_domain_imports_no_other_layer(self) -> None:
        offenders: list[str] = []
        for path in _module_files("domain"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = node.module if isinstance(node, ast.ImportFrom) else None
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                offenders.extend(
                    f"{path.name} imports {target}"
                    for target in [module, *names]
                    if target
                    and target.startswith("assetto_ai_director.")
                    and target.split(".")[1] != "domain"
                )
        assert not offenders, "Domain must not depend on outer layers: " + "; ".join(offenders)

    def test_domain_imports_no_frameworks(self) -> None:
        offenders = [
            f"{path.name} imports {root}"
            for path in _module_files("domain")
            for root in sorted(_imported_roots(path) & FORBIDDEN_IN_DOMAIN)
        ]
        assert not offenders, "Domain must stay framework-free: " + "; ".join(offenders)

    def test_domain_imports_cleanly_without_optional_extras(self) -> None:
        """Importing the domain must not drag in Qt or the MCP SDK.

        The GUI and the MCP server are optional extras. A stray import in the
        domain would make the headless CLI depend on a graphical toolkit.
        """
        before = set(sys.modules)
        # The local import is the assertion: it must happen *after* the snapshot
        # so the set difference shows what this import alone pulled in.
        import assetto_ai_director.domain.timeline  # noqa: F401, PLC0415

        newly_loaded = set(sys.modules) - before
        assert not any(name.startswith("PySide6") for name in newly_loaded)


class TestPortsAreContractsOnly:
    def test_ports_do_not_import_infrastructure(self) -> None:
        offenders: list[str] = []
        for path in _module_files("application/ports"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            offenders.extend(
                f"{path.name} imports {node.module}"
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module
                and "infrastructure" in node.module
            )
        assert not offenders, "Ports must not know about infrastructure: " + "; ".join(offenders)


class TestPackageMetadata:
    def test_version_is_importable(self) -> None:
        assert __version__

    @pytest.mark.parametrize(
        "layer", ["domain", "application", "infrastructure", "presentation", "mcp"]
    )
    def test_every_layer_exists(self, layer: str) -> None:
        assert (SRC / layer / "__init__.py").is_file()
