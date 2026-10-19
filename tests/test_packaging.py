"""The package metadata sparkrun reads to discover and install this plugin."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "src" / "sparkrun_tensorfold_plugin" / "__init__.py"


def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_the_api_version_is_a_literal_int_sparkrun_can_read_without_importing() -> None:
    tree = ast.parse(INIT.read_text())
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "SPARKRUN_PLUGIN_API_VERSION" for t in node.targets
        )
    ]
    assert values == [1]


def test_the_entry_point_names_the_module_and_plugin_id() -> None:
    entry = pyproject()["project"]["entry-points"]["sparkrun.plugins"]
    assert entry == {"tensorfold-plugin": "sparkrun_tensorfold_plugin"}


def test_the_version_has_one_home() -> None:
    assert "__version__" not in INIT.read_text()
    assert "version" not in pyproject()["project"].get("dynamic", [])


def test_the_runtime_dependency_is_only_sparkrun() -> None:
    assert pyproject()["project"]["dependencies"] == ["sparkrun>=0.3.10,<0.5"]


def test_sparkrun_0_4_reads_the_managed_plugin_manifest() -> None:
    manifest = pytest.importorskip("sparkrun.core.plugin_manifest")
    metadata = manifest.read_manifest(ROOT)
    assert metadata.name == "tensorfold-plugin"
    assert metadata.module == "sparkrun_tensorfold_plugin"
    assert metadata.distribution == "sparkrun-tensorfold-plugin"
    assert metadata.plugin_api == 1
    assert metadata.requires_sparkrun == ">=0.4.0,<0.5"
