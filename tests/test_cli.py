"""The plugin through the real `sparkrun` command line, under an isolated HOME.

Each test runs the installed sparkrun as a subprocess with its own empty home directory, so the
developer's real sparkrun configuration is never read or written, and with telemetry off. The two
ways a plugin is made available are covered: a plugin directory (works on every supported version)
and an installed package enabled by name (sparkrun 0.4 and later).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from sparkrun_tensorfold_profile_hello_engine import HELLO_ENGINE

from tests.conftest import EXAMPLE, PROFILE_PLUGIN_DIR
from tests.test_commands import MASTER, head_command

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PLACEHOLDER_HEAD = head_command(2).replace(MASTER, "'<rank-0-fabric-address>'")


def sparkrun(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home), "SPARKRUN_NO_TELEMETRY": "1"}
    return subprocess.run(
        [sys.executable, "-m", "sparkrun", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=300,
        check=False,
    )


def output(done: subprocess.CompletedProcess[str]) -> str:
    return done.stdout + done.stderr


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


@pytest.fixture
def via_plugin_directory(home: Path) -> Path:
    """Make the plugin and the example profile package available the way every version supports."""
    write_plugin_paths(home, [SRC, PROFILE_PLUGIN_DIR])
    return home


def write_plugin_paths(home: Path, paths: list[Path]) -> None:
    config = home / ".config" / "sparkrun"
    config.mkdir(parents=True)
    (config / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "features": {"core.external_plugins": True},
                "plugins": {"paths": [str(path) for path in paths]},
            }
        )
    )


def serve_command(text: str) -> str:
    lines = text.splitlines()
    return lines[lines.index("Serve command:") + 1].strip()


def test_the_example_validates(via_plugin_directory: Path) -> None:
    done = sparkrun(via_plugin_directory, "recipe", "validate", str(EXAMPLE))
    assert done.returncode == 0, output(done)
    assert "is valid" in output(done)


def test_a_dry_run_launches_the_cluster_and_shows_the_rank_zero_command(
    via_plugin_directory: Path,
) -> None:
    done = sparkrun(
        via_plugin_directory,
        "run", str(EXAMPLE), "--hosts", "node-a,node-b", "--dry-run", "--no-follow",
    )  # fmt: skip
    text = output(done)
    assert done.returncode == 0, text
    assert "Runtime:   tensorfold-plugin" in text
    # The profile's image reaches sparkrun's image preparation on every supported version.
    assert f"Distributing image {HELLO_ENGINE.image} to 2 host(s)" in text
    assert "node-a" in text and "node-b" in text
    assert "[5/6] Launching tensorfold-plugin runtime" in text
    assert serve_command(text) == PLACEHOLDER_HEAD


def test_a_recipe_that_cannot_launch_is_refused_by_name(
    via_plugin_directory: Path, tmp_path: Path
) -> None:
    data = yaml.safe_load(EXAMPLE.read_text())
    data["readiness"] = {"inference": True}
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(data))
    done = sparkrun(
        via_plugin_directory,
        "run", str(broken), "--hosts", "node-a,node-b", "--dry-run", "--no-follow",
    )  # fmt: skip
    assert done.returncode != 0
    assert "readiness.inference must be false" in output(done)


def test_one_host_is_refused_when_the_profile_does_not_run_on_one(
    via_plugin_directory: Path,
) -> None:
    done = sparkrun(
        via_plugin_directory,
        "run", str(EXAMPLE), "--hosts", "node-a", "--dry-run", "--no-follow",
    )  # fmt: skip
    assert done.returncode != 0
    assert "profile hello-engine runs on [2, 4] nodes, not 1" in output(done)


def test_without_the_plugin_sparkrun_does_not_know_the_runtime(home: Path) -> None:
    done = sparkrun(
        home,
        "run", str(EXAMPLE), "--hosts", "node-a,node-b", "--dry-run", "--no-follow",
    )  # fmt: skip
    assert "Unknown runtime: 'tensorfold-plugin'" in output(done)


def test_an_installed_plugin_is_listed_after_it_is_enabled(home: Path) -> None:
    pytest.importorskip("sparkrun.core.installed_plugins")
    assert "tensorfold-plugin" not in output(sparkrun(home, "plugins", "list"))
    enabled = sparkrun(home, "plugins", "enable", "tensorfold-plugin")
    assert enabled.returncode == 0, output(enabled)
    listed = output(sparkrun(home, "plugins", "list"))
    row = next(line for line in listed.splitlines() if line.startswith("tensorfold-plugin"))
    assert row.split() == [
        "tensorfold-plugin", "0.1.0", "installed", "enabled", "compatible", "none"
    ]  # fmt: skip


def test_an_enabled_installed_plugin_validates_and_dry_runs(home: Path) -> None:
    pytest.importorskip("sparkrun.core.installed_plugins")
    # The plugin is installed and enabled by name; its profile comes from a plugin directory.
    write_plugin_paths(home, [PROFILE_PLUGIN_DIR])
    assert sparkrun(home, "plugins", "enable", "tensorfold-plugin").returncode == 0
    valid = sparkrun(home, "recipe", "validate", str(EXAMPLE))
    assert valid.returncode == 0 and "is valid" in output(valid)
    done = sparkrun(
        home, "run", str(EXAMPLE), "--hosts", "node-a,node-b", "--dry-run", "--no-follow"
    )
    assert done.returncode == 0, output(done)
    assert serve_command(output(done)) == PLACEHOLDER_HEAD
