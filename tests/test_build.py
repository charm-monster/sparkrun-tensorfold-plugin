"""Building and verifying artifacts end to end, with real Docker and a test-only engine.

These run real containers (a small bash image) against real files in a temp directory. The only
things that are fake are the engine and its output, which is what a real profile supplies.
"""

from __future__ import annotations

import dataclasses
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from sparkrun_tensorfold_plugin.artifacts.build import ArtifactError, ensure_artifacts
from sparkrun_tensorfold_plugin.artifacts.jobs import LABEL
from sparkrun_tensorfold_plugin.profiles import REGISTRY
from sparkrun_tensorfold_plugin.profiles.model import (
    CommandSpec,
    Profile,
    Program,
    const,
)
from sparkrun_tensorfold_plugin.recipe_block import PrepareMode
from tests.fake_profile import MAGIC, fake_profile
from tests.helpers import Result, block_in, local_run

pytestmark = pytest.mark.docker

HOSTS = ["node-a", "node-b"]


def _docker_up() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0
    except FileNotFoundError:
        return False


if not _docker_up():
    pytest.skip("no Docker daemon", allow_module_level=True)


def containers() -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"label={LABEL}=1", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


@pytest.fixture(autouse=True)
def only_our_containers_are_cleaned() -> Iterator[None]:
    yield
    for name in containers():
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


class Recorder:
    """The real local shell, remembering every command it ran."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def __call__(self, host: str, command: str) -> Result:
        self.commands.append(command)
        return local_run(host, command)

    def started(self) -> list[str]:
        return [c for c in self.commands if "docker run" in c]


def ensure(
    root: Path,
    profile: Profile,
    recorder: Recorder,
    *,
    mode: PrepareMode = PrepareMode.BUILD,
    dry_run: bool = False,
    poll_seconds: float = 0.3,
    hosts: tuple[str, ...] = tuple(HOSTS),
) -> None:
    REGISTRY["fake"] = profile
    try:
        block, model = block_in(root, "fake")
        block = dataclasses.replace(block, prepare=mode)
        ensure_artifacts(
            recorder,
            list(hosts),
            profile,
            block,
            model,
            master_addr="192.0.2.1",
            dry_run=dry_run,
            poll_seconds=poll_seconds,
            heartbeat_seconds=60,
        )
    finally:
        del REGISTRY["fake"]


def cache_dir(root: Path) -> Path:
    return root / "cache" / "rank-cache"


def test_a_cold_build_produces_every_artifact_and_leaves_no_container(tmp_path: Path) -> None:
    ensure(tmp_path, fake_profile(), Recorder())
    assert (tmp_path / "assets" / "d.txt").read_text() == "hello!"
    assert (tmp_path / "bin" / "bin" / "tool").exists()
    for rank in (0, 1):
        data = (cache_dir(tmp_path) / f"rank{rank}of2-abc.bin").read_text()
        assert data == f"weights{MAGIC}"
    assert containers() == []


def named_hosts(count: int) -> tuple[str, ...]:
    return tuple(f"node-{i}" for i in range(count))


@pytest.mark.parametrize("count", [1, 3, 4])
def test_a_cold_build_covers_every_rank_of_any_node_count(tmp_path: Path, count: int) -> None:
    ensure(tmp_path, fake_profile(), Recorder(), hosts=named_hosts(count))
    for rank in range(count):
        data = (cache_dir(tmp_path) / f"rank{rank}of{count}-abc.bin").read_text()
        assert data == f"weights{MAGIC}"
    assert len(list(cache_dir(tmp_path).iterdir())) == count
    assert containers() == []


def test_some_ranks_present_among_four_is_never_started_over(tmp_path: Path) -> None:
    kept = [cache_dir(tmp_path) / f"rank{rank}of4-abc.bin" for rank in (0, 2)]
    kept[0].parent.mkdir(parents=True)
    for path in kept:
        path.write_text(f"precious{MAGIC}")
    before = [p.stat().st_mtime_ns for p in kept]
    recorder = Recorder()
    with pytest.raises(ArtifactError, match="present on node-0, node-2 but missing on node-1"):
        ensure(tmp_path, fake_profile(), recorder, hosts=named_hosts(4))
    assert [p.read_text() for p in kept] == [f"precious{MAGIC}"] * 2
    assert [p.stat().st_mtime_ns for p in kept] == before
    assert not any("sparkrun-tfp-build-cache" in c for c in recorder.started())


def test_a_second_run_starts_nothing_and_changes_nothing(tmp_path: Path) -> None:
    ensure(tmp_path, fake_profile(), Recorder())
    watched = [
        tmp_path / "assets" / "d.txt",
        tmp_path / "bin" / "bin" / "tool",
        cache_dir(tmp_path) / "rank0of2-abc.bin",
        cache_dir(tmp_path) / "rank1of2-abc.bin",
    ]
    before = [p.stat().st_mtime_ns for p in watched]
    second = Recorder()
    ensure(tmp_path, fake_profile(), second)
    assert second.started() == []
    assert [p.stat().st_mtime_ns for p in watched] == before


def test_verify_mode_reports_everything_missing_and_builds_nothing(tmp_path: Path) -> None:
    recorder = Recorder()
    with pytest.raises(ArtifactError) as raised:
        ensure(tmp_path, fake_profile(), recorder, mode=PrepareMode.VERIFY)
    message = str(raised.value)
    assert "tensorfold.prepare is 'verify'" in message
    for needle in ("data on node-a", "tool on node-a", "cache on node-b", "'build'"):
        assert needle in message
    assert recorder.started() == []
    assert not (tmp_path / "assets").exists()


def test_verify_mode_passes_when_everything_is_present(tmp_path: Path) -> None:
    ensure(tmp_path, fake_profile(), Recorder())
    again = Recorder()
    ensure(tmp_path, fake_profile(), again, mode=PrepareMode.VERIFY)
    assert again.started() == []


def test_a_partial_rank_cache_is_never_started_over(tmp_path: Path) -> None:
    existing = cache_dir(tmp_path) / "rank0of2-abc.bin"
    existing.parent.mkdir(parents=True)
    existing.write_text(f"precious{MAGIC}")
    before = existing.stat().st_mtime_ns
    recorder = Recorder()
    with pytest.raises(ArtifactError, match="present on node-a but missing on node-b"):
        ensure(tmp_path, fake_profile(), recorder)
    assert existing.read_text() == f"precious{MAGIC}"
    assert existing.stat().st_mtime_ns == before
    assert not any("sparkrun-tfp-build-cache" in c for c in recorder.started())


def test_a_failing_build_reports_the_exit_code_and_its_output(tmp_path: Path) -> None:
    broken = fake_profile(data_script="echo boom-from-the-script >&2; exit 3")
    with pytest.raises(ArtifactError) as raised:
        ensure(tmp_path, broken, Recorder())
    assert "data failed on" in str(raised.value)
    assert "exit code 3" in str(raised.value)
    assert "boom-from-the-script" in str(raised.value)
    assert containers() == []


def test_a_build_that_never_finishes_is_stopped_at_its_timeout(tmp_path: Path) -> None:
    slow = fake_profile(data_script="sleep 120", data_timeout=2)
    with pytest.raises(ArtifactError, match="data did not finish within 2 seconds"):
        ensure(tmp_path, slow, Recorder())
    assert containers() == []


def test_an_engine_that_exits_early_fails_the_whole_build(tmp_path: Path) -> None:
    dying = CommandSpec(
        program=Program(base=None, relative="bash"),
        subcommand="-c",
        positionals=(const("echo engine-gave-up >&2; exit 5"),),
        flags=(),
        env=(),
    )
    with pytest.raises(ArtifactError) as raised:
        ensure(tmp_path, fake_profile(engine=dying), Recorder())
    assert "stopped (exit code 5)" in str(raised.value)
    assert "engine-gave-up" in str(raised.value)
    assert containers() == []


def test_a_dry_run_touches_nothing(tmp_path: Path) -> None:
    recorder = Recorder()
    ensure(tmp_path, fake_profile(), recorder, dry_run=True)
    assert recorder.commands == []
    assert not (tmp_path / "assets").exists()


def test_a_build_that_does_not_produce_the_artifact_is_an_error(tmp_path: Path) -> None:
    lazy = fake_profile(data_script="true")
    with pytest.raises(ArtifactError, match="data still fails verification after its build"):
        ensure(tmp_path, lazy, Recorder())
