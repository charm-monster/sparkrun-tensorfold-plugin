"""Artifact definitions: build-script helpers, and validation of artifact data in any profile."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from sparkrun_tensorfold_profile_hello_engine import HELLO_ENGINE

from sparkrun_tensorfold_plugin.artifacts.context import script_context
from sparkrun_tensorfold_plugin.profiles.model import (
    Access,
    Artifact,
    CoordinatedPlan,
    Executables,
    FileSha256,
    Mount,
    Profile,
    ProfileError,
    Program,
    ScriptPlan,
    SizeManifest,
)
from sparkrun_tensorfold_plugin.profiles.scripts import fetch_verified, hf_download, user_script
from tests.helpers import block_in

REVISION = "0123456789abcdef0123456789abcdef01234567"


def artifact(name: str) -> Artifact:
    return next(a for a in HELLO_ENGINE.artifacts if a.name == name)


def test_artifacts_are_declared_in_dependency_order() -> None:
    assert [a.name for a in HELLO_ENGINE.artifacts] == [
        "model",
        "assets",
        "kit",
        "lookup_table",
        "binaries",
        "rank_cache",
    ]


def test_only_the_rank_cache_needs_every_rank_alive_to_build() -> None:
    together = [a.name for a in HELLO_ENGINE.artifacts if isinstance(a.plan, CoordinatedPlan)]
    assert together == ["rank_cache"]


def test_every_script_renders_with_a_real_context(tmp_path: Path) -> None:
    block, model = block_in(tmp_path)
    fields = script_context(
        HELLO_ENGINE, block, model, rank=1, world_size=2, master_addr="192.0.2.1"
    )
    rendered_any = False
    for art in HELLO_ENGINE.artifacts:
        if isinstance(art.plan, ScriptPlan):
            rendered = art.plan.script.format(**fields)
            rendered_any = True
            for name in HELLO_ENGINE.script_fields():
                assert "{" + name + "}" not in rendered, (art.name, name)
    assert rendered_any
    model_script = artifact("model").plan.script.format(**fields)  # type: ignore[union-attr]
    assert f"DEST={model}\n" in model_script


# --- the build-script helpers ------------------------------------------------------------------


def test_a_download_stops_at_a_pinned_revision_and_never_replaces_a_directory() -> None:
    script = hf_download(repo="org/name", revision=REVISION, dest="model")
    assert f"--revision {REVISION}" in script
    assert 'rmdir "$DEST"' in script and "refusing to replace" in script
    assert 'STAGE="$DEST.partial"' in script
    assert script.index('mv "$SRC" "$DEST"') < script.index('rm -rf "$STAGE"')


def test_a_download_can_take_only_some_files() -> None:
    script = hf_download(repo="org/name", revision=REVISION, dest="x", include=("a.bin", "b/*"))
    assert "--include a.bin --include 'b/*'" in script


def test_a_download_can_take_only_a_folder_of_the_repository() -> None:
    script = hf_download(repo="org/name", revision=REVISION, dest="x", repo_subdir="kit")
    assert 'SRC="$STAGE/kit"' in script
    assert 'SRC="$STAGE"' not in script


def test_a_fetched_file_is_verified_before_anything_runs_it() -> None:
    script = user_script(
        fetch_verified(url="https://example.invalid/tool.py", sha256="a" * 64, to="$HOME/tool.py"),
        "python3 $HOME/tool.py",
    )
    assert "curl -fsSL https://example.invalid/tool.py -o $HOME/tool.py" in script
    assert script.index("sha256sum -c -") < script.index("python3 $HOME/tool.py")


def test_a_user_script_gets_a_writable_home_and_stops_on_error() -> None:
    script = user_script("true")
    assert script.startswith("set -euo pipefail\n")
    assert 'mkdir -p "$HOME"' in script


def test_the_example_rank_cache_build_mounts_the_cache_and_waits_for_its_inputs() -> None:
    plan = artifact("rank_cache").plan
    assert isinstance(plan, CoordinatedPlan)
    assert plan.needs == ("model", "assets", "lookup_table")
    assert Mount("cache_dir", "/var/cache/hello", Access.RW) in plan.mounts


# --- validation of artifact data -------------------------------------------------------------


def replace_artifact(profile: Profile, name: str, **changes: object) -> Profile:
    arts = tuple(
        dataclasses.replace(a, **changes) if a.name == name else a  # type: ignore[arg-type]
        for a in profile.artifacts
    )
    return dataclasses.replace(profile, artifacts=arts)


def replace_plan(profile: Profile, name: str, **changes: object) -> Profile:
    art = next(a for a in profile.artifacts if a.name == name)
    return replace_artifact(profile, name, plan=dataclasses.replace(art.plan, **changes))  # type: ignore[arg-type]


def test_an_artifact_needing_a_later_one_is_rejected() -> None:
    broken = replace_plan(HELLO_ENGINE, "model", needs=("rank_cache",))
    with pytest.raises(ProfileError, match="model needs rank_cache, which is not declared before"):
        broken.validate()


def test_a_duplicate_artifact_is_rejected() -> None:
    broken = dataclasses.replace(
        HELLO_ENGINE, artifacts=(*HELLO_ENGINE.artifacts, HELLO_ENGINE.artifacts[0])
    )
    with pytest.raises(ProfileError, match="artifact model is declared twice"):
        broken.validate()


def test_a_mount_of_an_unknown_path_is_rejected() -> None:
    broken = replace_plan(HELLO_ENGINE, "model", mounts=(Mount("nope", None, Access.RW),))
    with pytest.raises(ProfileError, match="mounts unknown path nope"):
        broken.validate()


def test_a_script_field_that_does_not_exist_is_rejected() -> None:
    broken = replace_plan(HELLO_ENGINE, "model", script="echo {nope}")
    with pytest.raises(ProfileError, match=r"script uses unknown field \{nope\}"):
        broken.validate()


def test_a_check_on_an_unknown_target_is_rejected() -> None:
    broken = replace_artifact(HELLO_ENGINE, "model", check=SizeManifest("nope", (("a", 1),)))
    with pytest.raises(ProfileError, match="check on unknown target nope"):
        broken.validate()


def test_a_manifest_entry_with_whitespace_is_rejected() -> None:
    broken = replace_artifact(HELLO_ENGINE, "model", check=SizeManifest("model", (("a b", 1),)))
    with pytest.raises(ProfileError, match="bad manifest entry"):
        broken.validate()


def test_a_short_sha256_is_rejected() -> None:
    broken = replace_artifact(HELLO_ENGINE, "lookup_table", check=FileSha256("lookup_table", "abc"))
    with pytest.raises(ProfileError, match="64 hex characters"):
        broken.validate()


def test_an_executable_check_outside_a_path_key_is_rejected() -> None:
    broken = replace_artifact(
        HELLO_ENGINE, "binaries", check=Executables(rank0=(Program(None, "ls"),), others=())
    )
    with pytest.raises(ProfileError, match="executable check needs a program under a path key"):
        broken.validate()
