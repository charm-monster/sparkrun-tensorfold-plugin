"""Artifact checks, run as real shell scripts against real files in a temp directory."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from sparkrun_tensorfold_plugin.artifacts.verify import verify
from sparkrun_tensorfold_plugin.profiles.model import (
    Executables,
    FileSha256,
    Program,
    RankCacheFooter,
    Sha256Manifest,
    SizeManifest,
)
from tests.helpers import block_in, local_run

MAGIC = "TFDSRK01"


def check(root: Path, spec: object, rank: int = 0, world: int = 2) -> tuple[bool, str]:
    block, model = block_in(root)
    result = verify(
        local_run,
        "local",
        spec,
        block.profile,
        block,
        model,
        rank=rank,
        world_size=world,  # type: ignore[arg-type]
    )
    return result.ok, result.detail


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# --- SizeManifest -----------------------------------------------------------------------------


def sizes(root: Path, files: dict[str, bytes], listed: dict[str, int]) -> tuple[bool, str]:
    for name, data in files.items():
        write(root / "model" / name, data)
    spec = SizeManifest("model", tuple(listed.items()))
    return check(root, spec)


def test_a_complete_directory_passes(tmp_path: Path) -> None:
    ok, detail = sizes(tmp_path, {"a": b"123", "b": b"12345"}, {"a": 3, "b": 5})
    assert (ok, detail) == (True, "")


def test_a_missing_file_is_reported_by_name(tmp_path: Path) -> None:
    ok, detail = sizes(tmp_path, {"a": b"123"}, {"a": 3, "b": 5})
    assert (ok, detail) == (False, "b is missing")


def test_a_truncated_file_reports_both_sizes(tmp_path: Path) -> None:
    ok, detail = sizes(tmp_path, {"a": b"12"}, {"a": 3})
    assert (ok, detail) == (False, "a is 2 bytes, expected 3")


def test_a_missing_directory_is_reported(tmp_path: Path) -> None:
    ok, detail = check(tmp_path, SizeManifest("model", (("a", 1),)))
    assert (ok, detail) == (False, "directory does not exist")


def test_many_problems_are_summarised(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    listed = dict.fromkeys("abcde", 1)
    ok, detail = check(tmp_path, SizeManifest("model", tuple(listed.items())))
    assert not ok
    assert detail == "a is missing; b is missing; c is missing (and 2 more)"


def test_an_empty_file_with_a_zero_size_passes(tmp_path: Path) -> None:
    ok, _ = sizes(tmp_path, {"empty": b""}, {"empty": 0})
    assert ok


# --- Sha256Manifest / FileSha256 --------------------------------------------------------------


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_a_sha256_manifest_that_matches_passes(tmp_path: Path) -> None:
    write(tmp_path / "kit" / "one", b"1")
    write(tmp_path / "kit" / "MANIFEST", f"{sha(b'1')}  one\n".encode())
    assert check(tmp_path, Sha256Manifest("kit", "MANIFEST")) == (True, "")


def test_a_sha256_manifest_catches_a_changed_file(tmp_path: Path) -> None:
    write(tmp_path / "kit" / "one", b"changed")
    write(tmp_path / "kit" / "MANIFEST", f"{sha(b'1')}  one\n".encode())
    ok, detail = check(tmp_path, Sha256Manifest("kit", "MANIFEST"))
    assert not ok and "one" in detail


def test_a_sha256_manifest_catches_a_missing_file(tmp_path: Path) -> None:
    write(tmp_path / "kit" / "MANIFEST", f"{sha(b'1')}  one\n".encode())
    ok, detail = check(tmp_path, Sha256Manifest("kit", "MANIFEST"))
    assert not ok and "one" in detail


def test_a_missing_manifest_directory_fails(tmp_path: Path) -> None:
    ok, detail = check(tmp_path, Sha256Manifest("kit", "MANIFEST"))
    assert (ok, detail) == (False, "directory does not exist")


def test_a_file_with_the_pinned_hash_passes(tmp_path: Path) -> None:
    write(tmp_path / "cache" / "lookup.json", b"[0,1]")
    assert check(tmp_path, FileSha256("lookup_table", sha(b"[0,1]"))) == (True, "")


def test_a_file_with_another_hash_fails_naming_both(tmp_path: Path) -> None:
    write(tmp_path / "cache" / "lookup.json", b"[0,2]")
    ok, detail = check(tmp_path, FileSha256("lookup_table", sha(b"[0,1]")))
    assert not ok
    assert sha(b"[0,2]") in detail and sha(b"[0,1]") in detail


def test_an_absent_file_fails(tmp_path: Path) -> None:
    assert check(tmp_path, FileSha256("lookup_table", sha(b"x"))) == (False, "file does not exist")


# --- RankCacheFooter --------------------------------------------------------------------------

CACHE = RankCacheFooter("rank_cache_dir", "rank", ".bin", MAGIC)


def cache_file(root: Path, name: str, data: bytes) -> Path:
    return write(root / "cache" / "rank-cache" / name, data)


def test_a_complete_rank_cache_passes(tmp_path: Path) -> None:
    cache_file(tmp_path, "rank0of2-abc.bin", b"weights" + MAGIC.encode())
    assert check(tmp_path, CACHE) == (True, "")


def test_each_rank_looks_for_its_own_file(tmp_path: Path) -> None:
    cache_file(tmp_path, "rank0of2-abc.bin", b"w" + MAGIC.encode())
    ok, detail = check(tmp_path, CACHE, rank=1)
    assert (ok, detail) == (False, "found 0 matching file(s), need exactly one")


def test_the_world_size_is_part_of_the_name(tmp_path: Path) -> None:
    cache_file(tmp_path, "rank0of4-abc.bin", b"w" + MAGIC.encode())
    assert check(tmp_path, CACHE, rank=0, world=2)[0] is False
    assert check(tmp_path, CACHE, rank=0, world=4)[0] is True


def test_a_wrong_footer_means_the_file_is_incomplete(tmp_path: Path) -> None:
    cache_file(tmp_path, "rank0of2-abc.bin", b"weights-cut-off")
    ok, detail = check(tmp_path, CACHE)
    assert not ok and detail.endswith("does not end with TFDSRK01")


def test_two_matching_files_are_ambiguous(tmp_path: Path) -> None:
    cache_file(tmp_path, "rank0of2-abc.bin", b"w" + MAGIC.encode())
    cache_file(tmp_path, "rank0of2-def.bin", b"w" + MAGIC.encode())
    assert check(tmp_path, CACHE) == (False, "found 2 matching file(s), need exactly one")


def test_a_leftover_partial_file_fails(tmp_path: Path) -> None:
    cache_file(tmp_path, "rank0of2-abc.bin", b"w" + MAGIC.encode())
    cache_file(tmp_path, "rank0of2-abc.partial", b"half")
    ok, detail = check(tmp_path, CACHE)
    assert not ok and "unfinished file present" in detail


def test_a_missing_cache_directory_fails(tmp_path: Path) -> None:
    assert check(tmp_path, CACHE)[0] is False


def test_a_cache_path_with_spaces_is_handled(tmp_path: Path) -> None:
    root = tmp_path / "with space"
    cache_file(root, "rank0of2-abc.bin", b"w" + MAGIC.encode())
    assert check(root, CACHE) == (True, "")


# --- Executables ------------------------------------------------------------------------------

HEAD = Program("binaries", "native/bin/head")
WORKER = Program("binaries", "bin/worker")
PROGRAMS = Executables(rank0=(HEAD,), others=(WORKER,))


def install(root: Path, rel: str, mode: int = 0o755) -> None:
    write(root / "bin" / rel, b"#!/bin/sh\n").chmod(mode)


def test_rank_zero_needs_only_the_head_program(tmp_path: Path) -> None:
    install(tmp_path, "native/bin/head")
    assert check(tmp_path, PROGRAMS, rank=0) == (True, "")


def test_other_ranks_need_only_the_worker_program(tmp_path: Path) -> None:
    install(tmp_path, "bin/worker")
    assert check(tmp_path, PROGRAMS, rank=1) == (True, "")
    assert check(tmp_path, PROGRAMS, rank=3, world=4) == (True, "")


def test_a_node_without_its_own_program_fails_naming_it(tmp_path: Path) -> None:
    install(tmp_path, "bin/worker")
    ok, detail = check(tmp_path, PROGRAMS, rank=0)
    assert not ok and detail.startswith("not executable:") and detail.endswith("native/bin/head")


def test_a_program_without_the_executable_bit_fails(tmp_path: Path) -> None:
    install(tmp_path, "bin/worker", mode=0o644)
    ok, detail = check(tmp_path, PROGRAMS, rank=1)
    assert not ok and detail.endswith("bin/worker")


def test_the_check_target_must_exist_in_the_profile() -> None:
    block, model = block_in(Path("/x"))
    bad = dataclasses.replace(CACHE, target="nope")
    with pytest.raises(KeyError, match="no host path named nope"):
        verify(local_run, "h", bad, block.profile, block, model, rank=0, world_size=2)
