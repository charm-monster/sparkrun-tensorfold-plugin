"""Nothing in the core assumes a node count: ranks, key-file placement and the master address all
follow from what the profile's commands say, for any world size the profile allows.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest
from sparkrun_tensorfold_profile_hello_engine import HELLO_ENGINE

from sparkrun_tensorfold_plugin.commands import Topology, render_command
from sparkrun_tensorfold_plugin.containers import key_file_ranks, rank_options
from sparkrun_tensorfold_plugin.profiles import REGISTRY
from sparkrun_tensorfold_plugin.profiles.model import (
    CommandSpec,
    Env,
    Flag,
    Profile,
    Program,
    Source,
    block,
    topology,
)
from sparkrun_tensorfold_plugin.recipe_block import TensorfoldBlock
from tests.fake_profile import fake_profile
from tests.helpers import block_in

SIZES = (1, 2, 3, 4, 5, 8, 16)
RANK_ENV = (Env("RANK", topology("rank")), Env("WORLD", topology("world_size")))
KEY_FLAG = Flag("--key-file", block("api_key_file"))


def spec(program: str, *, key: bool) -> CommandSpec:
    return CommandSpec(
        program=Program(base=None, relative=program),
        subcommand=None,
        positionals=(),
        flags=(KEY_FLAG,) if key else (),
        env=RANK_ENV,
    )


def profile_with(*, worker_reads_key: bool = False) -> Profile:
    return dataclasses.replace(
        fake_profile(),
        world_sizes=frozenset(SIZES),
        head=spec("serve", key=True),
        worker=spec("follow", key=worker_reads_key),
    )


@pytest.fixture
def keyed(tmp_path: Path) -> Iterator[tuple[Profile, TensorfoldBlock, str]]:
    profile = profile_with()
    REGISTRY["fake"] = profile
    try:
        parsed, model = block_in(tmp_path, "fake")
        yield profile, parsed, model
    finally:
        del REGISTRY["fake"]


@pytest.mark.parametrize("world", SIZES)
def test_every_rank_of_any_world_gets_its_own_command(
    keyed: tuple[Profile, TensorfoldBlock, str], world: int
) -> None:
    profile, parsed, model = keyed
    key = parsed.api_key_file
    for rank in range(world):
        command = render_command(
            profile, parsed, model=model, defaults={}, topology=Topology(rank, world, None, 29500)
        )
        if rank == 0:
            assert command == f"env RANK=0 WORLD={world} serve --key-file {key}"
        else:
            assert command == f"env RANK={rank} WORLD={world} follow"


@pytest.mark.parametrize("world", SIZES)
def test_the_key_file_belongs_to_the_ranks_whose_command_names_it(world: int) -> None:
    assert key_file_ranks(profile_with(), world) == (0,)
    assert key_file_ranks(profile_with(worker_reads_key=True), world) == tuple(range(world))


@pytest.mark.parametrize("world", SIZES)
def test_the_key_mount_follows_the_ranks_that_read_it(
    keyed: tuple[Profile, TensorfoldBlock, str], world: int
) -> None:
    _, parsed, _ = keyed
    key = parsed.api_key_file
    mount = ["-v", f"{key}:{key}:ro"]
    assert rank_options(profile_with(), parsed, world) == {0: mount}
    everywhere = rank_options(profile_with(worker_reads_key=True), parsed, world)
    assert everywhere == dict.fromkeys(range(world), mount)


def test_a_world_the_profile_does_not_list_is_refused(
    keyed: tuple[Profile, TensorfoldBlock, str],
) -> None:
    profile, parsed, model = keyed
    with pytest.raises(ValueError, match="runs on .* nodes, not 6"):
        render_command(
            profile, parsed, model=model, defaults={}, topology=Topology(0, 6, None, 29500)
        )


def test_the_master_address_is_resolved_only_for_profiles_that_use_it() -> None:
    assert HELLO_ENGINE.uses_master_address()
    assert not profile_with().uses_master_address()
    assert HELLO_ENGINE.ranks_referencing(Source.TOPOLOGY, "master_addr", 4) == (0, 1, 2, 3)
