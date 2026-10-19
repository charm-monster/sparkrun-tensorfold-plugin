"""Rendered commands, checked against strings derived by hand from the example profile.

The expected strings are written out in full on purpose: a change to the rendering has to be a
visible change to these strings. They follow from ``examples/profile-plugin/.../profile.py`` and
``examples/hello-engine.yaml``: the environment in the profile's order, the program under its path
key, then positionals, then flags.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from sparkrun.core.recipe import Recipe

from sparkrun_tensorfold_plugin.commands import RenderError, Topology, render_command
from sparkrun_tensorfold_plugin.recipe_block import KEY, TensorfoldBlock

MASTER = "192.0.2.10"
MODEL = "/srv/tensorfold/models/hello-model"

FABRIC_ENV = (
    "HELLO_RDMA_DEVICES=rocep1s0f1,roceP2p1s0f1 HELLO_GID_INDEX=3 "
    "HELLO_SOCKET_IFNAME=enp1s0f1np1 HELLO_LOG_LEVEL=warn"
)


def head_command(world: int, streams: str = "") -> str:
    return (
        f"env {FABRIC_ENV} HELLO_WORLD={world} HELLO_MASTER={MASTER} HELLO_MASTER_PORT=29620 "
        "HELLO_CACHE=/srv/tensorfold/cache/rank-cache HELLO_KIT=/srv/tensorfold/models/kit "
        "HELLO_KERNELS=/srv/tensorfold/models/kit/kernels "
        "HELLO_ASSETS=/srv/tensorfold/models/hello-assets "
        "HELLO_LOOKUP=/srv/tensorfold/cache/lookup.json HELLO_GRAPHS=1"
        f"{streams} "
        f"/srv/tensorfold/bin/bin/hello-serve serve {MODEL} "
        "--host 0.0.0.0 --port 8000 --name hello-model "
        "--api-key-file /srv/tensorfold/api.key --context 8192 --parallel 4 --thinking "
        "--max-tokens 1024 --temperature 0.7 --top-p 0.9 --top-k 40 --backend cpu"
    )


def worker_command(rank: int, world: int, streams: str = "") -> str:
    return (
        f"env {FABRIC_ENV} /srv/tensorfold/bin/bin/hello-follow {MODEL} "
        f"/srv/tensorfold/cache/rank-cache {rank} {world} {MASTER} 29620 "
        "/srv/tensorfold/models/kit "
        "--assets /srv/tensorfold/models/hello-assets "
        "--lookup /srv/tensorfold/cache/lookup.json --pool 8192"
        f"{streams} --rdma rocep1s0f1,roceP2p1s0f1 --graphs 1"
    )


def render(data: dict[str, Any], rank: int, world: int = 2) -> str:
    recipe = Recipe.from_dict(data)
    block = recipe.plugin_item(KEY)
    assert isinstance(block, TensorfoldBlock)
    return render_command(
        block.profile,
        block,
        model=recipe.model,
        defaults=recipe.defaults,
        topology=Topology(rank, world, MASTER, block.rendezvous_port),
    )


def test_rank_zero_runs_the_serving_program(example_data: dict[str, Any]) -> None:
    assert render(example_data, rank=0) == head_command(world=2)


def test_rank_one_runs_the_follower_program(example_data: dict[str, Any]) -> None:
    assert render(example_data, rank=1) == worker_command(rank=1, world=2)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_four_nodes_number_every_follower(example_data: dict[str, Any], rank: int) -> None:
    assert render(example_data, rank=rank, world=4) == worker_command(rank=rank, world=4)


def test_four_nodes_tell_the_head_the_world_size(example_data: dict[str, Any]) -> None:
    assert render(example_data, rank=0, world=4) == head_command(world=4)


def test_streams_reach_both_ranks_from_one_value(example_data: dict[str, Any]) -> None:
    example_data["defaults"].update(streams=8, parallel=8)
    assert render(example_data, rank=0) == head_command(2, streams=" HELLO_STREAMS=8").replace(
        "--parallel 4", "--parallel 8"
    )
    assert render(example_data, rank=1) == worker_command(1, 2, streams=" --streams 8")


def test_thinking_false_selects_the_negated_flag(example_data: dict[str, Any]) -> None:
    example_data["defaults"]["thinking"] = False
    command = render(example_data, rank=0)
    assert " --no-thinking " in command
    assert " --thinking " not in command


def test_unset_optional_values_are_omitted_not_defaulted(example_data: dict[str, Any]) -> None:
    for key in ("thinking", "max_tokens", "temperature", "top_p", "top_k"):
        del example_data["defaults"][key]
    command = render(example_data, rank=0)
    assert command.endswith("--parallel 4 --backend cpu")
    for flag in ("--thinking", "--no-thinking", "--max-tokens", "--temperature", "--top-"):
        assert flag not in command


def test_values_are_shell_quoted(example_data: dict[str, Any]) -> None:
    example_data["model"] = "/srv/models/with space"
    command = render(example_data, rank=0)
    assert "serve '/srv/models/with space' " in command
    assert render(example_data, rank=1).count("'/srv/models/with space'") == 1


def test_a_world_size_the_profile_does_not_run_is_refused(example_data: dict[str, Any]) -> None:
    with pytest.raises(RenderError, match=r"runs on \[2, 4\] nodes, not 3"):
        render(example_data, rank=0, world=3)


def test_a_rank_outside_the_world_is_refused(example_data: dict[str, Any]) -> None:
    with pytest.raises(RenderError, match="rank 2 is outside a world of 2"):
        render(example_data, rank=2, world=2)


def test_a_missing_required_serve_value_is_an_error_not_a_default(
    example_data: dict[str, Any],
) -> None:
    del example_data["defaults"]["host"]
    with pytest.raises(RenderError, match="flag --host has no value"):
        render(example_data, rank=0)


def test_a_non_boolean_presence_value_is_refused(example_data: dict[str, Any]) -> None:
    example_data["defaults"]["thinking"] = "yes"
    with pytest.raises(RenderError, match="flag --thinking needs a boolean"):
        render(example_data, rank=0)


def test_rendering_uses_the_profile_it_is_given(example_data: dict[str, Any]) -> None:
    recipe = Recipe.from_dict(example_data)
    block = recipe.plugin_item(KEY)
    assert isinstance(block, TensorfoldBlock)
    narrowed = dataclasses.replace(block.profile, world_sizes=frozenset({2}))
    with pytest.raises(RenderError, match=r"runs on \[2\] nodes, not 4"):
        render_command(
            narrowed,
            block,
            model=recipe.model,
            defaults=recipe.defaults,
            topology=Topology(0, 4, MASTER, 29620),
        )
