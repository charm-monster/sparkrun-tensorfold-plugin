"""Per-container docker options, read from the launch scripts a real sparkrun executor writes."""

from __future__ import annotations

from typing import Any

import pytest

from sparkrun_tensorfold_plugin.containers import rank_options
from sparkrun_tensorfold_plugin.executors import scoped_launch_options
from sparkrun_tensorfold_plugin.recipe_block import KEY, TensorfoldBlock
from sparkrun_tensorfold_plugin.runtime import TensorfoldPluginRuntime
from tests.test_runtime import recipe_of

SHARED = ["--ipc", "host"]
CLUSTER = "sparkrun_test"


def launch_script(executor: Any, rank: int) -> str:
    script: str = executor.generate_launch_script(
        image="example/image:1",
        container_name=executor.node_container_name(CLUSTER, rank),
        command="sleep infinity",
        env={},
        volumes=None,
        nccl_env=None,
        extra_docker_opts=SHARED,
        sparkrun_labels=None,
    )
    return script


def runtime_executor() -> Any:
    return TensorfoldPluginRuntime()._resolve_executor()


@pytest.mark.parametrize("world", [2, 3, 4, 8])
def test_only_the_named_container_gets_its_own_options(world: int) -> None:
    executor = runtime_executor()
    named = executor.node_container_name(CLUSTER, 0)
    scoped = scoped_launch_options(executor, {named: ["-v", "/secret/key:/secret/key:ro"]})
    assert "/secret/key:/secret/key:ro" in launch_script(scoped, 0)
    for rank in range(1, world):
        assert "/secret/key" not in launch_script(scoped, rank)


def test_the_shared_options_survive_and_the_untouched_executor_is_unchanged() -> None:
    executor = runtime_executor()
    before = launch_script(executor, 0)
    scoped = scoped_launch_options(
        executor, {executor.node_container_name(CLUSTER, 0): ["--label", "scoped=yes"]}
    )
    after = launch_script(scoped, 0)
    assert "--ipc host --label scoped=yes example/image:1" in after
    assert launch_script(executor, 0) == before


def test_routing_through_a_host_executor_scopes_the_same_way() -> None:
    executor = runtime_executor()
    if not hasattr(executor, "for_host"):
        pytest.skip("this sparkrun version has one executor for every host")
    scoped = scoped_launch_options(
        executor, {executor.node_container_name(CLUSTER, 1): ["--label", "scoped=yes"]}
    )
    assert "scoped=yes" in launch_script(scoped.for_host("any-host"), 1)
    assert "scoped=yes" not in launch_script(scoped.for_host("any-host"), 0)


@pytest.mark.parametrize("world", [2, 4])
def test_the_runtime_mounts_the_key_on_rank_zero_only(
    example_data: dict[str, Any], world: int
) -> None:
    recipe = recipe_of(example_data)
    block = recipe.plugin_item(KEY)
    assert isinstance(block, TensorfoldBlock)
    runtime = TensorfoldPluginRuntime()
    scoped = runtime._with_rank_options(CLUSTER, block.profile, block, world)
    mount = "/srv/tensorfold/api.key:/srv/tensorfold/api.key:ro"
    assert mount in launch_script(scoped, 0)
    for rank in range(1, world):
        assert "api.key" not in launch_script(scoped, rank)
    assert rank_options(block.profile, block, world).keys() == {0}
