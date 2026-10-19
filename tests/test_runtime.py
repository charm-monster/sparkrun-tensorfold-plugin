"""The runtime class, driven with real sparkrun recipes and the real executor config type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from sparkrun.core.recipe import Recipe, RecipeError
from sparkrun.core.validation import ERROR, WARNING
from sparkrun_tensorfold_profile_hello_engine import HELLO_ENGINE

from sparkrun_tensorfold_plugin.containers import docker_options, rank_options
from sparkrun_tensorfold_plugin.recipe_block import KEY, TensorfoldBlock
from sparkrun_tensorfold_plugin.runtime import DISPLAY_MASTER_ADDR, TensorfoldPluginRuntime
from tests.test_commands import MASTER, head_command, worker_command


def recipe_of(data: dict[str, Any]) -> Recipe:
    return Recipe.from_dict(data)


def test_the_runtime_is_registered_under_its_own_name() -> None:
    runtime = TensorfoldPluginRuntime()
    assert runtime.runtime_name == "tensorfold-plugin"
    assert runtime.cluster_strategy() == "native"


def test_workers_never_wait_on_the_rendezvous_port() -> None:
    assert TensorfoldPluginRuntime().native_rendezvous_port() is None


def test_the_example_recipe_has_no_findings(example_data: dict[str, Any]) -> None:
    assert TensorfoldPluginRuntime().validate_recipe(recipe_of(example_data)) == []


def test_relational_problems_are_launch_blocking_errors(example_data: dict[str, Any]) -> None:
    example_data["readiness"] = {"inference": True}
    issues = TensorfoldPluginRuntime().validate_recipe(recipe_of(example_data))
    assert [i.severity for i in issues] == [ERROR]
    assert "readiness.inference must be false" in issues[0].message


def test_a_recipe_container_conflicts_with_the_profile_image(example_data: dict[str, Any]) -> None:
    example_data["container"] = "example/other:latest"
    (issue,) = TensorfoldPluginRuntime().validate_recipe(recipe_of(example_data))
    assert issue.severity == ERROR
    assert HELLO_ENGINE.image in issue.message


def test_an_unused_serve_value_is_a_warning(example_data: dict[str, Any]) -> None:
    example_data["defaults"]["bogus"] = 1
    (issue,) = TensorfoldPluginRuntime().validate_recipe(recipe_of(example_data))
    assert issue.severity == WARNING
    assert "defaults.bogus is not used by profile hello-engine" in issue.message


def test_a_recipe_without_the_block_is_an_error(example_data: dict[str, Any]) -> None:
    del example_data[KEY]
    (issue,) = TensorfoldPluginRuntime().validate_recipe(recipe_of(example_data))
    assert issue.severity == ERROR
    assert "has no 'tensorfold' block" in issue.message


def test_the_image_comes_from_the_profile_under_both_sparkrun_call_styles(
    example_data: dict[str, Any],
) -> None:
    runtime, recipe = TensorfoldPluginRuntime(), recipe_of(example_data)
    assert runtime.resolve_container(recipe) == HELLO_ENGINE.image
    assert runtime.resolve_container(recipe, {}) == HELLO_ENGINE.image
    assert runtime.resolve_container(recipe, host_hardware=None) == HELLO_ENGINE.image


def test_the_container_flags_are_exactly_what_the_profile_needs(
    example_data: dict[str, Any],
) -> None:
    recipe = recipe_of(example_data)
    block = recipe.plugin_item(KEY)
    assert isinstance(block, TensorfoldBlock)
    assert docker_options(block.profile, block) == [
        "--ipc",
        "host",
        "--user",
        "root",
        "--cap-add",
        "IPC_LOCK",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--device",
        "/dev/infiniband",
        "-v",
        "/srv/tensorfold/bin:/srv/tensorfold/bin:ro",
        "-v",
        "/srv/tensorfold/models/hello-assets:/srv/tensorfold/models/hello-assets:ro",
        "-v",
        "/srv/tensorfold/models/kit:/srv/tensorfold/models/kit:ro",
        "-v",
        "/srv/tensorfold/cache:/srv/tensorfold/cache:rw",
    ]


@pytest.mark.parametrize("world", [2, 4])
def test_the_key_file_is_mounted_on_the_rank_that_reads_it_and_no_other(
    example_data: dict[str, Any], world: int
) -> None:
    block = recipe_of(example_data).plugin_item(KEY)
    assert isinstance(block, TensorfoldBlock)
    assert rank_options(block.profile, block, world) == {
        0: ["-v", "/srv/tensorfold/api.key:/srv/tensorfold/api.key:ro"]
    }


def test_prepare_rejects_a_recipe_that_cannot_launch(example_data: dict[str, Any]) -> None:
    example_data["readiness"] = {}
    with pytest.raises(RecipeError, match="readiness.inference must be false"):
        TensorfoldPluginRuntime().prepare(recipe_of(example_data), ["a", "b"])


def test_the_displayed_serve_command_marks_the_unresolved_address(
    example_data: dict[str, Any],
) -> None:
    command = TensorfoldPluginRuntime().generate_command(
        recipe_of(example_data), {}, is_cluster=True, num_nodes=2
    )
    assert command == head_command(2).replace(MASTER, f"'{DISPLAY_MASTER_ADDR}'")


def test_one_node_is_refused_when_the_profile_does_not_run_on_one(
    example_data: dict[str, Any],
) -> None:
    with pytest.raises(RecipeError, match=r"runs on \[2, 4\] nodes, not 1"):
        TensorfoldPluginRuntime().generate_command(recipe_of(example_data), {}, is_cluster=False)


def test_node_commands_use_the_resolved_fabric_address(example_data: dict[str, Any]) -> None:
    runtime, recipe = TensorfoldPluginRuntime(), recipe_of(example_data)
    with runtime.master_address(MASTER):
        head = runtime.generate_node_command(recipe, {}, "192.0.2.9", 2, 0, 25000)
        worker = runtime.generate_node_command(recipe, {}, "192.0.2.9", 2, 1, 25000)
    assert head == head_command(2)
    assert worker == worker_command(1, 2)


def test_command_line_overrides_win_over_recipe_defaults(example_data: dict[str, Any]) -> None:
    runtime, recipe = TensorfoldPluginRuntime(), recipe_of(example_data)
    with runtime.master_address(MASTER):
        head = runtime.generate_node_command(recipe, {"port": 9000}, "x", 2, 0)
    assert head == head_command(2).replace("--port 8000", "--port 9000")


def test_a_node_command_needs_the_address_to_have_been_resolved(
    example_data: dict[str, Any],
) -> None:
    with pytest.raises(RuntimeError, match="fabric address has not been resolved"):
        TensorfoldPluginRuntime().generate_node_command(recipe_of(example_data), {}, "x", 2, 0)


def test_the_pinned_address_is_released_afterwards(example_data: dict[str, Any]) -> None:
    runtime = TensorfoldPluginRuntime()
    with runtime.master_address(MASTER):
        pass
    with pytest.raises(RuntimeError, match="has not been resolved"):
        runtime.generate_node_command(recipe_of(example_data), {}, "x", 2, 0)


def test_an_untrusted_recipe_cannot_launch(example_data: dict[str, Any]) -> None:
    with pytest.raises(RecipeError, match="pass --trust"):
        TensorfoldPluginRuntime()._run_cluster(
            ["a", "b"], "img", "", recipe_of(example_data), {}, dry_run=True, trust=False
        )


def test_a_world_size_the_profile_does_not_run_cannot_launch(example_data: dict[str, Any]) -> None:
    with pytest.raises(RecipeError, match=r"runs on \[2, 4\] nodes, got 3"):
        TensorfoldPluginRuntime()._run_cluster(
            ["a", "b", "c"], "img", "", recipe_of(example_data), {}, dry_run=True, trust=True
        )


def test_the_solo_launch_path_asks_the_profile_whether_one_node_can_run(
    example_data: dict[str, Any],
) -> None:
    with pytest.raises(RecipeError, match=r"runs on \[2, 4\] nodes, got 1"):
        TensorfoldPluginRuntime()._run_solo(
            "a", "image", recipe=recipe_of(example_data), trust=True
        )


@dataclass
class Result:
    returncode: int


def keyed_block(example_data: dict[str, Any]) -> TensorfoldBlock:
    block = recipe_of(example_data).plugin_item(KEY)
    assert isinstance(block, TensorfoldBlock)
    return block


def test_a_missing_key_file_names_the_host(example_data: dict[str, Any]) -> None:
    block = keyed_block(example_data)

    def run(host: str, command: str) -> Result:  # noqa: ARG001
        return Result(1)

    with pytest.raises(RecipeError, match="a: api_key_file /srv/tensorfold/api.key does not exist"):
        TensorfoldPluginRuntime._require_key_file(
            run, ["a", "b"], block.profile, block, dry_run=False
        )


@pytest.mark.parametrize("hosts", [["a", "b"], ["a", "b", "c", "d"]])
def test_only_the_node_of_a_rank_that_reads_the_key_is_asked_for_it(
    example_data: dict[str, Any], hosts: list[str]
) -> None:
    block = keyed_block(example_data)
    asked: list[tuple[str, str]] = []

    def run(host: str, command: str) -> Result:
        asked.append((host, command))
        return Result(0)

    TensorfoldPluginRuntime._require_key_file(run, hosts, block.profile, block, dry_run=False)
    assert asked == [("a", "test -f /srv/tensorfold/api.key")]


def test_a_dry_run_checks_no_host(example_data: dict[str, Any]) -> None:
    block = keyed_block(example_data)

    def run(host: str, command: str) -> Result:  # noqa: ARG001
        raise AssertionError("a dry run must not contact a host")

    TensorfoldPluginRuntime._require_key_file(run, ["a"], block.profile, block, dry_run=True)
