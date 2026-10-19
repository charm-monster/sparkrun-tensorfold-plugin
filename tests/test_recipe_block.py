"""The ``tensorfold`` recipe block, loaded through sparkrun's real Recipe machinery."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
from sparkrun.core.recipe import Recipe, RecipeError

from sparkrun_tensorfold_plugin.profiles import get_profile
from sparkrun_tensorfold_plugin.recipe_block import (
    KEY,
    PrepareMode,
    TensorfoldBlock,
    parse_block,
    relational_errors,
)
from tests.conftest import EXAMPLE

Mutation = Callable[[dict[str, Any]], None]


def _block(recipe: Recipe) -> TensorfoldBlock:
    block = recipe.plugin_item(KEY)
    assert isinstance(block, TensorfoldBlock)
    return block


def test_the_example_recipe_loads_through_sparkrun() -> None:
    recipe = Recipe.load(EXAMPLE)
    block = _block(recipe)
    assert block.profile is get_profile("hello-engine")
    assert block.prepare is PrepareMode.BUILD
    assert block.paths == {
        "binaries": "/srv/tensorfold/bin",
        "cache_dir": "/srv/tensorfold/cache",
        "assets": "/srv/tensorfold/models/hello-assets",
        "kit": "/srv/tensorfold/models/kit",
    }
    assert block.network == {
        "gid_index": 3,
        "rdma_devices": ("rocep1s0f1", "roceP2p1s0f1"),
        "socket_interface": "enp1s0f1np1",
    }
    assert block.rendezvous_port == 29620
    assert block.api_key_file == "/srv/tensorfold/api.key"


def test_export_is_canonical_and_round_trips(example_data: dict[str, Any]) -> None:
    recipe = Recipe.from_dict(example_data)
    exported = recipe.export_plugin_items()
    assert exported == {KEY: example_data[KEY]}
    assert parse_block(exported[KEY]) == _block(recipe)


def test_the_export_holds_no_key_material(example_data: dict[str, Any]) -> None:
    exported = Recipe.from_dict(example_data).export_plugin_items()[KEY]
    assert set(exported) == {
        "profile",
        "prepare",
        "paths",
        "network",
        "rendezvous_port",
        "api_key_file",
    }
    assert exported["api_key_file"] == "/srv/tensorfold/api.key"  # a path, not the key


def _set(path: tuple[str, ...], value: Any) -> Mutation:
    def apply(data: dict[str, Any]) -> None:
        node = data[KEY]
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value

    return apply


def _drop(path: tuple[str, ...]) -> Mutation:
    def apply(data: dict[str, Any]) -> None:
        node = data[KEY]
        for part in path[:-1]:
            node = node[part]
        del node[path[-1]]

    return apply


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_drop(("profile",)), "missing required key(s) profile"),
        (_set(("profile",), "nope"), "unknown profile 'nope'"),
        (_set(("profile",), 7), "unknown profile ''"),
        (_drop(("prepare",)), "missing required key(s) prepare"),
        (_set(("prepare",), "maybe"), "tensorfold.prepare must be one of: build, verify"),
        (_drop(("paths", "kit")), "missing required key(s) kit"),
        (_set(("paths", "kit"), "relative/kit"), "tensorfold.paths.kit must be an absolute path"),
        (_set(("paths", "kit"), "/"), "must not be the filesystem root"),
        (_set(("paths", "extra"), "/x"), "tensorfold.paths has unknown key(s) extra"),
        (_set(("paths",), ["/a"]), "tensorfold.paths must be a mapping"),
        (_drop(("network", "gid_index")), "missing required key(s) gid_index"),
        (_set(("network", "gid_index"), True), "tensorfold.network.gid_index must be an integer"),
        (_set(("network", "gid_index"), -1), "tensorfold.network.gid_index must be >= 0"),
        (
            _set(("network", "rdma_devices"), []),
            "tensorfold.network.rdma_devices must be a non-empty list",
        ),
        (_set(("network", "rdma_devices"), ["bad name"]), "may only contain letters, digits"),
        (_set(("network", "socket_interface"), ""), "must be a non-empty string"),
        (_set(("network", "surprise"), 1), "tensorfold.network has unknown key(s) surprise"),
        (_drop(("network",)), "missing required key(s) network"),
        (_set(("rendezvous_port",), 0), "between 1 and 65535"),
        (_set(("rendezvous_port",), 70000), "between 1 and 65535"),
        (_set(("rendezvous_port",), "29620"), "tensorfold.rendezvous_port must be an integer"),
        (_drop(("rendezvous_port",)), "missing required key(s) rendezvous_port"),
        (_drop(("api_key_file",)), "missing required key(s) api_key_file"),
        (_set(("api_key_file",), "key"), "tensorfold.api_key_file must be an absolute path"),
        (_set(("surprise",), 1), "tensorfold has unknown key(s) surprise"),
    ],
)
def test_an_invalid_block_fails_recipe_load_naming_the_key(
    example_data: dict[str, Any], mutation: Mutation, message: str
) -> None:
    mutation(example_data)
    with pytest.raises(
        RecipeError, match=r"tensorfold.*" + message.replace("(", r"\(").replace(")", r"\)")
    ):
        Recipe.from_dict(example_data)


def test_a_block_that_is_not_a_mapping_fails(example_data: dict[str, Any]) -> None:
    example_data[KEY] = "hello-engine"
    with pytest.raises(RecipeError, match="tensorfold must be a mapping, got str"):
        Recipe.from_dict(example_data)


def _good(data: dict[str, Any]) -> dict[str, Any]:
    recipe = Recipe.from_dict(copy.deepcopy(data))
    return {
        "block": _block(recipe),
        "model": recipe.model,
        "defaults": dict(recipe.defaults),
        "min_nodes": recipe.min_nodes,
        "max_nodes": recipe.max_nodes,
        "readiness_inference": recipe.readiness.get("inference"),
    }


def test_the_example_has_no_relational_errors(example_data: dict[str, Any]) -> None:
    assert relational_errors(**_good(example_data)) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model": "models/relative"}, "model must be an absolute directory"),
        ({"max_nodes": None}, "max_nodes is required"),
        ({"min_nodes": 1}, "runs on [2, 4] nodes; recipe allows 1..2"),
        ({"max_nodes": 8}, "runs on [2, 4] nodes; recipe allows 2..8"),
        ({"readiness_inference": True}, "readiness.inference must be false"),
        ({"readiness_inference": None}, "readiness.inference must be false"),
    ],
)
def test_recipe_level_problems_are_reported(
    example_data: dict[str, Any], change: dict[str, Any], message: str
) -> None:
    args = {**_good(example_data), **change}
    assert any(message in e for e in relational_errors(**args))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("port", "8000", "defaults.port must be of type int"),
        ("port", True, "defaults.port must be of type int"),
        ("host", 5, "defaults.host must be of type str"),
        ("thinking", 1, "defaults.thinking must be of type bool"),
        ("temperature", "hot", "defaults.temperature must be of type float"),
    ],
)
def test_serve_values_must_have_the_profile_type(
    example_data: dict[str, Any], key: str, value: Any, message: str
) -> None:
    args = _good(example_data)
    args["defaults"][key] = value
    assert message in relational_errors(**args)


@pytest.mark.parametrize("key", ["host", "port", "served_model_name", "max_model_len", "parallel"])
def test_required_serve_values_must_be_present(example_data: dict[str, Any], key: str) -> None:
    args = _good(example_data)
    del args["defaults"][key]
    assert f"defaults.{key} is required by profile hello-engine" in relational_errors(**args)


def test_optional_serve_values_may_be_absent_and_ints_are_floats(
    example_data: dict[str, Any],
) -> None:
    args = _good(example_data)
    for key in ("thinking", "max_tokens", "top_p", "top_k"):
        del args["defaults"][key]
    args["defaults"]["temperature"] = 1
    assert relational_errors(**args) == []


def test_parallel_is_capped_by_the_engine_default_when_streams_is_unset(
    example_data: dict[str, Any],
) -> None:
    args = _good(example_data)
    args["defaults"]["parallel"] = 5
    errors = relational_errors(**args)
    assert errors == [
        "defaults.parallel (5) may not exceed streams (4); set defaults.streams to raise the limit"
    ]


def test_parallel_is_capped_by_an_explicit_streams(example_data: dict[str, Any]) -> None:
    args = _good(example_data)
    args["defaults"].update(parallel=5, streams=6)
    assert relational_errors(**args) == []
    args["defaults"]["streams"] = 3
    assert any("may not exceed streams (3)" in e for e in relational_errors(**args))
