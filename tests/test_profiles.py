"""The profile model and registry, exercised through the example profile."""

from __future__ import annotations

import dataclasses

import pytest
from sparkrun_tensorfold_profile_hello_engine import HELLO_ENGINE

from sparkrun_tensorfold_plugin.profiles import REGISTRY, get_profile, register_profile
from sparkrun_tensorfold_plugin.profiles.model import (
    CommandSpec,
    CoordinatedPlan,
    DerivedPath,
    Flag,
    FlagMode,
    Profile,
    ProfileError,
    Ref,
    ServeKey,
    Source,
    ValueType,
    const,
    network,
    path,
    serve,
)


def _flag(spec: CommandSpec, name: str) -> Flag:
    return next(f for f in spec.flags if f.name == name)


def _env(spec: CommandSpec, name: str) -> Ref:
    return next(e.value for e in spec.env if e.name == name)


def test_the_registered_example_profile_is_the_one_returned() -> None:
    assert get_profile("hello-engine") is HELLO_ENGINE
    assert "hello-engine" in REGISTRY


def test_unknown_profile_names_what_is_available() -> None:
    with pytest.raises(ProfileError, match=r"unknown profile 'nope'.*hello-engine"):
        get_profile("nope")


def test_unknown_profile_with_nothing_registered_says_where_profiles_come_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sparkrun_tensorfold_plugin.profiles.REGISTRY", {})
    with pytest.raises(ProfileError, match=r"no profile is registered.*sparkrun plugin package"):
        get_profile("hello-engine")


def test_registering_the_same_profile_again_is_allowed() -> None:
    register_profile(HELLO_ENGINE)
    register_profile(HELLO_ENGINE)
    assert get_profile("hello-engine") is HELLO_ENGINE


def test_a_different_profile_under_a_taken_name_is_refused() -> None:
    other = dataclasses.replace(HELLO_ENGINE, readiness_path="/other")
    with pytest.raises(ProfileError, match="already registered with different contents"):
        register_profile(other)
    assert get_profile("hello-engine") is HELLO_ENGINE


def test_an_invalid_profile_is_refused_and_not_registered() -> None:
    broken_head = dataclasses.replace(
        HELLO_ENGINE.head, flags=(*HELLO_ENGINE.head.flags, Flag("--x", serve("not_declared")))
    )
    broken = dataclasses.replace(HELLO_ENGINE, name="broken-example", head=broken_head)
    with pytest.raises(ProfileError, match="unknown serve key not_declared"):
        register_profile(broken)
    assert "broken-example" not in REGISTRY


# A value the ranks must agree on is declared once: both commands point at the same Ref, so the
# two cannot drift apart. The example profile declares each such pair this way.
def test_cross_rank_values_are_declared_once() -> None:
    head, worker = HELLO_ENGINE.head, HELLO_ENGINE.worker
    assert _flag(head, "--context").value == _flag(worker, "--pool").value
    assert _env(head, "HELLO_RDMA_DEVICES") == _flag(worker, "--rdma").value
    assert _env(head, "HELLO_ASSETS") == _flag(worker, "--assets").value
    assert _env(head, "HELLO_LOOKUP") == _flag(worker, "--lookup").value
    assert _env(head, "HELLO_STREAMS") == _flag(worker, "--streams").value
    assert _env(head, "HELLO_KIT") == worker.positionals[-1]
    assert _env(head, "HELLO_CACHE") == worker.positionals[1]
    for name in ("HELLO_RDMA_DEVICES", "HELLO_GID_INDEX", "HELLO_SOCKET_IFNAME", "HELLO_LOG_LEVEL"):
        assert _env(head, name) == _env(worker, name)


def test_worker_positionals_follow_the_documented_usage_line() -> None:
    # hello-follow MODEL CACHE RANK WORLD MASTER PORT KIT
    worker = HELLO_ENGINE.worker
    assert [(r.source, r.name) for r in worker.positionals] == [
        (Source.MODEL, "model"),
        (Source.DERIVED, "rank_cache_dir"),
        (Source.TOPOLOGY, "rank"),
        (Source.TOPOLOGY, "world_size"),
        (Source.TOPOLOGY, "master_addr"),
        (Source.TOPOLOGY, "master_port"),
        (Source.PATH, "kit"),
    ]


def test_network_fields_come_from_the_commands() -> None:
    assert HELLO_ENGINE.referenced(Source.NETWORK) == {
        "rdma_devices",
        "socket_interface",
        "gid_index",
    }


def _replace(profile: Profile, **changes: object) -> Profile:
    return dataclasses.replace(profile, **changes)  # type: ignore[arg-type]


def test_a_reference_to_an_undeclared_serve_key_is_rejected() -> None:
    broken_head = dataclasses.replace(
        HELLO_ENGINE.head, flags=(*HELLO_ENGINE.head.flags, Flag("--x", serve("not_declared")))
    )
    with pytest.raises(ProfileError, match="unknown serve key not_declared"):
        _replace(HELLO_ENGINE, head=broken_head).validate()


def test_a_reference_to_an_undeclared_path_key_is_rejected() -> None:
    broken_worker = dataclasses.replace(
        HELLO_ENGINE.worker, positionals=(*HELLO_ENGINE.worker.positionals, path("not_declared"))
    )
    with pytest.raises(ProfileError, match="unknown path key not_declared"):
        _replace(HELLO_ENGINE, worker=broken_worker).validate()


def test_a_derived_path_on_an_undeclared_base_is_rejected() -> None:
    bad = (*HELLO_ENGINE.derived_paths, DerivedPath("x", base="not_declared", relative="y"))
    with pytest.raises(ProfileError, match="unknown path key not_declared"):
        _replace(HELLO_ENGINE, derived_paths=bad).validate()


def test_an_unknown_network_field_is_rejected() -> None:
    broken_head = dataclasses.replace(
        HELLO_ENGINE.head, flags=(*HELLO_ENGINE.head.flags, Flag("--x", network("not_a_field")))
    )
    with pytest.raises(ProfileError, match="unknown network field not_a_field"):
        _replace(HELLO_ENGINE, head=broken_head).validate()


def test_a_world_size_below_one_is_rejected() -> None:
    with pytest.raises(ProfileError, match="world sizes of at least 1"):
        _replace(HELLO_ENGINE, world_sizes=frozenset({0, 2})).validate()


@pytest.mark.parametrize("sizes", [{1}, {1, 2}, {3}, {2, 3, 5, 8, 16}])
def test_a_profile_may_run_on_any_set_of_node_counts(sizes: set[int]) -> None:
    _replace(HELLO_ENGINE, world_sizes=frozenset(sizes)).validate()


def test_a_duplicate_serve_key_is_rejected() -> None:
    twice = (*HELLO_ENGINE.serve_keys, ServeKey("port", ValueType.INT, required=True))
    with pytest.raises(ProfileError, match="serve key twice"):
        _replace(HELLO_ENGINE, serve_keys=twice).validate()


def test_a_constraint_on_an_undeclared_key_is_rejected() -> None:
    bad = dataclasses.replace(HELLO_ENGINE.constraints[0], right="not_declared")
    with pytest.raises(ProfileError, match="constraint uses unknown serve key not_declared"):
        _replace(HELLO_ENGINE, constraints=(bad,)).validate()


def test_a_const_ref_requires_a_value() -> None:
    with pytest.raises(ProfileError, match="CONST ref needs a value"):
        Ref(Source.CONST, "x")


def test_only_const_refs_carry_a_value() -> None:
    with pytest.raises(ProfileError, match="only a CONST ref carries a value"):
        Ref(Source.SERVE, "port", 1)


def test_a_value_flag_needs_a_value() -> None:
    with pytest.raises(ProfileError, match="mode VALUE but no value"):
        Flag("--x")


def test_a_presence_flag_needs_a_boolean_source() -> None:
    with pytest.raises(ProfileError, match="needs a boolean value"):
        Flag("--x", None, FlagMode.WHEN_TRUE)


def test_const_helper_keeps_the_literal() -> None:
    ref = const(1)
    assert ref.source is Source.CONST and ref.const == 1


def _without_interface(spec: CommandSpec) -> CommandSpec:
    return dataclasses.replace(
        spec, env=tuple(e for e in spec.env if e.name != "HELLO_SOCKET_IFNAME")
    )


def test_a_master_address_without_a_fabric_interface_is_rejected() -> None:
    # master_addr is read from the rank 0 fabric interface; commands that use it must also name it
    artifacts = tuple(
        dataclasses.replace(
            a, plan=dataclasses.replace(a.plan, command=_without_interface(a.plan.command))
        )
        if isinstance(a.plan, CoordinatedPlan)
        else a
        for a in HELLO_ENGINE.artifacts
    )
    broken = _replace(
        HELLO_ENGINE,
        head=_without_interface(HELLO_ENGINE.head),
        worker=_without_interface(HELLO_ENGINE.worker),
        artifacts=artifacts,
    )
    with pytest.raises(ProfileError, match="must also reference network.socket_interface"):
        broken.validate()
