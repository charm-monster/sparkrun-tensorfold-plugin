"""Render the command each rank runs from a profile, a recipe block and the launch topology.

Pure functions only: no I/O, no sparkrun imports. Every value is looked up through a profile
:class:`~sparkrun_tensorfold_plugin.profiles.model.Ref`, so a value the engine needs on both
the head and the workers is read from one place and cannot differ between them.
"""

from __future__ import annotations

import posixpath
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sparkrun_tensorfold_plugin.paths import host_path
from sparkrun_tensorfold_plugin.profiles.model import (
    CommandSpec,
    Flag,
    FlagMode,
    Profile,
    Program,
    Ref,
    Source,
)
from sparkrun_tensorfold_plugin.recipe_block import TensorfoldBlock

Value = str | int | float | bool | tuple[str, ...]


class RenderError(ValueError):
    """A command cannot be rendered because a required value is missing or malformed."""


@dataclass(frozen=True)
class Topology:
    """Values the launcher computes from the placement."""

    rank: int
    world_size: int
    master_addr: str | None  # None when no command or build script of the profile uses it
    master_port: int


def _resolve(
    ref: Ref,
    *,
    block: TensorfoldBlock,
    model: str,
    defaults: Mapping[str, Any],
    topology: Topology,
) -> Value | None:
    """The value a ref points at, or ``None`` if the recipe leaves it unset."""
    profile = block.profile
    match ref.source:
        case Source.CONST:
            if ref.const is None:
                raise RenderError(f"constant {ref.name} has no value")
            return ref.const
        case Source.MODEL:
            return host_path(profile, block, model, "model")
        case Source.SERVE:
            return defaults.get(ref.name)
        case Source.PATH | Source.DERIVED:
            return host_path(profile, block, model, ref.name)
        case Source.NETWORK:
            return block.network[ref.name]
        case Source.BLOCK:
            return {"api_key_file": block.api_key_file}[ref.name]
        case Source.TOPOLOGY:
            known: dict[str, Value | None] = {
                "rank": topology.rank,
                "world_size": topology.world_size,
                "master_addr": topology.master_addr,
                "master_port": topology.master_port,
            }
            value = known[ref.name]
            if value is None:
                raise RenderError(f"topology value {ref.name} was not resolved for this launch")
            return value


def _text(value: Value, where: str) -> str:
    if isinstance(value, bool):
        raise RenderError(f"{where}: a boolean cannot be written as a value")
    if isinstance(value, tuple):
        return ",".join(value)
    return str(value)


def program_path(program: Program, block: TensorfoldBlock) -> str:
    """Where a program lives on the host, or its bare name when it is found on PATH."""
    if program.base is None:
        return program.relative
    return posixpath.join(block.paths[program.base], program.relative)


def _flag_words(flag: Flag, value: Value | None) -> list[str]:
    if value is None:
        if flag.optional:
            return []
        raise RenderError(f"flag {flag.name} has no value")
    if flag.mode is FlagMode.VALUE:
        return [flag.name, shlex.quote(_text(value, flag.name))]
    if not isinstance(value, bool):
        raise RenderError(f"flag {flag.name} needs a boolean, got {value!r}")
    present = value if flag.mode is FlagMode.WHEN_TRUE else not value
    return [flag.name] if present else []


def render_command(
    profile: Profile,
    block: TensorfoldBlock,
    *,
    model: str,
    defaults: Mapping[str, Any],
    topology: Topology,
) -> str:
    """The complete serve command for ``topology.rank``."""
    return render_spec(
        profile.spec_for(topology.rank),
        profile,
        block,
        model=model,
        defaults=defaults,
        topology=topology,
    )


def render_spec(
    spec: CommandSpec,
    profile: Profile,
    block: TensorfoldBlock,
    *,
    model: str,
    defaults: Mapping[str, Any],
    topology: Topology,
) -> str:
    """The shell command a :class:`CommandSpec` describes, for one rank."""
    if topology.world_size not in profile.world_sizes:
        raise RenderError(
            f"profile {profile.name} runs on {sorted(profile.world_sizes)} nodes, "
            f"not {topology.world_size}"
        )
    if not 0 <= topology.rank < topology.world_size:
        raise RenderError(f"rank {topology.rank} is outside a world of {topology.world_size}")

    def look(ref: Ref) -> Value | None:
        return _resolve(ref, block=block, model=model, defaults=defaults, topology=topology)

    words: list[str] = []
    env_words: list[str] = []
    for env in spec.env:
        value = look(env.value)
        if value is None:
            if env.optional:
                continue
            raise RenderError(f"environment variable {env.name} has no value")
        env_words.append(f"{env.name}={shlex.quote(_text(value, env.name))}")
    if env_words:
        words += ["env", *env_words]

    words.append(shlex.quote(program_path(spec.program, block)))
    if spec.subcommand is not None:
        words.append(spec.subcommand)
    for position, ref in enumerate(spec.positionals):
        value = look(ref)
        if value is None:
            raise RenderError(f"positional argument {position} ({ref.name}) has no value")
        words.append(shlex.quote(_text(value, ref.name)))
    for flag in spec.flags:
        words += _flag_words(flag, look(flag.value) if flag.value is not None else None)
    return " ".join(words)
