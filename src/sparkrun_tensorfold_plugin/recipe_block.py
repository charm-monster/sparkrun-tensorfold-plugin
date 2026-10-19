"""The plugin-owned ``tensorfold`` recipe block.

All plugin-specific recipe data lives under one top-level key, so sparkrun's own recipe schema is
untouched and nothing about an engine leaks into ``defaults:`` beyond serve values. The accepted
fields are derived from the profile the block names: a profile that never references the network
accepts no ``network`` field, and one that never references an API key file accepts none.

Every field is validated and every required field must be present. There are no defaults.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sparkrun_tensorfold_plugin.profiles import Profile, ProfileError, get_profile
from sparkrun_tensorfold_plugin.profiles.model import Source, ValueType

KEY = "tensorfold"

NetworkValue = str | int | tuple[str, ...]

_TOKEN_EXTRAS = frozenset("_-.:")


class RecipeBlockError(ValueError):
    """The ``tensorfold`` block is invalid. sparkrun wraps this into a ``RecipeError``."""


class PrepareMode(Enum):
    BUILD = "build"  # build any missing artifact, then launch
    VERIFY = "verify"  # fail with the exact build command if an artifact is missing


@dataclass(frozen=True)
class TensorfoldBlock:
    profile: Profile
    prepare: PrepareMode
    paths: Mapping[str, str]
    network: Mapping[str, NetworkValue]
    rendezvous_port: int
    api_key_file: str | None

    def to_dict(self) -> dict[str, Any]:
        """Canonical YAML-safe form. It holds paths and names only, never a secret."""
        out: dict[str, Any] = {
            "profile": self.profile.name,
            "prepare": self.prepare.value,
            "paths": {k: self.paths[k] for k in sorted(self.paths)},
            "rendezvous_port": self.rendezvous_port,
        }
        if self.network:
            out["network"] = {
                k: list(v) if isinstance(v, tuple) else v for k, v in sorted(self.network.items())
            }
        if self.api_key_file is not None:
            out["api_key_file"] = self.api_key_file
        return out


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecipeBlockError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _check_keys(
    given: Mapping[str, Any], allowed: frozenset[str], required: frozenset[str], where: str
) -> None:
    unknown = sorted(set(given) - allowed)
    if unknown:
        allowed_list = ", ".join(sorted(allowed))
        raise RecipeBlockError(
            f"{where} has unknown key(s) {', '.join(unknown)}; allowed: {allowed_list}"
        )
    missing = sorted(required - set(given))
    if missing:
        raise RecipeBlockError(f"{where} is missing required key(s) {', '.join(missing)}")


def _token(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecipeBlockError(f"{where} must be a non-empty string")
    if not all(c.isalnum() or c in _TOKEN_EXTRAS for c in value):
        raise RecipeBlockError(f"{where} may only contain letters, digits and _ - . :")
    return value


def _abs_path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not os.path.isabs(value):
        raise RecipeBlockError(f"{where} must be an absolute path")
    if value.rstrip("/") == "":
        raise RecipeBlockError(f"{where} must not be the filesystem root")
    return value


def _int(value: Any, where: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecipeBlockError(f"{where} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise RecipeBlockError(f"{where} must be {bound}")
    return value


def _network(raw: Any, profile: Profile) -> dict[str, NetworkValue]:
    needed = profile.referenced(Source.NETWORK)
    if not needed:
        if raw is not None:
            raise RecipeBlockError(f"profile {profile.name} does not use a network block")
        return {}
    given = _mapping(raw, f"{KEY}.network")
    _check_keys(given, needed, needed, f"{KEY}.network")
    out: dict[str, NetworkValue] = {}
    for name in sorted(needed):
        where = f"{KEY}.network.{name}"
        value = given[name]
        if name == "rdma_devices":
            if not isinstance(value, list) or not value:
                raise RecipeBlockError(f"{where} must be a non-empty list")
            out[name] = tuple(_token(item, f"{where}[]") for item in value)
        elif name == "gid_index":
            out[name] = _int(value, where, minimum=0)
        else:
            out[name] = _token(value, where)
    return out


def parse_block(raw: Any) -> TensorfoldBlock:
    """Validate the raw ``tensorfold`` mapping and return it as a typed block."""
    given = _mapping(raw, KEY)
    if "profile" not in given:
        raise RecipeBlockError(f"{KEY} is missing required key(s) profile")
    try:
        profile = get_profile(given["profile"] if isinstance(given["profile"], str) else "")
    except ProfileError as error:
        raise RecipeBlockError(f"{KEY}.profile: {error}") from error

    wants_key_file = "api_key_file" in profile.referenced(Source.BLOCK)
    wants_network = bool(profile.referenced(Source.NETWORK))
    required = {"profile", "prepare", "paths", "rendezvous_port"}
    if wants_network:
        required.add("network")
    if wants_key_file:
        required.add("api_key_file")
    _check_keys(given, frozenset(required), frozenset(required), KEY)

    try:
        prepare = PrepareMode(given["prepare"])
    except ValueError:
        modes = ", ".join(m.value for m in PrepareMode)
        raise RecipeBlockError(f"{KEY}.prepare must be one of: {modes}") from None

    path_names = frozenset(p.name for p in profile.path_keys)
    raw_paths = _mapping(given["paths"], f"{KEY}.paths")
    _check_keys(raw_paths, path_names, path_names, f"{KEY}.paths")
    paths = {name: _abs_path(raw_paths[name], f"{KEY}.paths.{name}") for name in sorted(path_names)}

    return TensorfoldBlock(
        profile=profile,
        prepare=prepare,
        paths=paths,
        network=_network(given.get("network"), profile),
        rendezvous_port=_int(
            given["rendezvous_port"], f"{KEY}.rendezvous_port", minimum=1, maximum=65535
        ),
        api_key_file=(
            _abs_path(given["api_key_file"], f"{KEY}.api_key_file") if wants_key_file else None
        ),
    )


class RecipeBlockHandler:
    """sparkrun recipe-item handler for the ``tensorfold`` key.

    Relational checks (the block against ``defaults:``, ``model:`` and so on) are not reported
    here: sparkrun surfaces handler findings only as suggestions. They are raised as errors by
    the runtime's ``validate_recipe`` through :func:`relational_errors`.
    """

    def parse(self, value: Any, recipe: Any) -> TensorfoldBlock:  # noqa: ARG002
        return parse_block(value)

    def validate(self, value: Any, recipe: Any) -> list[str]:  # noqa: ARG002
        return []

    def export(self, value: Any, recipe: Any) -> dict[str, Any]:  # noqa: ARG002
        return value.to_dict()  # type: ignore[no-any-return]


# One object for the life of the process: sparkrun 0.3.10 treats a second registration as a
# no-op only when the handler object is identical.
HANDLER = RecipeBlockHandler()


_PYTHON_TYPES: dict[ValueType, tuple[type, ...]] = {
    ValueType.INT: (int,),
    ValueType.FLOAT: (int, float),
    ValueType.STR: (str,),
    ValueType.BOOL: (bool,),
}


def relational_errors(
    block: TensorfoldBlock,
    *,
    model: str,
    defaults: Mapping[str, Any],
    min_nodes: int,
    max_nodes: int | None,
    readiness_inference: Any,
) -> list[str]:
    """Problems that only appear when the block is read against the rest of the recipe."""
    profile = block.profile
    errors: list[str] = []

    if not os.path.isabs(model):
        errors.append(
            "model must be an absolute directory: the weights are pre-placed on every node and "
            "mounted at the same path"
        )

    lowest, highest = min(profile.world_sizes), max(profile.world_sizes)
    if max_nodes is None:
        errors.append(f"max_nodes is required (profile {profile.name} runs {lowest}..{highest})")
    if min_nodes < lowest or (max_nodes is not None and max_nodes > highest):
        errors.append(
            f"profile {profile.name} runs on {sorted(profile.world_sizes)} nodes; "
            f"recipe allows {min_nodes}..{max_nodes}"
        )

    if readiness_inference is not False:
        errors.append(
            "readiness.inference must be false: the server needs an API key, so sparkrun cannot "
            "run its inference probe and polls the unauthenticated "
            f"{profile.readiness_path} instead"
        )

    for key in profile.serve_keys:
        if key.name not in defaults:
            if key.required:
                errors.append(f"defaults.{key.name} is required by profile {profile.name}")
            continue
        value = defaults[key.name]
        allowed = _PYTHON_TYPES[key.type]
        if not isinstance(value, allowed) or (
            key.type is not ValueType.BOOL and isinstance(value, bool)
        ):
            errors.append(f"defaults.{key.name} must be of type {key.type.value}")

    for rule in profile.constraints:
        left = defaults.get(rule.left)
        right = defaults.get(rule.right, rule.right_default)
        if isinstance(left, int) and isinstance(right, int) and left > right:
            errors.append(
                f"defaults.{rule.left} ({left}) may not exceed {rule.right} ({right}); "
                f"set defaults.{rule.right} to raise the limit"
            )
    return errors
