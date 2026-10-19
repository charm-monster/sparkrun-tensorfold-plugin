"""Resolve a profile's host-path names to concrete paths."""

from __future__ import annotations

import posixpath

from sparkrun_tensorfold_plugin.profiles.model import Profile
from sparkrun_tensorfold_plugin.recipe_block import TensorfoldBlock


def host_path(profile: Profile, block: TensorfoldBlock, model: str, name: str) -> str:
    """The host path named ``name``: ``"model"``, a path key, or a derived path."""
    if name == "model":
        return model
    if name in block.paths:
        return block.paths[name]
    for derived in profile.derived_paths:
        if derived.name == name:
            return posixpath.join(block.paths[derived.base], derived.relative)
    raise KeyError(f"profile {profile.name} has no host path named {name}")
