"""Names a build script template may use.

Every host path is shell-quoted, so a script writes ``{model}`` where it needs the path and never
quotes it again. ``rank``, ``world_size`` and ``master_addr`` describe the node the script runs on.
"""

from __future__ import annotations

import shlex

from sparkrun_tensorfold_plugin.paths import host_path
from sparkrun_tensorfold_plugin.profiles.model import Profile
from sparkrun_tensorfold_plugin.recipe_block import TensorfoldBlock


def script_context(
    profile: Profile,
    block: TensorfoldBlock,
    model: str,
    *,
    rank: int,
    world_size: int,
    master_addr: str | None,
) -> dict[str, str]:
    values = {
        name: shlex.quote(host_path(profile, block, model, name))
        for name in sorted(profile.host_targets())
    }
    values["rank"] = str(rank)
    values["world_size"] = str(world_size)
    if master_addr is not None:
        values["master_addr"] = shlex.quote(master_addr)
    return values
