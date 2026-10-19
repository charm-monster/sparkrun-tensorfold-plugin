"""The docker flags a profile's containers need on top of what sparkrun sets itself.

They are passed as sparkrun's ``extra_docker_opts``, which both supported sparkrun versions append
after their own flags. Docker uses the last value of a repeated ``--ipc``, ``--user`` or
same-named ``--ulimit``, so what the profile asks for wins; list-like flags (``--cap-add``,
``--device``, ``-v``) add to what sparkrun sets. GPU access is left to sparkrun, which knows the
platform's spelling of it.

``extra_docker_opts`` reaches every container alike. What only some ranks need is
:func:`rank_options`, applied per container by :mod:`sparkrun_tensorfold_plugin.executors`.
"""

from __future__ import annotations

from sparkrun_tensorfold_plugin.profiles.model import Profile, Source
from sparkrun_tensorfold_plugin.recipe_block import TensorfoldBlock


def docker_options(profile: Profile, block: TensorfoldBlock) -> list[str]:
    """``docker run`` flags for every container of this launch."""
    docker = profile.docker
    options = ["--ipc", docker.ipc]
    if docker.user is not None:
        options += ["--user", docker.user]
    for capability in docker.cap_add:
        options += ["--cap-add", capability]
    for limit in docker.ulimits:
        options += ["--ulimit", limit]
    for device in docker.devices:
        options += ["--device", device]
    for key in profile.path_keys:
        path = block.paths[key.name]
        options += ["-v", f"{path}:{path}:{key.access.value}"]
    return options


def key_file_ranks(profile: Profile, world_size: int) -> tuple[int, ...]:
    """The ranks whose serve command reads the API key file."""
    return profile.ranks_referencing(Source.BLOCK, "api_key_file", world_size)


def rank_options(profile: Profile, block: TensorfoldBlock, world_size: int) -> dict[int, list[str]]:
    """Flags only some ranks need, by rank. A rank with none is absent.

    The API key file is mounted on exactly the ranks whose command names it, so a rank that never
    reads it needs no copy of it.
    """
    if block.api_key_file is None:
        return {}
    mount = ["-v", f"{block.api_key_file}:{block.api_key_file}:ro"]
    return {rank: list(mount) for rank in key_file_ranks(profile, world_size)}
