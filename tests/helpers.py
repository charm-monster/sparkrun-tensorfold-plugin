"""Shared test helpers: a real local shell as the "host", and blocks pointing at temp dirs."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sparkrun_tensorfold_plugin.profiles import get_profile
from sparkrun_tensorfold_plugin.profiles.model import Profile, Source
from sparkrun_tensorfold_plugin.recipe_block import TensorfoldBlock, parse_block

# Directory under the test root for each path key; a key not listed here uses its own name.
PATH_DIRS = {"binaries": "bin", "cache_dir": "cache"}


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str


def local_run(host: str, command: str) -> Result:  # noqa: ARG001
    """Run a command in a real local bash, as sparkrun does for a local host."""
    done = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=False)
    return Result(done.returncode, done.stdout, done.stderr)


def block_in(root: Path, profile_name: str = "hello-engine") -> tuple[TensorfoldBlock, str]:
    """A block whose paths are directories under ``root``, and the model directory."""
    profile = get_profile(profile_name)
    raw: dict[str, Any] = {
        "profile": profile_name,
        "prepare": "build",
        "paths": {
            key.name: str(root / PATH_DIRS.get(key.name, key.name)) for key in profile.path_keys
        },
        "rendezvous_port": 29620,
    }
    if profile.referenced(Source.NETWORK):
        raw["network"] = {"rdma_devices": ["rdma0"], "socket_interface": "eth0", "gid_index": 3}
    if profile.referenced(Source.BLOCK):
        raw["api_key_file"] = str(root / "api.key")
    return parse_block(raw), str(root / "model")


def profile_of(block: TensorfoldBlock) -> Profile:
    return block.profile
