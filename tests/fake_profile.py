"""A test-only engine profile.

It is built from the same model types as the real profiles but contains none of their knowledge,
which is the point: the build machinery must work for any profile. The "engine" is a few lines of
shell in a small image, so the tests run real containers without GPUs or large downloads.
"""

from __future__ import annotations

from sparkrun_tensorfold_plugin.profiles.model import (
    Access,
    Artifact,
    CommandSpec,
    CoordinatedPlan,
    DerivedPath,
    DockerRequirements,
    Env,
    Executables,
    Mount,
    PathKey,
    Profile,
    Program,
    RankCacheFooter,
    ScriptPlan,
    SizeManifest,
    const,
    model,
    topology,
)

IMAGE = "bash:5.2"
MAGIC = "FAKEFOOT"
CACHE_CHECK = RankCacheFooter("rank_cache_dir", "rank", ".bin", MAGIC)

# The fake engine: each rank writes its cache file ending in the footer, then keeps running like
# a real server would, so the tests also exercise stopping a job that is still alive.
ENGINE_SCRIPT = (
    'd=/cache/rank-cache; mkdir -p "$d"; '
    f'printf "weights{MAGIC}" > "$d/rank${{RANK}}of${{WORLD}}-abc.bin"; '
    'chmod -R a+rwX "$d"; sleep 300'
)

ENGINE = CommandSpec(
    program=Program(base=None, relative="bash"),
    subcommand="-c",
    positionals=(const(ENGINE_SCRIPT),),
    flags=(),
    env=(Env("RANK", topology("rank")), Env("WORLD", topology("world_size"))),
)

NOOP = CommandSpec(
    program=Program("binaries", "head"), subcommand=None, positionals=(model(),), flags=(), env=()
)


def fake_profile(
    *, data_script: str | None = None, engine: CommandSpec = ENGINE, data_timeout: int = 120
) -> Profile:
    return Profile(
        name="fake",
        description="test-only profile",
        image=IMAGE,
        world_sizes=frozenset({1, 2, 3, 4, 8}),
        readiness_path="/health",
        serve_keys=(),
        path_keys=(
            PathKey("binaries", Access.RW),
            PathKey("assets", Access.RW),
            PathKey("kit", Access.RO),
            PathKey("cache_dir", Access.RW),
        ),
        derived_paths=(DerivedPath("rank_cache_dir", base="cache_dir", relative="rank-cache"),),
        head=NOOP,
        worker=NOOP,
        docker=DockerRequirements(
            gpus=None, user=None, ipc="host", cap_add=(), ulimits=(), devices=()
        ),
        artifacts=(
            Artifact(
                name="data",
                check=SizeManifest("assets", (("d.txt", 6),)),
                plan=ScriptPlan(
                    image=IMAGE,
                    script=data_script or "printf 'hello!' > {assets}/d.txt",
                    mounts=(Mount("assets", None, Access.RW),),
                    needs=(),
                    as_user=True,
                    gpu=False,
                    timeout_s=data_timeout,
                ),
            ),
            Artifact(
                name="tool",
                check=Executables(
                    rank0=(Program("binaries", "bin/tool"),),
                    others=(Program("binaries", "bin/tool"),),
                ),
                plan=ScriptPlan(
                    image=IMAGE,
                    script=(
                        "mkdir -p {binaries}/bin\n"
                        "printf '#!/bin/sh\\n' > {binaries}/bin/tool\n"
                        "chmod +x {binaries}/bin/tool"
                    ),
                    mounts=(Mount("binaries", None, Access.RW),),
                    needs=("data",),
                    as_user=True,
                    gpu=False,
                    timeout_s=120,
                ),
            ),
            Artifact(
                name="cache",
                check=CACHE_CHECK,
                plan=CoordinatedPlan(
                    image=IMAGE,
                    setup="true",
                    command=engine,
                    mounts=(Mount("cache_dir", "/cache", Access.RW),),
                    needs=("data",),
                    done=CACHE_CHECK,
                    timeout_s=120,
                ),
            ),
        ),
    )
