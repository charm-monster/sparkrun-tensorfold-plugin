"""Profile for ``hello-engine``, a fictional tensor-parallel engine.

It exists to show every part of the profile contract with invented values, and the plugin's tests
use it as their fixture. Nothing here is a real program, model or download. A real profile has the
same shape: the programs each rank runs, where their inputs come from, what they must agree on
across ranks, what the containers need, and how each artifact is checked and built.

``hello-engine`` runs on 2 or 4 nodes. Rank 0 runs ``hello-serve``, which serves HTTP and waits for
the other ranks at a rendezvous address. Every other rank runs ``hello-follow``, which connects to
it.
"""

from __future__ import annotations

from sparkrun_tensorfold_plugin.profiles.model import (
    Access,
    Artifact,
    AtMost,
    CommandSpec,
    CoordinatedPlan,
    DerivedPath,
    DockerRequirements,
    Env,
    Executables,
    FileSha256,
    Flag,
    FlagMode,
    Mount,
    PathKey,
    Profile,
    Program,
    RankCacheFooter,
    ScriptPlan,
    ServeKey,
    Sha256Manifest,
    SizeManifest,
    ValueType,
    block,
    const,
    derived,
    model,
    network,
    path,
    serve,
    topology,
)
from sparkrun_tensorfold_plugin.profiles.scripts import (
    fetch_verified,
    hf_download,
    user_script,
)

# The engine documents four concurrent streams unless HELLO_STREAMS says otherwise.
ENGINE_DEFAULT_STREAMS = 4

IMAGE = "example.invalid/hello-engine:1.0"

RANK_CACHE_SUBDIR = "rank-cache"
LOOKUP_NAME = "lookup.json"
RANK_CACHE_MAGIC = "HELLORC1"  # last 8 bytes of a complete rank cache file
BUILD_CACHE_MOUNT = "/var/cache/hello"  # where the build container sees the cache directory

# Pinned sources. Every artifact below is built from exactly these.
ENGINE_REPO = "https://example.invalid/hello-engine.git"
ENGINE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
LOOKUP_SCRIPT_URL = "https://example.invalid/hello-engine/make_lookup.py"
LOOKUP_SCRIPT_SHA256 = "a" * 64
LOOKUP_SHA256 = "b" * 64

# Environment every rank needs so that the ranks agree on the fabric.
_FABRIC_ENV = (
    Env("HELLO_RDMA_DEVICES", network("rdma_devices")),
    Env("HELLO_GID_INDEX", network("gid_index")),
    Env("HELLO_SOCKET_IFNAME", network("socket_interface")),
    Env("HELLO_LOG_LEVEL", const("warn")),
)

_HEAD_PROGRAM = Program(base="binaries", relative="bin/hello-serve")
_WORKER_PROGRAM = Program(base="binaries", relative="bin/hello-follow")

_RANK_CACHE_CHECK = RankCacheFooter(
    target="rank_cache_dir", prefix="rank", suffix=".bin", magic=RANK_CACHE_MAGIC
)

# The build tool writes the rank cache while it starts up. Every rank runs the same command with
# its own rank; rank 0 also serves a throwaway HTTP port that nothing uses.
_RANK_CACHE_BUILD_COMMAND = CommandSpec(
    program=Program(base=None, relative="hello-build"),
    subcommand="warm",
    positionals=(model(),),
    flags=(
        Flag("--tp", topology("world_size")),
        Flag("--rank", topology("rank")),
        Flag("--master", topology("master_addr")),
        Flag("--port", const(18890)),
        Flag("--context", const(4096)),
        Flag("--vision", const(True), FlagMode.WHEN_TRUE),
    ),
    env=(
        Env("HELLO_RDMA_DEVICES", network("rdma_devices")),
        Env("HELLO_GID_INDEX", network("gid_index")),
        Env("HELLO_SOCKET_IFNAME", network("socket_interface")),
        Env("HELLO_CACHE", const(f"{BUILD_CACHE_MOUNT}/{RANK_CACHE_SUBDIR}")),
        Env("HELLO_ASSETS", path("assets")),
    ),
)

HELLO_ENGINE = Profile(
    name="hello-engine",
    description="A fictional tensor-parallel engine, for documentation and tests",
    image=IMAGE,
    world_sizes=frozenset({2, 4}),
    readiness_path="/health",
    serve_keys=(
        ServeKey("host", ValueType.STR, required=True),
        ServeKey("port", ValueType.INT, required=True),
        ServeKey("served_model_name", ValueType.STR, required=True),
        ServeKey("max_model_len", ValueType.INT, required=True),
        ServeKey("parallel", ValueType.INT, required=True),
        ServeKey("streams", ValueType.INT, required=False),
        ServeKey("thinking", ValueType.BOOL, required=False),
        ServeKey("max_tokens", ValueType.INT, required=False),
        ServeKey("temperature", ValueType.FLOAT, required=False),
        ServeKey("top_p", ValueType.FLOAT, required=False),
        ServeKey("top_k", ValueType.INT, required=False),
    ),
    path_keys=(
        PathKey("binaries", Access.RO),
        PathKey("assets", Access.RO),
        PathKey("kit", Access.RO),
        PathKey("cache_dir", Access.RW),
    ),
    derived_paths=(
        DerivedPath("rank_cache_dir", base="cache_dir", relative=RANK_CACHE_SUBDIR),
        DerivedPath("lookup_table", base="cache_dir", relative=LOOKUP_NAME),
        DerivedPath("kernels_dir", base="kit", relative="kernels"),
    ),
    head=CommandSpec(
        program=_HEAD_PROGRAM,
        subcommand="serve",
        positionals=(model(),),
        flags=(
            Flag("--host", serve("host")),
            Flag("--port", serve("port")),
            Flag("--name", serve("served_model_name")),
            Flag("--api-key-file", block("api_key_file")),
            Flag("--context", serve("max_model_len")),
            Flag("--parallel", serve("parallel")),
            Flag("--thinking", serve("thinking"), FlagMode.WHEN_TRUE, optional=True),
            Flag("--no-thinking", serve("thinking"), FlagMode.WHEN_FALSE, optional=True),
            Flag("--max-tokens", serve("max_tokens"), optional=True),
            Flag("--temperature", serve("temperature"), optional=True),
            Flag("--top-p", serve("top_p"), optional=True),
            Flag("--top-k", serve("top_k"), optional=True),
            Flag("--backend", const("cpu")),
        ),
        env=(
            *_FABRIC_ENV,
            Env("HELLO_WORLD", topology("world_size")),
            Env("HELLO_MASTER", topology("master_addr")),
            Env("HELLO_MASTER_PORT", topology("master_port")),
            Env("HELLO_CACHE", derived("rank_cache_dir")),
            Env("HELLO_KIT", path("kit")),
            Env("HELLO_KERNELS", derived("kernels_dir")),
            Env("HELLO_ASSETS", path("assets")),
            Env("HELLO_LOOKUP", derived("lookup_table")),
            Env("HELLO_GRAPHS", const(1)),
            Env("HELLO_STREAMS", serve("streams"), optional=True),
        ),
    ),
    worker=CommandSpec(
        program=_WORKER_PROGRAM,
        subcommand=None,
        positionals=(
            model(),
            derived("rank_cache_dir"),
            topology("rank"),
            topology("world_size"),
            topology("master_addr"),
            topology("master_port"),
            path("kit"),
        ),
        flags=(
            Flag("--assets", path("assets")),
            Flag("--lookup", derived("lookup_table")),
            Flag("--pool", serve("max_model_len")),
            Flag("--streams", serve("streams"), optional=True),
            Flag("--rdma", network("rdma_devices")),
            Flag("--graphs", const(1)),
        ),
        env=_FABRIC_ENV,
    ),
    docker=DockerRequirements(
        gpus=None,
        user="root",
        ipc="host",
        cap_add=("IPC_LOCK",),
        ulimits=("memlock=-1", "stack=67108864"),
        devices=("/dev/infiniband",),
    ),
    artifacts=(
        Artifact(
            name="model",
            check=SizeManifest("model", (("weights.bin", 1048576), ("config.json", 512))),
            plan=ScriptPlan(
                image=IMAGE,
                script=hf_download(
                    repo="example-org/hello-model",
                    revision="1111111111111111111111111111111111111111",
                    dest="model",
                ),
                mounts=(Mount("model", None, Access.RW, parent=True),),
                needs=(),
                as_user=True,
                gpu=False,
                timeout_s=3600,
            ),
        ),
        Artifact(
            name="assets",
            check=SizeManifest("assets", (("shard-0.bin", 4096), ("shard-1.bin", 4096))),
            plan=ScriptPlan(
                image=IMAGE,
                script=hf_download(
                    repo="example-org/hello-assets",
                    revision="2222222222222222222222222222222222222222",
                    dest="assets",
                    include=("shard-0.bin", "shard-1.bin"),
                ),
                mounts=(Mount("assets", None, Access.RW, parent=True),),
                needs=(),
                as_user=True,
                gpu=False,
                timeout_s=3600,
            ),
        ),
        Artifact(
            name="kit",
            check=Sha256Manifest("kit", "MANIFEST"),
            plan=ScriptPlan(
                image=IMAGE,
                script=hf_download(
                    repo="example-org/hello-kit",
                    revision="3333333333333333333333333333333333333333",
                    dest="kit",
                    include=("kit/*",),
                    repo_subdir="kit",
                ),
                mounts=(Mount("kit", None, Access.RW, parent=True),),
                needs=(),
                as_user=True,
                gpu=False,
                timeout_s=1800,
            ),
        ),
        Artifact(
            name="lookup_table",
            check=FileSha256("lookup_table", LOOKUP_SHA256),
            plan=ScriptPlan(
                image=IMAGE,
                script=user_script(
                    fetch_verified(
                        url=LOOKUP_SCRIPT_URL,
                        sha256=LOOKUP_SCRIPT_SHA256,
                        to="$HOME/make_lookup.py",
                    ),
                    'mkdir -p "$(dirname {lookup_table})"',
                    "python3 $HOME/make_lookup.py {model} {lookup_table}",
                ),
                mounts=(
                    Mount("model", None, Access.RO),
                    Mount("cache_dir", None, Access.RW),
                ),
                needs=("model",),
                as_user=True,
                gpu=False,
                timeout_s=900,
            ),
        ),
        Artifact(
            name="binaries",
            check=Executables(rank0=(_HEAD_PROGRAM,), others=(_WORKER_PROGRAM,)),
            plan=ScriptPlan(
                image=IMAGE,
                script=user_script(
                    "cd $HOME",
                    f"git clone --quiet {ENGINE_REPO} engine",
                    f"git -C engine checkout --quiet {ENGINE_COMMIT}",
                    "cd engine",
                    "make install PREFIX=$HOME/out",
                    "mkdir -p {binaries}",
                    "cp -a $HOME/out/. {binaries}/",
                ),
                mounts=(Mount("binaries", None, Access.RW),),
                needs=(),
                as_user=True,
                gpu=False,
                timeout_s=1800,
            ),
        ),
        Artifact(
            name="rank_cache",
            check=_RANK_CACHE_CHECK,
            plan=CoordinatedPlan(
                image=IMAGE,
                setup=f"pip install --quiet --no-cache-dir hello-build=={ENGINE_COMMIT[:7]}",
                command=_RANK_CACHE_BUILD_COMMAND,
                mounts=(
                    Mount("model", None, Access.RO),
                    Mount("assets", None, Access.RO),
                    Mount("cache_dir", BUILD_CACHE_MOUNT, Access.RW),
                ),
                needs=("model", "assets", "lookup_table"),
                done=_RANK_CACHE_CHECK,
                timeout_s=7200,
            ),
        ),
    ),
    constraints=(AtMost("parallel", "streams", ENGINE_DEFAULT_STREAMS),),
)
