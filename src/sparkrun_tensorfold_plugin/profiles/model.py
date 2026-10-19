"""Data model for engine profiles.

A profile is the only place the contract of one engine variant is written down: which program
each rank runs, which flags and environment variables it receives, which values must agree
across ranks, and which container capabilities it needs. The core never names any of these.

Mirrored values are declared once. A value is a :class:`Ref`; the head command and the worker
command both point at the same ``Ref`` wherever the engine requires them to agree, so they cannot
drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from string import Formatter


class ProfileError(ValueError):
    """A profile is internally inconsistent."""


class Source(Enum):
    """Where a value comes from when a command is rendered."""

    MODEL = "model"  # the recipe's ``model:`` (an absolute directory)
    SERVE = "serve"  # a key under the recipe's ``defaults:``
    PATH = "path"  # a key under ``tensorfold.paths`` (a host directory, identity-mounted)
    DERIVED = "derived"  # a path joined from a path key and a fixed relative name
    NETWORK = "network"  # a field of ``tensorfold.network``
    BLOCK = "block"  # a scalar field of the ``tensorfold`` block itself
    TOPOLOGY = "topology"  # computed from the placement at launch time
    CONST = "const"  # a literal fixed by the profile


class ValueType(Enum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"


class FlagMode(Enum):
    VALUE = "value"  # ``--flag value``
    WHEN_TRUE = "when_true"  # bare ``--flag`` emitted only when the value is true
    WHEN_FALSE = "when_false"  # bare ``--flag`` emitted only when the value is false


class Access(Enum):
    RO = "ro"
    RW = "rw"


# Topology values the launcher computes. Every profile may reference any of these.
TOPOLOGY_NAMES = frozenset({"rank", "world_size", "master_addr", "master_port"})

# Fields of ``tensorfold.network`` the core knows how to validate.
NETWORK_NAMES = frozenset({"rdma_devices", "socket_interface", "gid_index"})

# Scalar fields of the ``tensorfold`` block that a profile may reference.
BLOCK_NAMES = frozenset({"api_key_file"})

ConstValue = str | int | float | bool


@dataclass(frozen=True)
class Ref:
    """A named value and where it is read from."""

    source: Source
    name: str
    const: ConstValue | None = None

    def __post_init__(self) -> None:
        if self.source is Source.CONST and self.const is None:
            raise ProfileError("a CONST ref needs a value")
        if self.source is not Source.CONST and self.const is not None:
            raise ProfileError("only a CONST ref carries a value")


def model() -> Ref:
    return Ref(Source.MODEL, "model")


def serve(name: str) -> Ref:
    return Ref(Source.SERVE, name)


def path(name: str) -> Ref:
    return Ref(Source.PATH, name)


def derived(name: str) -> Ref:
    return Ref(Source.DERIVED, name)


def network(name: str) -> Ref:
    return Ref(Source.NETWORK, name)


def block(name: str) -> Ref:
    return Ref(Source.BLOCK, name)


def topology(name: str) -> Ref:
    return Ref(Source.TOPOLOGY, name)


def const(value: ConstValue) -> Ref:
    return Ref(Source.CONST, str(value), value)


@dataclass(frozen=True)
class ServeKey:
    """A key a recipe may set under ``defaults:``."""

    name: str
    type: ValueType
    required: bool


@dataclass(frozen=True)
class PathKey:
    """A host directory the recipe supplies under ``tensorfold.paths``."""

    name: str
    access: Access


@dataclass(frozen=True)
class DerivedPath:
    """``<base path key>/<relative>``: a fixed name inside a recipe-supplied directory."""

    name: str
    base: str
    relative: str


@dataclass(frozen=True)
class Flag:
    name: str
    value: Ref | None = None
    mode: FlagMode = FlagMode.VALUE
    optional: bool = False  # an unset optional value omits the flag; it is not defaulted

    def __post_init__(self) -> None:
        if self.mode is FlagMode.VALUE and self.value is None:
            raise ProfileError(f"flag {self.name} has mode VALUE but no value")
        if self.mode is not FlagMode.VALUE and self.value is None:
            raise ProfileError(f"flag {self.name} needs a boolean value to decide presence")


@dataclass(frozen=True)
class Env:
    name: str
    value: Ref
    optional: bool = False  # an unset optional value omits the variable; it is not defaulted


@dataclass(frozen=True)
class Program:
    """An executable at ``<base path key>/<relative>``, or ``relative`` alone when ``base`` is
    ``None`` (a command found on the container's PATH)."""

    base: str | None
    relative: str


@dataclass(frozen=True)
class CommandSpec:
    """The command one rank runs: ``env K=V ... program [subcommand] positionals flags``."""

    program: Program
    subcommand: str | None
    positionals: tuple[Ref, ...]
    flags: tuple[Flag, ...]
    env: tuple[Env, ...]


@dataclass(frozen=True)
class SizeManifest:
    """Every named file in ``target`` exists with exactly this size in bytes."""

    target: str  # "model", a path key, or a derived path
    files: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Sha256Manifest:
    """``sha256sum -c`` of a manifest file inside ``target`` passes."""

    target: str
    file: str


@dataclass(frozen=True)
class FileSha256:
    """The file at ``target`` has this SHA-256."""

    target: str
    sha256: str


@dataclass(frozen=True)
class RankCacheFooter:
    """``target`` holds exactly one ``<prefix>{rank}of{world}-*<suffix>`` file whose last bytes
    are ``magic``, and no partial leftovers. The engine writes the footer last, so a valid footer
    means the file is complete."""

    target: str
    prefix: str
    suffix: str
    magic: str


@dataclass(frozen=True)
class Executables:
    """The programs a node needs are present and executable: ``rank0`` on the node that runs rank
    0, ``others`` on every other node. Nodes run different programs, so a node is only required to
    have its own."""

    rank0: tuple[Program, ...]
    others: tuple[Program, ...]


Check = SizeManifest | Sha256Manifest | FileSha256 | RankCacheFooter | Executables


@dataclass(frozen=True)
class Mount:
    """A host path made visible to a build container.

    ``container`` is the path inside the container; ``None`` means the same path as on the host.
    ``parent`` mounts the directory that contains the path instead of the path itself, for
    artifacts the job creates.
    """

    host: str  # "model" or a path key
    container: str | None
    access: Access
    parent: bool = False


@dataclass(frozen=True)
class ScriptPlan:
    """Build an artifact by running a bash script in one container on each node that lacks it.

    The script is a ``str.format`` template. See ``artifacts/context.py`` for the names it may use.
    """

    image: str
    script: str
    mounts: tuple[Mount, ...]
    needs: tuple[str, ...]  # artifacts that must verify before this one is built
    as_user: bool  # run as the invoking user (files stay owned by them) rather than root
    gpu: bool
    timeout_s: int


@dataclass(frozen=True)
class CoordinatedPlan:
    """Build an artifact with one container per rank, all started together, because the engine
    needs every rank alive to produce it. ``setup`` runs first in each container; then ``command``
    for that rank. The containers are stopped once every rank passes ``done``."""

    image: str
    setup: str
    command: CommandSpec
    mounts: tuple[Mount, ...]
    needs: tuple[str, ...]
    done: Check
    timeout_s: int


BuildPlan = ScriptPlan | CoordinatedPlan


@dataclass(frozen=True)
class Artifact:
    name: str
    check: Check
    plan: BuildPlan


@dataclass(frozen=True)
class DockerRequirements:
    """Container capabilities the engine needs, applied to every rank's container."""

    gpus: str | None  # None: no GPU flag (the engine does not need one)
    user: str | None  # container user the engine is proven with; None: any user will do
    ipc: str
    cap_add: tuple[str, ...]
    ulimits: tuple[str, ...]
    devices: tuple[str, ...]


@dataclass(frozen=True)
class AtMost:
    """``left`` must not exceed ``right``; ``right_default`` is the engine's own documented
    value, used only when the recipe leaves ``right`` unset."""

    left: str
    right: str
    right_default: int


def _spec_refs(spec: CommandSpec) -> tuple[Ref, ...]:
    """Every reference one command uses."""
    return (
        *spec.positionals,
        *(f.value for f in spec.flags if f.value is not None),
        *(e.value for e in spec.env),
    )


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    image: str
    world_sizes: frozenset[int]
    readiness_path: str
    serve_keys: tuple[ServeKey, ...]
    path_keys: tuple[PathKey, ...]
    derived_paths: tuple[DerivedPath, ...]
    head: CommandSpec
    worker: CommandSpec
    docker: DockerRequirements
    artifacts: tuple[Artifact, ...]
    constraints: tuple[AtMost, ...] = field(default=())

    def command_specs(self) -> tuple[CommandSpec, ...]:
        """The serve commands and every build command."""
        builds = tuple(
            a.plan.command for a in self.artifacts if isinstance(a.plan, CoordinatedPlan)
        )
        return (self.head, self.worker, *builds)

    def refs(self) -> tuple[Ref, ...]:
        """Every reference in every command."""
        return tuple(ref for spec in self.command_specs() for ref in _spec_refs(spec))

    def referenced(self, source: Source) -> frozenset[str]:
        """Names of one source that the commands actually use."""
        return frozenset(r.name for r in self.refs() if r.source is source)

    def spec_for(self, rank: int) -> CommandSpec:
        """The serve command spec of one rank: rank 0 serves, every other rank follows."""
        return self.head if rank == 0 else self.worker

    def ranks_referencing(self, source: Source, name: str, world_size: int) -> tuple[int, ...]:
        """The ranks, in a world of ``world_size``, whose serve command uses this reference."""
        return tuple(
            rank
            for rank in range(world_size)
            if any(r.source is source and r.name == name for r in _spec_refs(self.spec_for(rank)))
        )

    def uses_master_address(self) -> bool:
        """Whether any command or build script needs rank 0's fabric address."""
        if "master_addr" in self.referenced(Source.TOPOLOGY):
            return True
        return any(
            field_name == "master_addr"
            for art in self.artifacts
            if isinstance(art.plan, ScriptPlan)
            for _, field_name, _, _ in Formatter().parse(art.plan.script)
        )

    def serve_key(self, name: str) -> ServeKey:
        for key in self.serve_keys:
            if key.name == name:
                return key
        raise ProfileError(f"profile {self.name} declares no serve key {name}")

    def validate(self) -> None:
        """Raise :class:`ProfileError` if any reference does not resolve."""
        serve_names = {k.name for k in self.serve_keys}
        path_names = {p.name for p in self.path_keys}
        derived_names = {d.name for d in self.derived_paths}

        if len(serve_names) != len(self.serve_keys):
            raise ProfileError(f"profile {self.name} declares a serve key twice")
        if len(path_names) != len(self.path_keys):
            raise ProfileError(f"profile {self.name} declares a path key twice")
        if not self.world_sizes or min(self.world_sizes) < 1:
            raise ProfileError(f"profile {self.name} needs world sizes of at least 1")

        for d in self.derived_paths:
            if d.base not in path_names:
                raise ProfileError(f"derived path {d.name} uses unknown path key {d.base}")
        for spec in self.command_specs():
            if spec.program.base is not None and spec.program.base not in path_names:
                raise ProfileError(f"program uses unknown path key {spec.program.base}")

        for ref in self.refs():
            if ref.source is Source.MODEL and ref.name != "model":
                raise ProfileError("the only model reference is 'model'")
            if ref.source is Source.SERVE and ref.name not in serve_names:
                raise ProfileError(f"unknown serve key {ref.name}")
            if ref.source is Source.PATH and ref.name not in path_names:
                raise ProfileError(f"unknown path key {ref.name}")
            if ref.source is Source.DERIVED and ref.name not in derived_names:
                raise ProfileError(f"unknown derived path {ref.name}")
            if ref.source is Source.NETWORK and ref.name not in NETWORK_NAMES:
                raise ProfileError(f"unknown network field {ref.name}")
            if ref.source is Source.BLOCK and ref.name not in BLOCK_NAMES:
                raise ProfileError(f"unknown block field {ref.name}")
            if ref.source is Source.TOPOLOGY and ref.name not in TOPOLOGY_NAMES:
                raise ProfileError(f"unknown topology value {ref.name}")

        self._validate_artifacts(path_names)

        if self.uses_master_address() and (
            "socket_interface" not in self.referenced(Source.NETWORK)
        ):
            raise ProfileError(
                "master_addr is read from the rank 0 fabric interface, so the commands must "
                "also reference network.socket_interface"
            )

        for rule in self.constraints:
            for key in (rule.left, rule.right):
                if key not in serve_names:
                    raise ProfileError(f"constraint uses unknown serve key {key}")

    def host_targets(self) -> frozenset[str]:
        """Names that resolve to a host path: the model, each path key, each derived path."""
        return frozenset(
            {"model"} | {p.name for p in self.path_keys} | {d.name for d in self.derived_paths}
        )

    def script_fields(self) -> frozenset[str]:
        """Names a build script may use as ``{field}``."""
        return self.host_targets() | {"rank", "world_size", "master_addr"}

    def _validate_artifacts(self, path_names: set[str]) -> None:
        targets = self.host_targets()
        mountable = {"model"} | path_names
        seen: list[str] = []
        for art in self.artifacts:
            if art.name in seen:
                raise ProfileError(f"artifact {art.name} is declared twice")
            plan = art.plan
            for needed in plan.needs:
                if needed not in seen:
                    raise ProfileError(
                        f"artifact {art.name} needs {needed}, which is not declared before it"
                    )
            self._validate_check(art.check, targets, path_names)
            for mount in plan.mounts:
                if mount.host not in mountable:
                    raise ProfileError(f"artifact {art.name} mounts unknown path {mount.host}")
            if isinstance(plan, ScriptPlan):
                for _, field_name, _, _ in Formatter().parse(plan.script):
                    if field_name and field_name not in self.script_fields():
                        raise ProfileError(
                            f"artifact {art.name} script uses unknown field {{{field_name}}}"
                        )
            else:
                self._validate_check(plan.done, targets, path_names)
            seen.append(art.name)

    def _validate_check(self, check: Check, targets: frozenset[str], path_names: set[str]) -> None:
        if isinstance(check, Executables):
            for program in (*check.rank0, *check.others):
                if program.base is None or program.base not in path_names:
                    raise ProfileError("an executable check needs a program under a path key")
            return
        if check.target not in targets:
            raise ProfileError(f"check on unknown target {check.target}")
        if isinstance(check, SizeManifest):
            for name, size in check.files:
                if not name or any(c.isspace() for c in name) or size < 0:
                    raise ProfileError(f"bad manifest entry {name!r}")
        if isinstance(check, FileSha256) and len(check.sha256) != 64:
            raise ProfileError("a SHA-256 is 64 hex characters")
