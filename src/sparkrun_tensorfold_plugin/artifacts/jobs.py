"""Run a build script in a detached container and observe it.

Jobs are detached and polled with short commands, so a build that takes hours never holds one SSH
session open. Every container carries a label, so leftovers can be found and removed.
"""

from __future__ import annotations

import base64
import posixpath
import shlex
from dataclasses import dataclass

from sparkrun_tensorfold_plugin.fabric import HostRunner
from sparkrun_tensorfold_plugin.paths import host_path
from sparkrun_tensorfold_plugin.profiles.model import Access, DockerRequirements, Mount, Profile
from sparkrun_tensorfold_plugin.recipe_block import TensorfoldBlock

LABEL = "sparkrun-tensorfold-plugin.build"
NAME_PREFIX = "sparkrun-tfp-build"


class JobError(RuntimeError):
    """A container could not be started or observed."""


@dataclass(frozen=True)
class ResolvedMount:
    host: str
    container: str
    access: Access

    @property
    def spec(self) -> str:
        return f"{self.host}:{self.container}:{self.access.value}"


@dataclass(frozen=True)
class JobState:
    status: str  # running, exited, created, missing, ...
    exit_code: int


def job_name(artifact: str, rank: int) -> str:
    return f"{NAME_PREFIX}-{artifact.replace('_', '-')}-{rank}"


def resolve_mounts(
    mounts: tuple[Mount, ...], profile: Profile, block: TensorfoldBlock, model: str
) -> list[ResolvedMount]:
    resolved = []
    for mount in mounts:
        path = host_path(profile, block, model, mount.host)
        if mount.parent:
            path = posixpath.dirname(path.rstrip("/"))
        resolved.append(ResolvedMount(path, mount.container or path, mount.access))
    return resolved


def run_command(
    *,
    name: str,
    image: str,
    script: str,
    mounts: list[ResolvedMount],
    docker: DockerRequirements | None,
    gpu: bool,
    as_user: bool,
) -> str:
    """The ``docker run`` command that starts one job.

    The script travels base64-encoded, so no quoting of its contents can go wrong between here
    and the container.
    """
    args = ["docker", "run", "-d", "--name", name, "--label", f"{LABEL}=1", "--network", "host"]
    if gpu:
        if docker is None:
            raise JobError("a GPU job needs the profile's docker requirements")
        args += ["--ipc", docker.ipc]
        args += [flag for cap in docker.cap_add for flag in ("--cap-add", cap)]
        args += [flag for limit in docker.ulimits for flag in ("--ulimit", limit)]
        args += [flag for device in docker.devices for flag in ("--device", device)]
        if docker.gpus is not None:
            args += ["--gpus", docker.gpus]
    rendered = " ".join(shlex.quote(a) for a in args)
    if as_user:
        rendered += ' --user "$(id -u):$(id -g)"'
    for mount in mounts:
        rendered += f" -v {shlex.quote(mount.spec)}"
    encoded = base64.b64encode(script.encode()).decode()
    inner = shlex.quote(f"echo {encoded} | base64 -d | bash")
    return f"{rendered} {shlex.quote(image)} bash -c {inner}"


def start(
    run: HostRunner,
    host: str,
    *,
    name: str,
    image: str,
    script: str,
    mounts: list[ResolvedMount],
    docker: DockerRequirements | None,
    gpu: bool,
    as_user: bool,
) -> None:
    """Replace any stale job of this name, create the writable host directories, start the job."""
    writable = " ".join(shlex.quote(m.host) for m in mounts if m.access is Access.RW)
    setup = f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1 || true"
    if writable:
        setup += f"\nmkdir -p {writable}"
    prepared = run(host, setup)
    if prepared.returncode != 0:
        raise JobError(f"{host}: could not prepare {name}: {prepared.stderr.strip()}")
    command = run_command(
        name=name,
        image=image,
        script=script,
        mounts=mounts,
        docker=docker,
        gpu=gpu,
        as_user=as_user,
    )
    started = run(host, command)
    if started.returncode != 0:
        raise JobError(f"{host}: could not start {name}: {started.stderr.strip()}")


def state(run: HostRunner, host: str, name: str) -> JobState:
    result = run(
        host,
        f"docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {shlex.quote(name)} "
        "2>/dev/null || echo 'missing 0'",
    )
    words = result.stdout.split()
    if len(words) != 2 or not words[1].lstrip("-").isdigit():
        raise JobError(f"{host}: cannot read the state of {name}: {result.stdout.strip()!r}")
    return JobState(words[0], int(words[1]))


def logs(run: HostRunner, host: str, name: str, lines: int = 40) -> str:
    result = run(host, f"docker logs --tail {lines} {shlex.quote(name)} 2>&1")
    return result.stdout.strip()


def remove(run: HostRunner, host: str, name: str) -> None:
    run(host, f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1 || true")
