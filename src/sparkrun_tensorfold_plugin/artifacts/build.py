"""Make sure every artifact a profile needs exists on every node, building what is missing.

Order and safety rules:

* Artifacts are handled in the order the profile lists them, so an artifact's inputs exist before
  it is built.
* A present artifact is never rebuilt. Presence is decided from the files alone, before any engine
  is started.
* An artifact that must be built by all ranks together (the engine needs every rank alive) is built
  only when it is missing on every rank. If some ranks have it and others do not, the plugin stops:
  starting the engine against the ranks that have it could make the engine rewrite their copy.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from sparkrun_tensorfold_plugin.artifacts import jobs
from sparkrun_tensorfold_plugin.artifacts.context import script_context
from sparkrun_tensorfold_plugin.artifacts.verify import verify
from sparkrun_tensorfold_plugin.commands import Topology, render_spec
from sparkrun_tensorfold_plugin.fabric import HostRunner
from sparkrun_tensorfold_plugin.paths import host_path
from sparkrun_tensorfold_plugin.profiles.model import (
    Artifact,
    CoordinatedPlan,
    Profile,
    ScriptPlan,
)
from sparkrun_tensorfold_plugin.recipe_block import PrepareMode, TensorfoldBlock

logger = logging.getLogger(__name__)


class ArtifactError(RuntimeError):
    """An artifact is missing and was not built, or a build failed."""


@dataclass(frozen=True)
class Gap:
    artifact: str
    host: str
    rank: int
    detail: str


@dataclass(frozen=True)
class Launch:
    """Everything a build needs to know about this launch."""

    run: HostRunner
    hosts: list[str]
    profile: Profile
    block: TensorfoldBlock
    model: str
    master_addr: str | None
    poll_seconds: float
    heartbeat_seconds: float


def find_gaps(launch: Launch, artifact: Artifact) -> list[Gap]:
    """The nodes on which ``artifact`` is missing or incomplete."""

    def check(item: tuple[int, str]) -> Gap | None:
        rank, host = item
        result = verify(
            launch.run,
            host,
            artifact.check,
            launch.profile,
            launch.block,
            launch.model,
            rank=rank,
            world_size=len(launch.hosts),
        )
        return None if result.ok else Gap(artifact.name, host, rank, result.detail)

    with ThreadPoolExecutor(max_workers=len(launch.hosts)) as pool:
        return [gap for gap in pool.map(check, enumerate(launch.hosts)) if gap is not None]


def ensure_artifacts(
    run: HostRunner,
    hosts: list[str],
    profile: Profile,
    block: TensorfoldBlock,
    model: str,
    *,
    master_addr: str | None,
    dry_run: bool,
    poll_seconds: float = 15.0,
    heartbeat_seconds: float = 120.0,
) -> None:
    """Verify every artifact on every node; in ``build`` mode, build what is missing."""
    if dry_run:
        logger.info("Dry run: not verifying or building artifacts")
        return
    launch = Launch(run, hosts, profile, block, model, master_addr, poll_seconds, heartbeat_seconds)
    outstanding: list[Gap] = []
    for artifact in profile.artifacts:
        gaps = find_gaps(launch, artifact)
        if not gaps:
            logger.info("Artifact %s: present on all %d node(s)", artifact.name, len(hosts))
            continue
        if block.prepare is PrepareMode.VERIFY:
            outstanding += gaps
            continue
        _build(launch, artifact, gaps)
        still = find_gaps(launch, artifact)
        if still:
            detail = "; ".join(f"{g.host}: {g.detail}" for g in still)
            raise ArtifactError(
                f"{artifact.name} still fails verification after its build: {detail}"
            )
    if outstanding:
        lines = "\n".join(
            f"  - {g.artifact} on {g.host} ({_where(launch, g.artifact)}): {g.detail}"
            for g in outstanding
        )
        raise ArtifactError(
            "tensorfold.prepare is 'verify' and these artifacts are missing or incomplete:\n"
            f"{lines}\nSet tensorfold.prepare to 'build' to build them."
        )


def _where(launch: Launch, artifact: str) -> str:
    target = next(a for a in launch.profile.artifacts if a.name == artifact).check
    name = getattr(target, "target", None)
    return host_path(launch.profile, launch.block, launch.model, name) if name else "programs"


def _build(launch: Launch, artifact: Artifact, gaps: list[Gap]) -> None:
    plan = artifact.plan
    where = ", ".join(g.host for g in gaps)
    logger.info(
        "Artifact %s: missing on %s; building. This can take a long time.", artifact.name, where
    )
    if isinstance(plan, ScriptPlan):
        _build_scripts(launch, artifact, plan, gaps)
    else:
        _build_together(launch, artifact, plan, gaps)


def _build_scripts(launch: Launch, artifact: Artifact, plan: ScriptPlan, gaps: list[Gap]) -> None:
    started: list[tuple[Gap, str]] = []
    try:
        for gap in gaps:
            fields = script_context(
                launch.profile,
                launch.block,
                launch.model,
                rank=gap.rank,
                world_size=len(launch.hosts),
                master_addr=launch.master_addr,
            )
            name = jobs.job_name(artifact.name, gap.rank)
            jobs.start(
                launch.run,
                gap.host,
                name=name,
                image=plan.image,
                script=plan.script.format(**fields),
                mounts=jobs.resolve_mounts(plan.mounts, launch.profile, launch.block, launch.model),
                docker=launch.profile.docker if plan.gpu else None,
                gpu=plan.gpu,
                as_user=plan.as_user,
            )
            started.append((gap, name))
        _wait_for_exit(launch, artifact.name, started, plan.timeout_s)
    finally:
        for gap, name in started:
            jobs.remove(launch.run, gap.host, name)


def _wait_for_exit(
    launch: Launch, artifact: str, started: list[tuple[Gap, str]], timeout_s: int
) -> None:
    deadline = time.monotonic() + timeout_s
    next_heartbeat = time.monotonic() + launch.heartbeat_seconds
    pending = list(started)
    while pending:
        for item in list(pending):
            gap, name = item
            state = jobs.state(launch.run, gap.host, name)
            if state.status == "missing":
                raise ArtifactError(f"{artifact}: the build container on {gap.host} disappeared")
            if state.status == "exited":
                if state.exit_code != 0:
                    tail = jobs.logs(launch.run, gap.host, name)
                    raise ArtifactError(
                        f"{artifact} failed on {gap.host} (exit code {state.exit_code}):\n{tail}"
                    )
                pending.remove(item)
        if not pending:
            return
        now = time.monotonic()
        if now > deadline:
            raise ArtifactError(f"{artifact} did not finish within {timeout_s} seconds")
        if now >= next_heartbeat:
            hosts = ", ".join(g.host for g, _ in pending)
            logger.info("Still building %s on %s", artifact, hosts)
            next_heartbeat = now + launch.heartbeat_seconds
        time.sleep(launch.poll_seconds)


def _build_together(
    launch: Launch, artifact: Artifact, plan: CoordinatedPlan, gaps: list[Gap]
) -> None:
    if len(gaps) != len(launch.hosts):
        missing = {g.host for g in gaps}
        present = ", ".join(h for h in launch.hosts if h not in missing)
        raise ArtifactError(
            f"{artifact.name} is present on {present} but missing on {', '.join(sorted(missing))}. "
            "The engine rewrites an existing copy when its key differs, so it will not be started "
            "to fill in the missing one. Remove the remaining copies yourself to rebuild all ranks "
            "together, or copy the missing one from a node that has it."
        )
    logger.warning(
        "Building %s needs every rank's engine running at once. It uses the GPUs on all %d "
        "node(s), so stop other GPU workloads first, and it can take hours.",
        artifact.name,
        len(launch.hosts),
    )
    mounts = jobs.resolve_mounts(plan.mounts, launch.profile, launch.block, launch.model)
    started: list[tuple[Gap, str]] = []
    try:
        for gap in gaps:
            topology = Topology(
                gap.rank, len(launch.hosts), launch.master_addr, launch.block.rendezvous_port
            )
            command = render_spec(
                plan.command,
                launch.profile,
                launch.block,
                model=launch.model,
                defaults={},
                topology=topology,
            )
            name = jobs.job_name(artifact.name, gap.rank)
            jobs.start(
                launch.run,
                gap.host,
                name=name,
                image=plan.image,
                script=f"set -euo pipefail\n{plan.setup}\nexec {command}\n",
                mounts=mounts,
                docker=launch.profile.docker,
                gpu=True,
                as_user=False,
            )
            started.append((gap, name))
        _wait_for_done(launch, artifact, plan, started)
    finally:
        for gap, name in started:
            jobs.remove(launch.run, gap.host, name)


def _wait_for_done(
    launch: Launch, artifact: Artifact, plan: CoordinatedPlan, started: list[tuple[Gap, str]]
) -> None:
    deadline = time.monotonic() + plan.timeout_s
    next_heartbeat = time.monotonic() + launch.heartbeat_seconds
    while True:
        for gap, name in started:
            state = jobs.state(launch.run, gap.host, name)
            if state.status in ("exited", "missing"):
                tail = jobs.logs(launch.run, gap.host, name)
                raise ArtifactError(
                    f"{artifact.name}: the engine on {gap.host} stopped (exit code "
                    f"{state.exit_code}) before every rank finished:\n{tail}"
                )
        remaining = [
            g
            for g, _ in started
            if not verify(
                launch.run,
                g.host,
                plan.done,
                launch.profile,
                launch.block,
                launch.model,
                rank=g.rank,
                world_size=len(launch.hosts),
            ).ok
        ]
        if not remaining:
            return
        now = time.monotonic()
        if now > deadline:
            raise ArtifactError(f"{artifact.name} did not finish within {plan.timeout_s} seconds")
        if now >= next_heartbeat:
            names = ", ".join(g.host for g in remaining)
            logger.info("Still building %s; waiting for %s", artifact.name, names)
            next_heartbeat = now + launch.heartbeat_seconds
        time.sleep(launch.poll_seconds)
