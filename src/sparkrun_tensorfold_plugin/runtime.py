"""The ``tensorfold-plugin`` sparkrun runtime.

Engine-agnostic: every fact about an engine variant comes from the profile the recipe names.
sparkrun's stock native-cluster orchestrator does the launching (ranked containers, serve log,
``status`` / ``logs`` / ``stop``); this class supplies the per-rank commands and the checks that
must pass first.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from sparkrun.core.recipe import Recipe, RecipeError
from sparkrun.orchestration.primitives import build_ssh_kwargs, run_command_on_host
from sparkrun.runtimes.base import BASE_CONSUMED_CONFIG_KEYS, RuntimePlugin

from sparkrun_tensorfold_plugin.artifacts.build import ensure_artifacts
from sparkrun_tensorfold_plugin.commands import RenderError, Topology, render_command
from sparkrun_tensorfold_plugin.containers import docker_options, key_file_ranks, rank_options
from sparkrun_tensorfold_plugin.executors import scoped_launch_options
from sparkrun_tensorfold_plugin.fabric import HostRunner, resolve_fabric_address
from sparkrun_tensorfold_plugin.profiles.model import Profile
from sparkrun_tensorfold_plugin.recipe_block import (
    KEY,
    TensorfoldBlock,
    relational_errors,
)

# Shown in `Serve command:` and in dry runs, where the fabric address has not been read from the
# host. It never reaches a container: a real launch resolves the address before any command runs.
DISPLAY_MASTER_ADDR = "<rank-0-fabric-address>"


@dataclass(frozen=True)
class _Bound:
    """The profile and block of the recipe being launched."""

    profile: Profile
    block: TensorfoldBlock
    model: str


class TensorfoldPluginRuntime(RuntimePlugin):
    runtime_name = "tensorfold-plugin"
    default_image_prefix = ""

    # Set per launch in prepare() / _run_cluster(). sparkrun 0.4 copies the runtime for each
    # launch; 0.3.10 shares one instance, which is safe because a process runs one launch at a time.
    _bound: _Bound | None = None
    _master_addr: str | None = None
    _master_pinned = False

    # --- recipe binding ---------------------------------------------------------------------

    @staticmethod
    def _block(recipe: Recipe) -> TensorfoldBlock:
        block = recipe.plugin_item(KEY)
        if not isinstance(block, TensorfoldBlock):
            raise RecipeError(f"recipe '{recipe.name}' has no '{KEY}' block")
        return block

    def _bind(self, recipe: Recipe, overrides: Mapping[str, Any] | None = None) -> _Bound:
        """Validate the recipe against its profile and remember what this launch needs."""
        block = self._block(recipe)
        problems = relational_errors(
            block,
            model=recipe.model,
            defaults={**recipe.defaults, **(overrides or {})},
            min_nodes=recipe.min_nodes,
            max_nodes=recipe.max_nodes,
            readiness_inference=recipe.readiness.get("inference"),
        )
        if problems:
            raise RecipeError(f"{self.runtime_name}: " + "; ".join(problems))
        bound = _Bound(block.profile, block, recipe.model)
        self._bound = bound
        self.readiness_health_path = block.profile.readiness_path
        return bound

    # --- sparkrun runtime interface ---------------------------------------------------------

    def cluster_strategy(self) -> str:
        return "native"

    def native_rendezvous_port(self, *args: Any, **kwargs: Any) -> int | None:  # noqa: ARG002
        # Rank 0 opens its rendezvous port only after loading the model and accepts exactly one
        # connection; the followers retry on their own. Waiting on the port would time out a
        # healthy launch and a probe could consume the one accept.
        return None

    def resolve_container(
        self,
        recipe: Recipe,
        overrides: dict[str, Any] | None = None,  # noqa: ARG002
        *,
        host_hardware: Any = None,  # noqa: ARG002
    ) -> str:
        return self._block(recipe).profile.image

    def prepare(
        self,
        recipe: Recipe,
        hosts: list[str],  # noqa: ARG002
        config: Any = None,  # noqa: ARG002
        dry_run: bool = False,  # noqa: ARG002
        transfer_mode: str = "auto",  # noqa: ARG002
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self._bind(recipe, overrides)

    def validate_recipe(self, recipe: Recipe) -> list[Any]:
        issues = super().validate_recipe(recipe)
        try:
            block = self._block(recipe)
        except RecipeError as error:
            issues.append(self.recipe_error(str(error)))
            return issues
        issues += [
            self.recipe_error(message)
            for message in relational_errors(
                block,
                model=recipe.model,
                defaults=recipe.defaults,
                min_nodes=recipe.min_nodes,
                max_nodes=recipe.max_nodes,
                readiness_inference=recipe.readiness.get("inference"),
            )
        ]
        if recipe.container:
            issues.append(
                self.recipe_error(
                    f"container is declared by profile {block.profile.name} "
                    f"({block.profile.image}); remove it from the recipe"
                )
            )
        used = {k.name for k in block.profile.serve_keys}
        issues += [
            self.recipe_warning(f"defaults.{key} is not used by profile {block.profile.name}")
            for key in recipe.defaults
            if key not in used and key not in BASE_CONSUMED_CONFIG_KEYS
        ]
        return issues

    # --- commands ---------------------------------------------------------------------------

    @contextmanager
    def master_address(self, address: str | None) -> Iterator[None]:
        """Pin the address rank 0 listens on while per-rank commands are generated.

        ``None`` pins "no address": the profile's commands do not use one.
        """
        self._master_addr, self._master_pinned = address, True
        try:
            yield
        finally:
            self._master_addr, self._master_pinned = None, False

    def _render(
        self,
        recipe: Recipe,
        overrides: Mapping[str, Any],
        rank: int,
        world: int,
        master: str | None,
    ) -> str:
        bound = self._bind(recipe, overrides)
        try:
            return render_command(
                bound.profile,
                bound.block,
                model=bound.model,
                defaults={**recipe.defaults, **overrides},
                topology=Topology(rank, world, master, bound.block.rendezvous_port),
            )
        except RenderError as error:
            raise RecipeError(str(error)) from error

    def generate_command(
        self,
        recipe: Recipe,
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,  # noqa: ARG002
        skip_keys: set[str] | frozenset[str] = frozenset(),  # noqa: ARG002
    ) -> str:
        """The rank 0 command, for display. Real launches use :meth:`generate_node_command`."""
        bound = self._bind(recipe, overrides)
        display = DISPLAY_MASTER_ADDR if bound.profile.uses_master_address() else None
        return self._render(recipe, overrides, 0, num_nodes if is_cluster else 1, display)

    def generate_node_command(
        self,
        recipe: Recipe,
        overrides: dict[str, Any],
        head_ip: str,  # noqa: ARG002
        num_nodes: int,
        node_rank: int,
        init_port: int = 0,  # noqa: ARG002
        skip_keys: set[str] | frozenset[str] = frozenset(),  # noqa: ARG002
        hosts: list[str] | None = None,  # noqa: ARG002
        placement: Any = None,  # noqa: ARG002
    ) -> str:
        # sparkrun's head_ip is a management address and init_port a CLI default; the engine
        # needs rank 0's fabric address and the recipe's own rendezvous port.
        if not self._master_pinned:
            raise RuntimeError("the rank 0 fabric address has not been resolved for this launch")
        return self._render(recipe, overrides, node_rank, num_nodes, self._master_addr)

    # --- launch -----------------------------------------------------------------------------

    def _run_solo(  # type: ignore[override]
        self, host: str, image: str, serve_command: str = "", **kwargs: Any
    ) -> int:
        """One node is a world of one: the profile decides whether it can run that way."""
        return self._run_cluster([host], image, serve_command, **kwargs)

    def _run_cluster(  # type: ignore[override]
        self,
        hosts: list[str],
        image: str,
        serve_command: str = "",
        recipe: Recipe | None = None,
        overrides: dict[str, Any] | None = None,
        *,
        cluster_id: str = "sparkrun0",
        config: Any = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> int:
        if recipe is None:
            raise RecipeError(f"{self.runtime_name} needs a recipe to launch")
        bound = self._bind(recipe, overrides)
        profile, block = bound.profile, bound.block

        # The recipe names host paths that are bind-mounted into containers, so an untrusted
        # recipe may not launch (the same rule sparkrun applies to executor_config volumes).
        if not kwargs.get("trust", False):
            raise RecipeError(
                f"{self.runtime_name} bind-mounts the host paths named by the recipe's "
                f"'{KEY}' block; run it from a local recipe or pass --trust"
            )
        if len(hosts) not in profile.world_sizes:
            raise RecipeError(
                f"profile {profile.name} runs on {sorted(profile.world_sizes)} nodes, "
                f"got {len(hosts)}"
            )
        ssh_kwargs = build_ssh_kwargs(config)

        def run(host: str, command: str) -> Any:
            return run_command_on_host(
                host, command, ssh_kwargs=ssh_kwargs, timeout=900, dry_run=dry_run, quiet=True
            )

        self._require_key_file(run, hosts, profile, block, dry_run)
        master: str | None = None
        if profile.uses_master_address():
            master = (
                DISPLAY_MASTER_ADDR
                if dry_run
                else resolve_fabric_address(run, hosts[0], str(block.network["socket_interface"]))
            )
        ensure_artifacts(
            run, hosts, profile, block, recipe.model, master_addr=master, dry_run=dry_run
        )
        kwargs.pop("init_port", None)
        extra_docker_opts = [
            *(kwargs.pop("extra_docker_opts", None) or []),
            *docker_options(profile, block),
        ]
        previous = self.executor
        self.executor = self._with_rank_options(cluster_id, profile, block, len(hosts))
        try:
            with self.master_address(master):
                return self._run_native_cluster(
                    hosts=hosts,
                    image=image,
                    serve_command=serve_command,
                    recipe=recipe,
                    overrides=overrides,
                    cluster_id=cluster_id,
                    config=config,
                    dry_run=dry_run,
                    init_port=block.rendezvous_port,
                    extra_docker_opts=extra_docker_opts,
                    banner_title=f"{profile.name} Cluster Launcher",
                    port_label="Rendezvous Port",
                    node_label="tensorfold rank",
                    **kwargs,
                )
        finally:
            self.executor = previous

    def _with_rank_options(
        self, cluster_id: str, profile: Profile, block: TensorfoldBlock, world_size: int
    ) -> Any:
        """The active executor, adding to each container the options only its rank needs."""
        executor = self._resolve_executor()
        per_rank = rank_options(profile, block, world_size)
        if not per_rank:
            return executor
        return scoped_launch_options(
            executor,
            {
                executor.node_container_name(cluster_id, rank): opts
                for rank, opts in per_rank.items()
            },
        )

    @staticmethod
    def _require_key_file(
        run: HostRunner, hosts: list[str], profile: Profile, block: TensorfoldBlock, dry_run: bool
    ) -> None:
        """The key file is mounted, never read: only its presence is checked, and only on the
        nodes of the ranks whose command reads it."""
        if block.api_key_file is None or dry_run:
            return
        for rank in key_file_ranks(profile, len(hosts)):
            host = hosts[rank]
            if run(host, f"test -f {shlex.quote(block.api_key_file)}").returncode != 0:
                raise RecipeError(f"{host}: api_key_file {block.api_key_file} does not exist")

    def _stop_cluster(  # type: ignore[override]
        self, hosts: list[str], cluster_id: str, config: Any = None, dry_run: bool = False
    ) -> int:
        return self._stop_native_cluster(hosts, cluster_id, config=config, dry_run=dry_run)
