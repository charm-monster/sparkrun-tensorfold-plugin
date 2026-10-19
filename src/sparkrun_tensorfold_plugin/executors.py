"""Per-container docker options on top of sparkrun's one shared list.

sparkrun's native launcher gives every container the same ``extra_docker_opts``. Each container is
created by ``generate_launch_script(container_name=...)``, and a ranked container's name is
``Executor.node_container_name(cluster_id, rank)``. :func:`scoped_launch_options` returns an
executor of the same class whose launch scripts add a container's own options, found by that name.
sparkrun 0.4 routes each host through ``for_host``; 0.3.10 has no such method. Both are covered.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def _scoped_launch(executor: Any, options_by_container: Mapping[str, list[str]]) -> Any:
    clone = copy.copy(executor)
    original = executor.generate_launch_script

    def generate_launch_script(**kwargs: Any) -> str:
        own = options_by_container.get(kwargs["container_name"], [])
        shared = kwargs.get("extra_docker_opts") or []
        script: str = original(**{**kwargs, "extra_docker_opts": [*shared, *own]})
        return script

    clone.generate_launch_script = generate_launch_script
    return clone


def scoped_launch_options(executor: Any, options_by_container: Mapping[str, list[str]]) -> Any:
    """``executor`` with ``options_by_container[name]`` added to that container's launch."""
    clone = _scoped_launch(executor, options_by_container)
    if hasattr(executor, "for_host"):
        clone.for_host = lambda host: _scoped_launch(executor.for_host(host), options_by_container)
    return clone
