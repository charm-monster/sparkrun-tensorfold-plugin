"""sparkrun runtime plugin that launches tensorfold engines from engine profiles."""

from __future__ import annotations

from typing import Any

from sparkrun.plugins import register_recipe_item

from sparkrun_tensorfold_plugin.recipe_block import HANDLER, KEY
from sparkrun_tensorfold_plugin.runtime import TensorfoldPluginRuntime

SPARKRUN_PLUGIN_API_VERSION = 1


__all__ = ["SPARKRUN_PLUGIN_API_VERSION", "TensorfoldPluginRuntime", "register"]


def register(v: Any) -> None:  # noqa: ARG001
    """sparkrun calls this once per process after it has discovered the module."""
    register_recipe_item(KEY, HANDLER, owner=__name__)
