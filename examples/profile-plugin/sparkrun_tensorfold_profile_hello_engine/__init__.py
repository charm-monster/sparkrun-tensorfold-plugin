"""sparkrun plugin that registers the ``hello-engine`` example profile.

A profile package is a sparkrun plugin of its own: sparkrun discovers it like any other plugin and
calls ``register`` once, and ``register`` adds the profile to the tensorfold plugin's registry. The
tensorfold plugin must be enabled alongside it. Name a profile package
``sparkrun_tensorfold_profile_<profile name with underscores>``.
"""

from __future__ import annotations

from typing import Any

from sparkrun_tensorfold_plugin.profiles import register_profile
from sparkrun_tensorfold_profile_hello_engine.profile import HELLO_ENGINE

SPARKRUN_PLUGIN_API_VERSION = 1

__all__ = ["HELLO_ENGINE", "SPARKRUN_PLUGIN_API_VERSION", "register"]


def register(v: Any) -> None:  # noqa: ARG001
    """sparkrun calls this once per process after it has discovered the module."""
    register_profile(HELLO_ENGINE)
