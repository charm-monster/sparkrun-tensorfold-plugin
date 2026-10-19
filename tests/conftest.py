from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from sparkrun_tensorfold_profile_hello_engine import HELLO_ENGINE

import sparkrun_tensorfold_plugin
from sparkrun_tensorfold_plugin.profiles import register_profile

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE = EXAMPLES / "hello-engine.yaml"
PROFILE_PLUGIN_DIR = EXAMPLES / "profile-plugin"


@pytest.fixture(scope="session", autouse=True)
def plugin_registered() -> None:
    """Register the recipe block and the example profile exactly as sparkrun does at startup."""
    sparkrun_tensorfold_plugin.register(None)
    register_profile(HELLO_ENGINE)


@pytest.fixture
def example_data() -> dict[str, Any]:
    """The shipped example recipe as plain data, fresh for each test to mutate."""
    return copy.deepcopy(yaml.safe_load(EXAMPLE.read_text()))
