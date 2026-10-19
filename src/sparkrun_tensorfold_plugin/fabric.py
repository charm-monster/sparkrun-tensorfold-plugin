"""Find rank 0's address on the high-speed fabric.

The engine's rank 0 binds its rendezvous listener to this exact IPv4 address, and the followers
connect to it, so it must be an address that is local to rank 0 on the fabric interface. It is not
the management address sparkrun detects, so it is read from the host itself.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from typing import Protocol


class FabricError(RuntimeError):
    """The fabric address of a host could not be determined unambiguously."""


class HostResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


HostRunner = Callable[[str, str], HostResult]


def parse_fabric_address(output: str, *, interface: str, host: str) -> str:
    """The one global IPv4 address in ``ip -j -4 addr show dev <interface>`` output."""
    try:
        interfaces = json.loads(output)
    except json.JSONDecodeError as error:
        raise FabricError(f"{host}: `ip -j` returned invalid JSON: {error}") from error
    if not isinstance(interfaces, list):
        raise FabricError(f"{host}: `ip -j` did not return a list")
    matches = [i for i in interfaces if isinstance(i, dict) and i.get("ifname") == interface]
    if len(matches) != 1:
        raise FabricError(f"{host}: expected one interface named {interface}, found {len(matches)}")
    addresses = [
        a["local"]
        for a in matches[0].get("addr_info", [])
        if a.get("family") == "inet" and a.get("scope") == "global"
    ]
    if len(addresses) != 1:
        raise FabricError(
            f"{host}: interface {interface} must have exactly one global IPv4 address, "
            f"found {len(addresses)}"
        )
    return str(addresses[0])


def resolve_fabric_address(run: HostRunner, host: str, interface: str) -> str:
    """Ask ``host`` for its address on ``interface``."""
    result = run(host, f"ip -j -4 addr show dev {shlex.quote(interface)}")
    if result.returncode != 0:
        reason = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise FabricError(f"{host}: could not read interface {interface}: {reason}")
    return parse_fabric_address(result.stdout, interface=interface, host=host)
