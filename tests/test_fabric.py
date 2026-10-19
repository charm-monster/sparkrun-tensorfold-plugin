"""Fabric address lookup, against real `ip -j -4 addr show` output captured from a DGX Spark."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from sparkrun_tensorfold_plugin.fabric import (
    FabricError,
    parse_fabric_address,
    resolve_fabric_address,
)

# Shape captured from `ip -j -4 addr show dev <fabric interface>`; the address is from the
# documentation range.
FABRIC = [
    {
        "ifindex": 4,
        "ifname": "enp1s0f1np1",
        "flags": ["BROADCAST", "MULTICAST", "UP", "LOWER_UP"],
        "mtu": 9000,
        "operstate": "UP",
        "addr_info": [
            {
                "family": "inet",
                "local": "192.0.2.2",
                "prefixlen": 24,
                "broadcast": "192.0.2.255",
                "scope": "global",
                "label": "enp1s0f1np1",
            }
        ],
    }
]


def dump(value: object) -> str:
    return json.dumps(value)


def test_the_global_ipv4_address_is_returned() -> None:
    assert parse_fabric_address(dump(FABRIC), interface="enp1s0f1np1", host="a") == "192.0.2.2"


def test_loopback_scope_does_not_count() -> None:
    loopback = [
        {"ifname": "lo", "addr_info": [{"family": "inet", "local": "127.0.0.1", "scope": "host"}]}
    ]
    with pytest.raises(FabricError, match="exactly one global IPv4 address, found 0"):
        parse_fabric_address(dump(loopback), interface="lo", host="a")


def test_two_global_addresses_are_ambiguous() -> None:
    twice = [{"ifname": "x", "addr_info": FABRIC[0]["addr_info"] * 2}]
    with pytest.raises(FabricError, match="exactly one global IPv4 address, found 2"):
        parse_fabric_address(dump(twice), interface="x", host="a")


def test_a_different_interface_is_not_a_match() -> None:
    with pytest.raises(FabricError, match="expected one interface named other, found 0"):
        parse_fabric_address(dump(FABRIC), interface="other", host="a")


@pytest.mark.parametrize("output", ["", "not json", "{}"])
def test_malformed_output_is_an_error(output: str) -> None:
    with pytest.raises(FabricError):
        parse_fabric_address(output, interface="enp1s0f1np1", host="a")


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str


def test_the_host_is_asked_for_the_interface_it_was_given() -> None:
    seen: list[tuple[str, str]] = []

    def run(host: str, command: str) -> Result:
        seen.append((host, command))
        return Result(0, dump(FABRIC), "")

    assert resolve_fabric_address(run, "spark-1", "enp1s0f1np1") == "192.0.2.2"
    assert seen == [("spark-1", "ip -j -4 addr show dev enp1s0f1np1")]


def test_a_failing_host_command_reports_why() -> None:
    def run(host: str, command: str) -> Result:  # noqa: ARG001
        return Result(1, "", 'Device "enp1s0f1np1" does not exist.\n')

    with pytest.raises(FabricError, match=r"spark-1: could not read interface .*does not exist"):
        resolve_fabric_address(run, "spark-1", "enp1s0f1np1")
