"""Is an artifact present and complete on a node?

Each check is a short shell script that needs nothing installed on the node beyond coreutils, so
verification works on a node that has never run the plugin. A check answers from the files alone.
It never starts the engine, because an engine started against a stale cache deletes it.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from sparkrun_tensorfold_plugin.commands import program_path
from sparkrun_tensorfold_plugin.fabric import HostRunner
from sparkrun_tensorfold_plugin.paths import host_path
from sparkrun_tensorfold_plugin.profiles.model import (
    Check,
    Executables,
    FileSha256,
    Profile,
    RankCacheFooter,
    Sha256Manifest,
    SizeManifest,
)
from sparkrun_tensorfold_plugin.recipe_block import TensorfoldBlock

MISSING_DIR = "MISSING_DIR"
_MISSING = "MISSING"


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    detail: str


def _q(path: str) -> str:
    return shlex.quote(path)


def check_script(
    check: Check,
    profile: Profile,
    block: TensorfoldBlock,
    model: str,
    *,
    rank: int,
    world_size: int,
) -> str:
    """The shell script that answers ``check`` on one node."""
    match check:
        case Executables(rank0=rank0, others=others):
            needed = rank0 if rank == 0 else others
            words = " ".join(_q(program_path(p, block)) for p in needed)
            return (
                f"for p in {words}; do "
                '[ -x "$p" ] || { echo "not executable: $p"; exit 1; }; done; echo ok'
            )
        case SizeManifest(target=name, files=files):
            target = _q(host_path(profile, block, model, name))
            names = " ".join(_q(file_name) for file_name, _ in files)
            return (
                f"cd {target} 2>/dev/null || {{ echo {MISSING_DIR}; exit 0; }}\n"
                f"for f in {names}; do\n"
                '  if [ -f "$f" ]; then echo "$f $(wc -c < "$f" | tr -d \' \')"; '
                f'else echo "$f {_MISSING}"; fi\n'
                "done"
            )
        case Sha256Manifest(target=name, file=manifest):
            target = _q(host_path(profile, block, model, name))
            return (
                f"cd {target} 2>/dev/null || {{ echo 'directory does not exist'; exit 2; }}\n"
                f"sha256sum -c --quiet {_q(manifest)}"
            )
        case FileSha256(target=name):
            target = _q(host_path(profile, block, model, name))
            return f"[ -f {target} ] || {{ echo {_MISSING}; exit 0; }}\nsha256sum -- {target}"
        case RankCacheFooter(target=name, prefix=prefix, suffix=suffix, magic=magic):
            target = _q(host_path(profile, block, model, name))
            pattern = f"{target}/{prefix}{rank}of{world_size}-*{suffix}"
            return "\n".join(
                [
                    "n=0; found=",
                    f"for f in {pattern}; do",
                    '  [ -f "$f" ] && { n=$((n+1)); found=$f; }',
                    "done",
                    'if [ "$n" -ne 1 ]; then',
                    '  echo "found $n matching file(s), need exactly one"; exit 1',
                    "fi",
                    f"for f in {target}/*.partial*; do",
                    '  [ -e "$f" ] && { echo "unfinished file present: $f"; exit 1; }',
                    "done",
                    f'if [ "$(tail -c {len(magic)} "$found")" != {_q(magic)} ]; then',
                    f'  echo "$found does not end with {magic}"; exit 1',
                    "fi",
                    "echo ok",
                ]
            )


def interpret(check: Check, returncode: int, output: str) -> CheckResult:
    """Turn a check script's exit code and output into an answer."""
    text = output.strip()
    if isinstance(check, SizeManifest):
        return _interpret_sizes(check, text)
    if isinstance(check, FileSha256):
        if text == _MISSING:
            return CheckResult(False, "file does not exist")
        found = text.split()[0] if text else ""
        if found != check.sha256:
            return CheckResult(
                False, f"SHA-256 is {found or 'unreadable'}, expected {check.sha256}"
            )
        return CheckResult(True, "")
    if returncode != 0:
        return CheckResult(False, text.splitlines()[0] if text else f"exit code {returncode}")
    return CheckResult(True, "")


def _interpret_sizes(check: SizeManifest, text: str) -> CheckResult:
    if text == MISSING_DIR:
        return CheckResult(False, "directory does not exist")
    seen: dict[str, str] = {}
    for line in text.splitlines():
        name, _, size = line.rpartition(" ")
        seen[name] = size
    problems: list[str] = []
    for name, expected in check.files:
        got = seen.get(name)
        if got is None or got == _MISSING:
            problems.append(f"{name} is missing")
        elif got != str(expected):
            problems.append(f"{name} is {got} bytes, expected {expected}")
    if not problems:
        return CheckResult(True, "")
    shown = "; ".join(problems[:3])
    more = f" (and {len(problems) - 3} more)" if len(problems) > 3 else ""
    return CheckResult(False, shown + more)


def verify(
    run: HostRunner,
    host: str,
    check: Check,
    profile: Profile,
    block: TensorfoldBlock,
    model: str,
    *,
    rank: int,
    world_size: int,
) -> CheckResult:
    script = check_script(check, profile, block, model, rank=rank, world_size=world_size)
    result = run(host, script)
    return interpret(check, result.returncode, result.stdout + result.stderr)
