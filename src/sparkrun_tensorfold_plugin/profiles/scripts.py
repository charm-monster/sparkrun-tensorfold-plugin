"""Script templates profile authors use to describe how an artifact is built.

Each function returns a bash script as a ``str.format`` template. The named fields (``{model}``,
``{cache_dir}``, ...) are filled with shell-quoted host paths at launch; see
``artifacts/context.py``. Write literal braces doubled. These scripts are plain data in a profile;
the core only runs them.
"""

from __future__ import annotations

import shlex

# A container run as the invoking user has no home directory of its own.
_USER_ENV = (
    "set -euo pipefail",
    "export HOME=/tmp/tfhome PATH=/tmp/tfhome/.local/bin:$PATH",
    'mkdir -p "$HOME"',
)


def hf_download(
    *,
    repo: str,
    revision: str,
    dest: str,
    include: tuple[str, ...] = (),
    repo_subdir: str | None = None,
) -> str:
    """Download a pinned Hugging Face revision into the directory named by field ``dest``.

    The download resumes into ``<dest>.partial`` and moves into place only when complete. An
    existing non-empty destination is never replaced. With ``repo_subdir``, only that folder of the
    repository becomes the destination.
    """
    patterns = "".join(f" --include {shlex.quote(p)}" for p in include)
    source = f'"$STAGE/{repo_subdir}"' if repo_subdir else '"$STAGE"'
    return "\n".join(
        [
            *_USER_ENV,
            "pip install --quiet --no-cache-dir --user 'huggingface_hub[cli]'",
            f"DEST={{{dest}}}",
            'STAGE="$DEST.partial"',
            'mkdir -p "$STAGE"',
            f'hf download {shlex.quote(repo)} --revision {revision} --local-dir "$STAGE"{patterns}',
            f"SRC={source}",
            '[ -d "$SRC" ] || {{ echo "download produced no $SRC" >&2; exit 1; }}',
            'if [ -e "$DEST" ]; then',
            '  rmdir "$DEST" 2>/dev/null ||',
            '    {{ echo "refusing to replace non-empty $DEST" >&2; exit 1; }}',
            "fi",
            'mv "$SRC" "$DEST"',
            'rm -rf "$STAGE"',
            "",
        ]
    )


def fetch_verified(*, url: str, sha256: str, to: str) -> str:
    """Download ``url`` to ``to`` and stop unless it has the given SHA-256."""
    return "\n".join(
        [
            f"curl -fsSL {shlex.quote(url)} -o {to}",
            f'echo "{sha256}  "{to} | sha256sum -c -',
        ]
    )


def user_script(*lines: str) -> str:
    """A script that runs as the invoking user, with a writable HOME."""
    return "\n".join([*_USER_ENV, *lines, ""])
