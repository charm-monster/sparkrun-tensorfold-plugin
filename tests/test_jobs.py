"""The docker command that starts a build job (no Docker needed)."""

from __future__ import annotations

import base64
import re

from sparkrun_tensorfold_plugin.artifacts.jobs import LABEL, ResolvedMount, job_name, run_command
from sparkrun_tensorfold_plugin.profiles.model import Access, DockerRequirements

DOCKER = DockerRequirements(
    gpus="all",
    user="root",
    ipc="host",
    cap_add=("IPC_LOCK",),
    ulimits=("memlock=-1", "stack=67108864"),
    devices=("/dev/infiniband",),
)


def decoded_script(command: str) -> str:
    encoded = re.search(r"echo (\S+) \| base64 -d \| bash", command)
    assert encoded is not None
    return base64.b64decode(encoded.group(1)).decode()


def test_a_user_script_job_has_only_what_it_needs() -> None:
    command = run_command(
        name="n",
        image="img:1",
        script="echo hi",
        mounts=[ResolvedMount("/h", "/c", Access.RW)],
        docker=None,
        gpu=False,
        as_user=True,
    )
    assert command.startswith(
        f'docker run -d --name n --label {LABEL}=1 --network host --user "$(id -u):$(id -g)" '
        "-v /h:/c:rw img:1 bash -c "
    )
    assert decoded_script(command) == "echo hi"
    assert "--gpus" not in command and "--ipc" not in command


def test_a_gpu_job_gets_every_docker_requirement_of_the_profile() -> None:
    command = run_command(
        name="n", image="img", script="x", mounts=[], docker=DOCKER, gpu=True, as_user=False
    )
    for flag in (
        "--ipc host",
        "--cap-add IPC_LOCK",
        "--ulimit memlock=-1",
        "--ulimit stack=67108864",
        "--device /dev/infiniband",
        "--gpus all",
    ):
        assert flag in command
    assert "--user" not in command


def test_a_profile_without_a_gpu_flag_emits_none() -> None:
    plain = DockerRequirements(gpus=None, user=None, ipc="host", cap_add=(), ulimits=(), devices=())
    command = run_command(
        name="n", image="img", script="x", mounts=[], docker=plain, gpu=True, as_user=False
    )
    assert "--gpus" not in command and "--ipc host" in command


def test_a_script_survives_quotes_and_newlines() -> None:
    script = "echo 'it'\"s\" $HOME\nprintf '%s' \"a b\"\n"
    command = run_command(
        name="n", image="img", script=script, mounts=[], docker=None, gpu=False, as_user=False
    )
    assert decoded_script(command) == script


def test_mount_paths_with_spaces_are_quoted() -> None:
    command = run_command(
        name="n",
        image="img",
        script="x",
        mounts=[ResolvedMount("/a b", "/a b", Access.RO)],
        docker=None,
        gpu=False,
        as_user=False,
    )
    assert "-v '/a b:/a b:ro'" in command


def test_job_names_are_stable_and_per_rank() -> None:
    assert job_name("rank_cache", 1) == "sparkrun-tfp-build-rank-cache-1"
