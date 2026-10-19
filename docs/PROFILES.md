# Writing a profile

A profile is the single place one engine variant's contract is written down. Adding support for
another fork or model family means adding a profile; the core does not change, and **this package
ships no profile**: yours lives in your own package, which can be private. The worked example is
[`examples/profile-plugin`](../examples/profile-plugin) (a fictional engine, `hello-engine`, that
touches every part of the contract), with [`examples/hello-engine.yaml`](../examples/hello-engine.yaml)
as its recipe. `tests/fake_profile.py` is the smallest complete profile.

A profile is a frozen `Profile` (see `profiles/model.py`, the contract, with the helpers in
`profiles/scripts.py`). `Profile.validate()` runs when the profile is registered and rejects any
reference that does not resolve, so a typo fails at startup, not at launch.

## Values and where they come from

Commands never contain a literal that varies. They contain a `Ref`, which names a value and where
it is read from:

| Constructor | Source |
|---|---|
| `model()` | the recipe's `model:` directory |
| `serve("name")` | `defaults.name` in the recipe, declared by a `ServeKey` |
| `path("name")` | `tensorfold.paths.name`, declared by a `PathKey` |
| `derived("name")` | a fixed name inside a path key's directory, declared by a `DerivedPath` |
| `network("name")` | `tensorfold.network.name`: `rdma_devices`, `socket_interface`, `gid_index` |
| `block("name")` | a scalar of the `tensorfold` block: `api_key_file` |
| `topology("name")` | computed at launch: `rank`, `world_size`, `master_addr`, `master_port` |
| `const(value)` | a literal fixed by the profile |

**Values the engine needs on several ranks are declared once.** If the head's `--context` and the
worker's `--pool` must be equal, both point at `serve("max_model_len")`. They cannot drift apart,
and a recipe sets the value in one place.

The recipe block accepts exactly the `network` and `block` fields the profile's commands use, and
requires all of them.

## Commands

`head` is what rank 0 runs and `worker` what every other rank runs. Each is a `CommandSpec`:

```python
CommandSpec(
    program=Program(base="binaries", relative="native/bin/server"),  # <path key>/<relative>
    subcommand="serve",
    positionals=(model(),),
    flags=(
        Flag("--port", serve("port")),
        Flag("--thinking", serve("thinking"), FlagMode.WHEN_TRUE, optional=True),
    ),
    env=(Env("HELLO_WORLD", topology("world_size")),),
)
```

* A flag with a value is written `--name value`. `WHEN_TRUE` and `WHEN_FALSE` flags are bare and
  appear only when the boolean is true or false.
* `optional=True` means an unset value omits the flag. It is omitted, not defaulted. A required
  flag with no value is an error.
* Everything is shell-quoted.
* If a command or build script uses `topology("master_addr")` / `{master_addr}`, the commands must
  also use `network("socket_interface")`: the address is read from that interface on rank 0. A
  profile that never uses it needs no network block and no lookup.

### Node counts

A profile's `world_sizes` lists the node counts its engine runs on: any set of integers of one or
more (the example says 2 and 4). The core never assumes a count.
Ranks, per-rank commands, build ranks and the placement of the API key file all follow from the
profile: the key file is mounted on, and required at, exactly the nodes of the ranks whose command
uses `block("api_key_file")`. A profile that lists 1 runs as a world of one.

## Container needs

`DockerRequirements` states what the engine needs: GPU, the container user it is proven with, the
IPC mode, capabilities, ulimits and devices. They are passed to every container as extra docker
options. The profile's path keys become bind mounts at the same path on the host and in the
container.

## Constraints

`AtMost("parallel", "streams", 4)` checks that `defaults.parallel` does not exceed
`defaults.streams`, using the engine's own documented default when the recipe leaves `streams`
unset.

## Artifacts

Anything besides the image that the engine needs to run is an `Artifact`: a **check** that decides
whether it is present and complete, and a **plan** that builds it. See [ARTIFACTS.md](ARTIFACTS.md).
Artifacts are listed in dependency order; a plan's `needs` may only name earlier artifacts.

Scripts are `str.format` templates. The fields they may use are the host path of `model`, of every
path key and of every derived path, plus `rank`, `world_size` and `master_addr`, all shell-quoted.
Write literal braces doubled. `profiles/scripts.py` has helpers for the common steps (a pinned
Hugging Face download, a hash-verified fetch).

## Packaging and registering a profile

A profile package is a sparkrun plugin of its own. sparkrun discovers it like any plugin and calls
its `register` hook once; the hook hands the profile to this plugin:

```python
# sparkrun_tensorfold_profile_<name>/__init__.py
from sparkrun_tensorfold_plugin.profiles import register_profile

from sparkrun_tensorfold_profile_<name>.profile import MY_PROFILE

SPARKRUN_PLUGIN_API_VERSION = 1


def register(v):
    register_profile(MY_PROFILE)
```

* Name the module `sparkrun_tensorfold_profile_<profile name, underscores>`. The profile's own
  `name` is what recipes put in `tensorfold.profile`.
* `register_profile` validates the profile and adds it. Registering the same profile twice is fine;
  a different profile under a taken name is an error.
* Both plugins must be enabled: this one and the profile package. A recipe naming a profile that is
  not registered fails with a message saying so.
* On sparkrun 0.3.10, list both directories under `plugins.paths` (see [INSTALL.md](INSTALL.md)). On
  0.4, install the profile package in sparkrun's environment with a `sparkrun.plugins` entry point,
  or list its directory under `plugins.paths` as well.
* In tests, call `register_profile(PROFILE)` directly, as `tests/conftest.py` does for the example.

## Pin everything

Every source a profile builds from should be pinned: image tag or digest, git commits, Hugging Face
revisions, toolchain URLs with their hashes. A profile that tracks a moving branch is not
reproducible.
