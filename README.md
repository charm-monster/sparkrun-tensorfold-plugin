# sparkrun-tensorfold-plugin

A [sparkrun](https://github.com/spark-arena/sparkrun) runtime plugin that launches
[tensorfold](https://github.com/ashhart/TensorFold) inference engines across one or more nodes with
one command: `sparkrun run <recipe>`.

tensorfold's multi-node engines do not run one program on every node. Rank 0 serves HTTP and every
other rank runs a different worker program that connects to it, and the engines need a rendezvous
address, per-rank numbering, a matching set of values on every rank, and artifacts such as weight
caches that are not a plain download. This plugin knows how to do all of that. It does not know
about any one model or fork: each engine variant is described by a **profile**, which lives in its
own package, and a recipe holds only values.

| Layer | Holds | Knows about |
|---|---|---|
| This plugin | sparkrun integration, ranks, trust, readiness, artifact lifecycle | profiles, never a specific engine |
| Profile (its own package) | one engine variant's contract: programs, flags, environment, values that must agree across ranks, container needs, how to build each artifact | exactly one engine variant |
| Recipe | paths, network, ports, sampling | nothing about command-line spelling |

**This package ships no profile.** To run a real engine you need a profile package for it, which can
be public or private. [docs/PROFILES.md](docs/PROFILES.md) explains how to write one, and
[`examples/profile-plugin`](examples/profile-plugin) is a complete example for a fictional engine
(`hello-engine`) that the tests use too.

## Requirements

* sparkrun 0.3.10 or any 0.4.x. One package supports both.
* Docker on every node, passwordless SSH between the control machine and the nodes, and whatever
  hardware and fabric your engine's profile asks for (what `sparkrun setup check` verifies).
* A profile package for your engine.

## Install

Pick the line for your sparkrun version. Details and alternatives are in
[docs/INSTALL.md](docs/INSTALL.md).

```sh
# sparkrun 0.4: install from git. Review the source first; sparkrun asks you to trust it.
sparkrun plugins add https://github.com/iamchadarmstrong/sparkrun-tensorfold-plugin --enable --trust

# sparkrun 0.3.10: clone, then point sparkrun at the plugin directory.
git clone https://github.com/iamchadarmstrong/sparkrun-tensorfold-plugin
sparkrun setup features enable core.external_plugins
# add to ~/.config/sparkrun/config.yaml:
#   plugins:
#     paths:
#       - /path/to/sparkrun-tensorfold-plugin/src
#       - /path/to/your/profile-package-directory
```

Then enable your profile package alongside it (see [docs/INSTALL.md](docs/INSTALL.md)). Check that
sparkrun sees both: `sparkrun plugins list` (0.4) or `sparkrun setup plugins list` (0.3).

## Use

1. Copy [`examples/hello-engine.yaml`](examples/hello-engine.yaml) (or your profile's own example)
   and edit the paths, the network and the serve values for your cluster. Every field is explained
   in the file.
2. `sparkrun recipe validate my-recipe.yaml`
3. `sparkrun run my-recipe.yaml --hosts node-1,node-2` (or `--cluster NAME`)

The first launch with `tensorfold.prepare: build` also builds whatever is missing; see below. Use
`sparkrun logs`, `sparkrun status` and `sparkrun stop` as for any other workload.

## The recipe

Serve values live under `defaults:` like any sparkrun recipe. Everything else this plugin reads is
under one key, `tensorfold:`.

| Key | Meaning |
|---|---|
| `tensorfold.profile` | Which engine profile to use. |
| `tensorfold.prepare` | `build`: build any missing artifact, then launch. `verify`: fail with a list of what is missing. Required; there is no implicit choice. |
| `tensorfold.paths` | Host directories the profile names (weights, caches, binaries, ...). Identical on every node and mounted at the same path in the containers. |
| `tensorfold.network` | The RDMA devices, socket interface and GID index of the fabric between the nodes, for profiles that use them. |
| `tensorfold.rendezvous_port` | The TCP port rank 0 listens on for its followers. |
| `tensorfold.api_key_file` | Host path of the file the server reads its API key from. It is mounted read-only on the nodes whose rank's command reads it, and must exist there. **This plugin never reads it.** |
| `readiness.inference: false` | Required. The server needs a key, so sparkrun cannot run its inference probe and polls the unauthenticated health endpoint the profile names. |

Every required value is validated, and a missing one is an error that names the key. Nothing has a
default.

## Artifacts: what is checked and what is built

Before a launch the plugin checks, on every node, that the profile's artifacts (model weights,
kernel kits, lookup tables, engine binaries, per-rank caches, whatever the engine needs) are present
and complete. Checks use only file sizes, hashes and a cache footer, so they work on a node that
has never run the plugin.

With `prepare: build`, missing artifacts are built in detached containers, in dependency order, and
polled with short commands. See [docs/ARTIFACTS.md](docs/ARTIFACTS.md) for the exact rules. Two
matter most:

* **A present artifact is never rebuilt.** Presence is decided from the files alone before any
  engine starts, because an engine may delete a cache whose key differs.
* **A per-rank cache is built only when it is missing on every rank.** If some ranks have it and
  others do not, the plugin stops and says so instead of starting the engine over a copy it could
  rewrite.

Building a per-rank cache can run the engine on every node's GPU at once for a long time. Stop
other GPU workloads first.

## Security

* The API key file is bind-mounted, never read, logged or exported. Recipe exports and sparkrun's
  job metadata contain only its path.
* The recipe names host paths that are bind-mounted into containers, and a profile may ask for
  elevated container settings (a root user, `--ipc host`, device access). Untrusted recipes are
  refused: run a local recipe file or pass `--trust`.
* A profile package is code that sparkrun runs with your permissions, like any plugin. Install only
  profiles you have read.

## Development

```sh
uv sync
uv run ruff check src tests examples && uv run ruff format --check src tests examples && uv run mypy src examples/profile-plugin/sparkrun_tensorfold_profile_hello_engine
uv run pytest -m "not docker"                     # fast; no Docker needed
uv run pytest                                      # also runs real containers and the CLI
uv run --with "sparkrun==0.3.10" pytest            # the same suite against the other version
```

The tests load the fictional example profile the way a real one is loaded. The Docker tests run real
containers of a small public image against a test-only engine profile. They are slow on Docker
Desktop, where every container start takes several seconds.

## License

MIT. See [LICENSE](LICENSE).
