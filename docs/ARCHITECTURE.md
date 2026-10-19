# Architecture

## Layers

```
recipe (values)  ->  profile (one engine variant's contract)  ->  core (sparkrun integration)
```

* **Core** never names an engine, a binary, a flag, an environment variable or a model. The repo's
  audit greps (`.claude/rules/ssot.md`) fail if it does.
* **Profiles** (`profiles/`) are frozen dataclasses. Everything specific to an engine variant is
  data there, and each fact is written once. A value the engine needs on several ranks is one
  `Ref` that the head command and the worker command both point at, so they cannot differ.
* **Recipes** hold values under `defaults:` plus one `tensorfold:` block.

## What happens on `sparkrun run`

1. sparkrun loads the recipe. The `tensorfold` block is parsed and validated against its profile
   (`recipe_block.py`); an invalid block fails the load and names the key.
2. sparkrun checks the recipe with the runtime's `validate_recipe`: the block against `defaults:`,
   the model path, the node range and `readiness.inference` (`recipe_block.relational_errors`).
3. sparkrun places the nodes, distributes the profile's image, and calls the runtime's
   `_run_cluster` (`runtime.py`), which:
   1. refuses an untrusted recipe, before any side effect;
   2. refuses a node count the profile does not run on (any count, one included, if the profile
      lists it);
   3. checks that the API key file exists on the nodes of the ranks whose command reads it (it is
      only tested, never opened);
   4. reads rank 0's address on the fabric interface (`fabric.py`), only if a command or build
      script of the profile uses it;
   5. verifies every artifact on every node and, in `build` mode, builds what is missing
      (`artifacts/`);
   6. hands the profile's container flags to sparkrun's stock native-cluster orchestrator, with the
      options only some ranks need added to those ranks' containers.
4. The orchestrator starts a ranked container per node, executes rank 0's command, and then the
   workers' commands (`commands.py` renders each from the profile).
5. sparkrun polls the profile's health path until the server answers.

## Decisions worth knowing

* **Workers do not wait on the rendezvous port.** `native_rendezvous_port()` returns `None`. Rank 0
  opens its port only after loading the model and accepts exactly one connection, and the
  followers retry on their own, so waiting would time out a healthy launch and a probe could
  consume the one accept.
* **The rendezvous address is rank 0's fabric address**, read from the host, not the management
  address sparkrun detects. The engine binds its listener to that exact address.
* **Container flags travel as `extra_docker_opts`.** sparkrun 0.4 resolves executors before it
  prepares the runtime and 0.3 after, so executor settings cannot carry them on both. Both append
  `extra_docker_opts` after their own flags, and Docker uses the last value of a repeated `--ipc`,
  `--user` or same-named `--ulimit`.
* **Some options are per rank.** sparkrun gives every container the same `extra_docker_opts`, but
  a rank that never reads the API key file should not need a copy of it. `containers.rank_options`
  derives, from which ranks' commands reference the key, the options only those ranks get, and
  `executors.scoped_launch_options` adds them to those containers' launch scripts by container
  name (`Executor.node_container_name`), through `generate_launch_script` and, on sparkrun 0.4,
  `for_host`. Nothing names a rank number or a node count.
* **Readiness uses the profile's unauthenticated health path.** sparkrun's default probe asks
  `/v1/models`, which needs the key. The plugin never reads the key, so it asks for
  `readiness.inference: false` and sparkrun polls the health path.
* **One version home.** The package version lives only in `pyproject.toml`; sparkrun reads it from
  package metadata.

## Layout

```
src/sparkrun_tensorfold_plugin/
  __init__.py       register(); the literal plugin API version sparkrun reads without importing
  runtime.py        TensorfoldPluginRuntime
  recipe_block.py   the tensorfold block: parsing, validation, relational checks, export
  commands.py       render a profile's command for one rank
  containers.py     the docker flags every container needs, and the ones only some ranks need
  executors.py      adds per-container docker flags to sparkrun's executor
  fabric.py         rank 0's address on the fabric interface
  paths.py          resolve a profile's host-path names
  profiles/         the profile contract (model.py), build-script helpers (scripts.py) and the
                    registry; no profile is built in
  artifacts/        verify.py (checks), jobs.py (containers), build.py (orchestration), context.py
```
