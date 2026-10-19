# Artifacts

An engine usually needs more than a container image: weights, kernels, caches that are derived from
the weights, and the engine's own binaries. A profile lists each as an `Artifact`. The plugin
checks them all before it launches and, if the recipe says `prepare: build`, builds what is missing.

## Checks

A check is a short shell script run on the node. It needs only coreutils on the node.

| Check | Passes when |
|---|---|
| `SizeManifest` | every listed file exists in the directory with exactly that size |
| `Sha256Manifest` | `sha256sum -c` of a manifest file inside the directory passes |
| `FileSha256` | the file has the pinned SHA-256 |
| `RankCacheFooter` | exactly one `<prefix><rank>of<world>-*<suffix>` file exists, it ends with the profile's magic bytes, and no unfinished file is in the directory |
| `Executables` | each program exists and is executable |

Downloads are verified against sizes recorded from the pinned revision, so a launch never contacts
Hugging Face just to check.

**Checks never start the engine.** Some engines rewrite a cache whose key differs from the one they
compute. The key can include the modification times of the weight files, so a perfectly good cache
can look stale to the engine. Presence is therefore decided from the files alone.

## Builds

With `tensorfold.prepare: build`, artifacts are handled in the order the profile lists them. For
each one missing on some node:

* **`ScriptPlan`**: a bash script runs in one container on each node that lacks the artifact, in
  parallel. Downloads resume into `<destination>.partial` and move into place only when complete.
  An existing non-empty destination is never replaced.
* **`CoordinatedPlan`**: one container per rank, all started together, because the engine needs
  every rank alive to produce the artifact. The containers are stopped as soon as every rank's
  check passes.

After a build the artifact is checked again, and a build that did not produce it is an error.

### Safety rules

* An artifact built by a `CoordinatedPlan` is built only if it is missing on **every** rank. If it
  is present on some and missing on others the plugin stops and tells you which. Starting the
  engine for the missing rank would also start it for the others, and it could rewrite their copy.
  Remove the remaining copies to rebuild all ranks together, or copy the missing one from a node
  that has it.
* Jobs are detached containers polled with short commands. Nothing holds an SSH session open for
  hours. Each job has a timeout in its plan.
* Every build container is labelled `sparkrun-tensorfold-plugin.build=1` and is removed whether
  the build succeeds, fails or times out.
* Scripts that write files run as your user, so the files are yours. A `CoordinatedPlan` runs the
  engine as root, as the engine is proven to run.

### `prepare: verify`

Nothing is built. If anything is missing the launch fails with a list of every missing artifact,
the node, and the problem, e.g. `rank_cache on node-2 (/data/cache/rank-cache): found 0 matching
file(s), need exactly one`.

## The example profile's artifacts

`examples/profile-plugin` declares one artifact of each kind, so it shows what a profile can
express:

| Artifact | Check | Built by |
|---|---|---|
| `model` | file sizes at a pinned Hugging Face revision | `hf download` of that revision |
| `assets` | the sizes of the listed files | `hf download` of those files at a pinned revision |
| `kit` | the kit's own `MANIFEST` with `sha256sum -c` | `hf download` of one folder at a pinned revision |
| `lookup_table` | pinned SHA-256 | a pinned, hash-checked generator script, no GPU needed |
| `binaries` | the programs each rank runs are executable | a build of a pinned commit in the profile's image |
| `rank_cache` | one complete file per rank | the build tool at a pinned version, one container per rank, until every rank's file is complete |

Every pin is in the profile. A cold build of a per-rank cache can use the GPUs on every node for a
long time and a lot of disk, so a profile's own documentation should say how much.
