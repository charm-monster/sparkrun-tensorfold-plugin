# Installing

The plugin is pure Python with no dependencies beyond sparkrun. It must be importable by the
Python that runs sparkrun, and sparkrun must be told to load it. The two supported versions do that
differently.

Install it wherever you run `sparkrun run`, called the control machine. The nodes need only Docker
and SSH access.

## sparkrun 0.4

### From git (managed install)

```sh
sparkrun plugins add https://github.com/iamchadarmstrong/sparkrun-tensorfold-plugin --enable --trust
```

sparkrun copies the plugin's source, records the commit, and asks for trust because plugin code
runs with your permissions. Review the source first. Without `--trust` it prompts. Update with
`sparkrun plugins update`, remove with `sparkrun plugins remove tensorfold-plugin`.

Use `--ref <tag-or-commit>` to pin a version. `--dry-run` shows what would be installed.

### As a package

If you install sparkrun as a `uv` tool, add this package to the same environment, then enable it:

```sh
uv tool install sparkrun --with "git+https://github.com/iamchadarmstrong/sparkrun-tensorfold-plugin"
sparkrun plugins enable tensorfold-plugin
```

Packages are never loaded until you enable them. A managed install and a package install of the
same module conflict; use one.

## sparkrun 0.3.10

sparkrun 0.3.10 loads plugins from directories you list, and only when its external-plugin feature
is on.

```sh
git clone https://github.com/iamchadarmstrong/sparkrun-tensorfold-plugin
sparkrun setup features enable core.external_plugins
```

Then add the clone's `src` directory to `~/.config/sparkrun/config.yaml`:

```yaml
features:
  core.external_plugins: true
plugins:
  paths:
    - /path/to/sparkrun-tensorfold-plugin/src
```

sparkrun imports every top-level module in each listed directory, so point it at `src`, which holds
only this plugin, not at a directory that holds other Python packages.

## Install a profile

This plugin ships no engine profile. The engine you want to run comes from a profile package (see
[PROFILES.md](PROFILES.md)), which is itself a sparkrun plugin and must be enabled alongside this
one. To try the plugin without a real engine, use the fictional example in
`examples/profile-plugin`.

On sparkrun 0.3.10 (and on 0.4 when the profile is not installed as a package), add the profile
package's directory next to this plugin's `src`:

```yaml
features:
  core.external_plugins: true
plugins:
  paths:
    - /path/to/sparkrun-tensorfold-plugin/src
    - /path/to/your/profile-package-directory
```

## Check

```sh
sparkrun plugins list              # 0.4:   tensorfold-plugin ... enabled  compatible
sparkrun setup plugins list        # 0.3.10: sparkrun_tensorfold_plugin ... external  on
sparkrun recipe validate examples/hello-engine.yaml
sparkrun run examples/hello-engine.yaml --hosts a,b --dry-run
```

A dry run contacts no node. It prints the rank 0 command with the rendezvous address shown as
`<rank-0-fabric-address>`; a real launch reads that address from rank 0's fabric interface.

## Updating sparkrun

`uv tool upgrade sparkrun` and `sparkrun update` may rebuild the tool environment without the
`--with` packages. If the plugin disappears after an update, repeat the install step above.

## Running sparkrun from a cluster node

sparkrun 0.3.10 connects over SSH to every host in the host list, including the machine it runs on,
for its image check and sync. If you run `sparkrun run` from one of the nodes, that node must be
able to SSH to itself without a password. `sparkrun setup ssh` does this: it includes the machine
running sparkrun in its SSH mesh by default. Running from a machine that is not in the cluster
needs nothing extra.

Keep the host order fixed. The first host runs rank 0, and artifacts such as the per-rank cache
belong to a rank. Pass `--hosts` explicitly in the same order every time, rather than relying on a
scheduler that may reorder an idle cluster.

## Notes

* A launch from a macOS machine in sparkrun's default follow mode did not print sparkrun's
  "Endpoint ready" line in testing on 0.3.10, though the workload was serving and `status`, `logs`
  and `stop` worked. sparkrun's own readiness probe succeeds against the same server when run on
  the head node. If you want the readiness report, run sparkrun from a cluster node.
* Ending the follow with Ctrl-C detaches from the logs; it does not stop the workload.
