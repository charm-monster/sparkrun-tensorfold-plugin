# Single source of truth

Every shared concept has exactly one declaration site. Import from it. Do not redeclare it.

| Concept | Declared in |
|---|---|
| Engine flags, env vars, positional layout, mirror rules, world sizes, image digest, docker requirements, artifact pins and build steps | a profile package, never this repository (the fictional example is `examples/profile-plugin`) |
| Profile data model | `profiles/model.py` |
| Recipe block keys and validation | `recipe_block.py` |
| Rendering of head and worker commands | `commands.py` |
| Artifact presence rules | `artifacts/verify.py` |
| Build orchestration | `artifacts/build.py` |

## Audit greps (each must return zero matches)

Run from the repository root before declaring work complete. If a grep matches, fix the source
duplication. Do not suppress the grep.

```sh
# Values specific to one machine or user must not appear anywhere in the repository
grep -rnE --exclude=ssot.md '/home/[a-z]+|/Users/[a-z]+|\b10\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\.' \
  src tests docs examples README.md CLAUDE.md .claude
```

The vocabulary of any real engine, model or fork must not appear anywhere either. This repository
ships no profile, so there is nothing to list here: whoever maintains a profile keeps the word list
for it with that profile, and runs it against a checkout of this repository.
