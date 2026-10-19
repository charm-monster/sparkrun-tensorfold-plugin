# Release-trigger model

release-please reads `changelog-sections` in `release-please-config.json`. A commit whose type is
listed there is visible: it appears in the changelog and opens or updates a release PR. Any other
type is hidden and lands silently. The decision uses the commit subject only, never the diff.

| Visible (release-triggering) | Hidden (no release) |
|---|---|
| `feat`, `fix`, `style`, `build`, `refactor`, `ci` | `docs`, `test`, `perf`, `chore`, `revert` |

A commit that touches only documentation or only tests must use a hidden type. Updates to
`CLAUDE.md` or `.claude/` are `chore:`. A bump of a runtime dependency is `build:` or `fix:`.
