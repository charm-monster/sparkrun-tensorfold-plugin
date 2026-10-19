# Testing

A test is valid only if it fails when the behavior it claims to verify is broken. Before trusting a
non-trivial test, break the production code, watch it go red for the right reason, then restore.

* Assert observable output: the command string, the file on disk, the raised error. Never assert
  that a mock was called.
* Use real collaborators. Real temp directories, real files, real sparkrun `Recipe` objects, real
  Docker for the build tests. Mock only to force an error edge such as disk full.
* Do not hand-build internal state that the real pipeline would produce. Load recipes from YAML
  files through sparkrun.
* Push each test to the lowest layer that can detect the bug. Do not re-assert the same logic at
  several layers.
* Every test file must pass against both supported sparkrun versions (0.3.10 and 0.4.0).
