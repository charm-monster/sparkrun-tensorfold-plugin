# Changelog

## 0.1.1 (2026-10-21)


### Features

* add the tensorfold runtime for sparkrun ([1de33b0](https://github.com/charm-monster/sparkrun-tensorfold-plugin/commit/1de33b0bc9de9b1dbb0c46186d0492441ffbdec0))


### Build System

* update uv-build requirement ([72aea10](https://github.com/charm-monster/sparkrun-tensorfold-plugin/commit/72aea10fd7c44dfa4673fbad5c862dc4f88bb4a5))


### Code Refactoring

* type the runtime so it checks cleanly against both supported sparkrun versions ([98e45cc](https://github.com/charm-monster/sparkrun-tensorfold-plugin/commit/98e45cc09e85ce40c51eaf9f0c4f07d65aa11c9f))


### Continuous Integration

* fetch full history in the lockfile job so amending the release commit keeps its parent ([9a276a3](https://github.com/charm-monster/sparkrun-tensorfold-plugin/commit/9a276a3d8d1c757334ce2995ecaa00ad6073eb8b))
* harden the workflows, add Dependabot with a cooldown, and add a security policy ([fabacd9](https://github.com/charm-monster/sparkrun-tensorfold-plugin/commit/fabacd90cc0a1fda99204520000c0abad67da0ea))
* make the owner the code owner of the workflows and release configuration ([3eb5e97](https://github.com/charm-monster/sparkrun-tensorfold-plugin/commit/3eb5e974e831a98fa2cf071782239965eeff028a))
* move release-please to v5, release 0.1.0 first, and keep uv.lock in step with the release ([b007e47](https://github.com/charm-monster/sparkrun-tensorfold-plugin/commit/b007e47938f37966a7ec512932e412cc733eb543))
* open release PRs per component so release-please creates the release after the merge ([d18fafc](https://github.com/charm-monster/sparkrun-tensorfold-plugin/commit/d18fafcefb28d935c1760f67a4f6dbaf7c577a33))
* open the release PR with a personal access token so its checks run ([ce3ed31](https://github.com/charm-monster/sparkrun-tensorfold-plugin/commit/ce3ed31f12bdfeb2be6d8a9d0ae0abdffdbcb518))
* release 0.1.1 first ([876cccb](https://github.com/charm-monster/sparkrun-tensorfold-plugin/commit/876cccb3304f9be4d56743a5a936bf2f045c4313))
* start releases at 0.1.1 ([6ac887f](https://github.com/charm-monster/sparkrun-tensorfold-plugin/commit/6ac887f49c06537b00788c7d540d9c99ae796e7d))
* type-check against each supported sparkrun version ([04a6628](https://github.com/charm-monster/sparkrun-tensorfold-plugin/commit/04a66284d9d61d0663c9a99024f17fc1e0f736d5))
* update the uv.lock version through release-please instead of a workflow step ([97721fd](https://github.com/charm-monster/sparkrun-tensorfold-plugin/commit/97721fd6dc746c354fbeebe5f422f76b61b94f2a))
