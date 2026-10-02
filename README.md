# ds4-on-halo (local companion)

A small local setup/configuration/benchmark companion for the exact engine commit in `engine.json`. It consumes a user-supplied model and existing Linux ROCm toolchain, and uses upstream `ds4-bench`. This provisional local name has no public repository URL. Historical tests do not qualify the cleaned integrated engine.

Requirements: Python3.11+, Git, make/C compiler; a complete existing ROCm SDK for GPU builds. The engine preserves upstream history and licenses. The companion installs no packages, downloads no model, changes no machine settings and starts no service.

```sh
python3 scripts/halo.py bootstrap --source /path/to/local/release/ds4 --destination /path/to/new/engine
python3 scripts/halo.py build --engine /path/to/new/engine --backend rocm --hipcc /path/to/existing/hipcc --jobs 2
```

Bootstrap copies independent Git objects, removes the source remote and checks out the exact pin. Build validates the pin and a clean source tree, then records the source and executable hashes inside Git metadata. A real benchmark refuses a changed executable or a missing ROCm build record. For an offline smoke build use `--backend cpu`. Build success alone is not GPU qualification.

Use `python3 scripts/prepare_cases.py --template /path/to/local.toml --destination /path/to/new/cases` to prepare the bounded qualification configurations without running them. Copy `config/example.toml` outside tracked configuration and supply absolute model/prompt paths, expected model hash/bytes, context/chunk and recorded hardware/power policy. The benchmark clears inherited `DS4_*` switches and uses only the explicit `[environment]` table plus its arm/diagnostic output variables. Existing library search paths remain under the executor's control. `LD_PRELOAD` is refused. The runner explicitly passes `--power 100`, which disables the engine's injected duty-cycle sleeps. That upstream option does not configure hardware power limits. The supplied config records the actual machine policy; new source/toolchain observations remain separate from historical results.

```sh
python3 scripts/halo.py bench --engine /path/to/new/engine --config /path/to/case.toml --output /path/to/new/run --arm halo --dry-run --payloads
python3 scripts/halo.py bench --engine /path/to/new/engine --config /path/to/case.toml --output /path/to/new/qual-halo --arm halo --verify-model --payloads
```

`--arm native` disables the added selectors. Optional `--logits` adds the upstream JSON frontier dump. `--payloads` records the complete session payload, every FP32 logit and token IDs at prefill/decode/intermediate restore, outside timing boundaries. Per-phase manifests bind exact API extents and the actual snapshot/replay restoration method. It can consume substantial disk space. Fresh final frontiers do not restore. These private readbacks stay in untracked result directories.

For a historical core control:

```sh
python3 scripts/halo.py baseline --source /path/to/local/release/ds4 --destination /path/to/new/base
python3 scripts/halo.py build --engine /path/to/new/base --baseline --backend rocm --hipcc /path/to/existing/hipcc
python3 scripts/halo.py bench --engine /path/to/new/base --config /path/to/case.toml --output /path/to/new/qual-base --arm base --verify-model --payloads
python3 scripts/compare_payloads.py /path/to/new/qual-base /path/to/new/qual-halo --output /path/to/new/comparison.json
```

The baseline checks out the exact upstream core and copies only manifest-bound `ds4_bench.c` readback/timing instrumentation. Its verified modification is intentional; no Halo backend code is copied. The comparison requires matching source/model/prompt/config identity, complete payload inventory and exact decode count. It checks byte equality and, for intermediate frontiers, prefill-versus-restored equality within both arms. `--require-restore` requires serialized snapshot restoration at every intermediate frontier, and refuses fresh-only or replay cases. A passing comparison describes tested cases only.

See the engine's `docs/HALO_QUALIFICATION.md` for the bounded qualification sequence and remaining gates. Save ordinary timing runs separately without `--payloads/--logits`, in alternating arms; precision CSV is supplementary to the original upstream official CSV. Loading is outside prefill, and process success is recorded as `PASS_PROCESS`, never numerical qualification. Do not publish historical performance as performance of the new engine.

Run `python3 -m unittest discover -s tests -v` for the parser contract checks. [Offline verification](verification.json) records the real bootstrap/control checks and CPU build; synthetic fixture passes are not model/GPU correctness evidence.
