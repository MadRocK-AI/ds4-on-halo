# ds4-on-halo

The pinned setup and verification companion for [the DS4 Strix Halo prefill fork](https://github.com/msala9/ds4). The engine adds routed MoE, attention, projection and resident-key indexer paths for single-device AMD `gfx1151`; this repository handles exact source-pin bootstrap, build identity, benchmark configuration and full bitwise payload comparison.

## Performance

**Up to 449.03 token/s prefill, with bitwise-preserved logits and state in verified cases.** DeepSeek V4 Flash 0731 on AMD Strix Halo (`gfx1151`), 128 GB unified memory.

| Result | Prefill |
|---|---:|
| Halo best recorded mean, resident 4K | **449.03 token/s** |
| Halo controlled resident 4K test | **447.51 token/s** |
| Official DS4 published 4K interval | 295.27 token/s |

**Measured improvement: +43.78%** in the controlled resident 4K comparison against upstream DS4 rebuilt on the same machine. The official published value uses 2K increments; it is context, not the denominator of that controlled gain. [Official DS4 source](https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/speed-bench/gfx1151-prefill-results.md) - [Measurement records](https://github.com/msala9/ds4/blob/main/docs/halo/peak-performance.json).

[Prefill charts from 2K to 128K, source measurements and PNG/SVG/PDF downloads](https://github.com/msala9/ds4/blob/main/docs/HALO_PERFORMANCE.md#prefill-across-context-lengths).

## Quality

**Full FP32 logits, complete serialized state and token IDs are bitwise identical to the reference in the verified cases.** Model weights and quantization are preserved. Coverage includes fresh32K/64K/128K prompts, 223 incremental payload comparisons and 31 snapshot restorations. [Verification evidence](https://github.com/msala9/ds4/blob/main/docs/HALO_EVIDENCE.md).

Both repositories are private under msala9; the planned official home is madrock. The companion consumes an existing Linux ROCm toolchain and a user-supplied model.

## Setup and build

Requirements: Python3.11+, Git, make/C compiler; a complete existing ROCm SDK for GPU builds. The engine preserves upstream history and licenses. The companion installs no packages, downloads no model, changes no machine settings and starts no service.

Clone the private engine repository with your existing GitHub access, then pass its local directory as `--source`. The companion checks out its exact pin.

```sh
git clone https://github.com/msala9/ds4-on-halo.git
cd ds4-on-halo
git clone https://github.com/msala9/ds4.git /path/to/local/release/ds4
python3 scripts/halo.py bootstrap --source /path/to/local/release/ds4 --destination /path/to/new/engine
python3 scripts/halo.py build --engine /path/to/new/engine --backend rocm --hipcc /path/to/existing/hipcc --jobs 2
```

Bootstrap copies independent Git objects, removes the source remote and checks out the exact pin. Build validates the pin and a clean source tree, then records the source and executable hashes inside Git metadata. A real benchmark refuses a changed executable or a missing ROCm build record. For an offline smoke build use `--backend cpu`. Build success alone is not GPU qualification.

## Benchmark configuration and comparisons

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

See the engine's [reproduction protocol](https://github.com/msala9/ds4/blob/main/docs/HALO_QUALIFICATION.md) and [release acceptance/build evidence](https://github.com/msala9/ds4/blob/main/docs/HALO_RELEASE.md). Save ordinary timing runs separately without `--payloads/--logits`, in alternating arms; precision CSV is supplementary to the original upstream official CSV. Loading is outside prefill, and process success is recorded as `PASS_PROCESS`, never numerical qualification. Do not publish historical performance as performance of the new engine.

Run `python3 -m unittest discover -s tests -v` for the parser contract checks. [Offline verification](verification.json) records the real bootstrap/control checks and CPU build; synthetic fixture passes are not model/GPU correctness evidence.
