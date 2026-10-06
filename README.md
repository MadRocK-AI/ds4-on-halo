# ds4-on-halo

The installer, launcher and verification companion for [the DS4 Strix Halo prefill fork](https://github.com/MadRocK-AI/ds4). The engine adds routed MoE, attention, projection and resident-key indexer paths for single-device AMD `gfx1151`; this repository handles setup, exact source pins, build identity and verification.

**DS4 Strix Halo rc.2: 454.59 token/s prepared prefill (+44.12%), with bitwise-identical complete payloads in the tested cases.** The rc.2 source passed the unchanged performance threshold and actual installer-built numerical and launcher/API acceptance on a second Halo system with IOMMU off. This companion pins the exact cleaned qualified source. [Installed acceptance](distribution-acceptance.json). [Live record](live-smoke.json) · [Validation scope](https://github.com/MadRocK-AI/ds4/blob/main/docs/HALO_RELEASE.md).

## Quick start

On a **single-device Strix Halo gfx1151 with 128 GB memory**, use x86_64 Linux, Python 3.11+, Git, make, `rocminfo` and the [documented existing ROCm SDK](https://github.com/MadRocK-AI/ds4/blob/main/docs/HALO_RELEASE.md#build-dependencies), including rocWMMA 2.2.1. Supply the existing DeepSeek V4 Flash 0731 IQ2 GGUF; the installer verifies its complete SHA256.

```sh
git clone https://github.com/MadRocK-AI/ds4-on-halo.git
cd ds4-on-halo
bash install.sh --model /absolute/path/to/the/model.gguf
~/.local/bin/ds4-halo serve
```

Installation checks the host and SDK, clones the exact engine pin, builds the engine and installs the launcher. It uses per-user directories, downloads source only and leaves packages, drivers and model weights under your control. The foreground server defaults to **127.0.0.1:8000**, **32K context** and **2K chunks**, with the Halo paths enabled.

```sh
ds4-halo doctor
ds4-halo status --verify
ds4-halo serve --context 65536
ds4-halo serve --context 131072 --dry-run
ds4-halo update
ds4-halo rollback
```

Add `~/.local/bin` to PATH or use the absolute launcher path. Updates preserve the previous installation and activate atomically; they never kill a running server. Changed source/binaries are refused. `--dry-run` previews the command without model/GPU execution. [Installation, supported model, SDK overlays and troubleshooting](docs/INSTALLATION.md). [Version history](CHANGELOG.md). The current packaging version is **0.1.0-rc.2**.

## Performance

**Official DS4 published results:** 231.91 token/s at 2K, 292.86 for its warm 4K result and 295.27 in the later 2K→4K interval. [Official Strix Halo report](https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/speed-bench/gfx1151-prefill-results.md). Those are different workloads/SDK conditions. The **315.41** control is our measurement of pinned original DS4 on the same machine and protocol as Halo; **+44.12%** refers to that matched comparison.

**Qualified engine prefill chain, 2026-10-06: 454.59 token/s mean**, versus fresh same-system original DS4 at 315.41 (**+44.12%**), with IOMMU off. Two independent processes per engine each used three pure-prefill warmups and one measured complete 4K request, capacity 4,352, generation disabled. Samples were **454.52 / 454.65**; the unchanged 440 minimum passed. The same-binary IOMMU-on mean was 429.79; the observed before/after increase is **5.77%**, with all complete compared payloads bitwise identical. The cleaned rc.2 installed artifact also matched the complete retained reference payloads and passed its API checks. [Candidate identities, all samples and scope](https://github.com/MadRocK-AI/ds4/blob/main/docs/halo/halo2-qualification.json).

Historical 447.51/449.03 and long-context measurements remain separately documented in [the performance evidence](https://github.com/MadRocK-AI/ds4/blob/main/docs/HALO_PERFORMANCE.md).


## Quality

The latest patched candidate matched fresh original-DS4 token IDs, full FP32 logits and complete states bitwise in the tested 4K and cache-lifetime cases, including 16-token generation checks. The current pin removes only optional runtime diagnostics from that qualified source and has its own successful installed-artifact comparison. Historical checkpoints retain their wider fresh32K/64K/128K, 223 incremental payload and 31 snapshot comparisons; that wider matrix has not been rerun on this candidate. [Verification evidence](https://github.com/MadRocK-AI/ds4/blob/main/docs/HALO_EVIDENCE.md).

## Advanced source setup

The original source/bootstrap tooling remains available for controlled experiments:

```sh
python3 scripts/halo.py bootstrap --source /path/to/clean/pinned/ds4 --destination /path/to/new/engine
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

See the engine's [reproduction protocol](https://github.com/MadRocK-AI/ds4/blob/main/docs/HALO_QUALIFICATION.md) and [release acceptance/build evidence](https://github.com/MadRocK-AI/ds4/blob/main/docs/HALO_RELEASE.md). Save ordinary timing runs separately without `--payloads/--logits`, in alternating arms; precision CSV is supplementary to the original upstream official CSV. Loading is outside prefill, and process success is recorded as `PASS_PROCESS`, never numerical qualification. Historical throughput is scoped to its recorded checkpoints and benchmark conditions; it is not a new measurement of the integrated release executable.

Run `python3 -m unittest discover -s tests -v` for the payload and installer lifecycle checks. [Verification history](verification.json) records bootstrap/build and packaging checks. The [rc.2 installed acceptance](distribution-acceptance.json) verified all six complete numerical payloads and generated eight tokens for each of two real API requests with 14 / 4,214-token prompts and 4K chunks. The older [rc.1 smoke](live-smoke.json) remains separately recorded. Both budgets ended during thinking; broad answer quality and a new long-context/performance matrix were not assessed; complete installed numerical equality was verified separately.

[Published package provenance](release-provenance.json) records the tested package identity, published metadata identity and unchanged executable inputs.
