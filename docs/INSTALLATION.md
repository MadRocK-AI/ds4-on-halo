# Install and run DS4 Halo

Target configuration: **x86_64 Linux, single AMD Strix Halo gfx1151, 128 GB unified memory**. Install Python 3.11+, Git, make, a C compiler, `rocminfo` and a complete ROCm SDK first. The validated SDK recipe uses AMD clang 23 / HIP 7.15, rocBLAS 5.6, hipBLASLt, hipCUB, rocPRIM and **rocWMMA 2.2.1**; [exact dependency identities](https://github.com/MadRocK-AI/ds4/blob/main/docs/HALO_RELEASE.md#build-dependencies) are recorded in the engine.

Use the existing **DeepSeek V4 Flash 0731 IQ2** model:

- Filename: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`.
- Size: **86,720,111,488 bytes**.
- SHA256: `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.

The installer reads the model once to verify its full hash. It downloads source when needed; it does not download weights or install packages/drivers. Model loading and inference are not part of installation.

This is release candidate **0.1.0-rc.1**. WSL checks cover installation, update/rollback and ROCm build/link; live model execution through the launcher has not been checked on Halo. [Verification record](../verification.json).

## Setup

```sh
git clone https://github.com/MadRocK-AI/ds4-on-halo.git
cd ds4-on-halo
bash install.sh --model /absolute/path/to/the/model.gguf
```

`hipcc` is detected on PATH or in `/opt/rocm/bin`; override it with `--hipcc /absolute/path/to/hipcc`. The installer checks the GPU, memory, permissions and available build space, compiles/links a small SDK probe without executing it, clones the exact pinned engine, builds it and checks executable/library resolution. No sudo is used.

It installs source and binaries under `${XDG_DATA_HOME:-$HOME/.local/share}/ds4-halo`, and a launcher at `~/.local/bin/ds4-halo`. Add `~/.local/bin` to PATH if necessary, or use that absolute launcher path. Override these destinations with `--prefix` and `--bin-dir`; existing unrelated launcher files are refused.

To use an existing clean engine checkout at the exact pin, add `--source /path/to/ds4`. Existing installed builds are reused only after source and binary checks. Failure before activation preserves the active installation; failed build directories remain under `versions/.pending-*` for inspection.

## Start

```sh
ds4-halo doctor
ds4-halo status --verify
ds4-halo serve
```

The foreground server listens on **127.0.0.1:8000**, with **32K context** and **2K chunks**. The launcher enables `DS4_ROCM_HALO_PREFILL=1` and selects one ROCm device. Ctrl+C stops the foreground process; no system service is installed.

```sh
ds4-halo serve --context 65536 --chunk 2048 --port 8001
ds4-halo serve --context 131072 --dry-run
ds4-halo run --prompt "Explain why the sky is blue."
```

`--dry-run` prints the command without GPU enumeration or model execution. The supported launcher context range is 2K–128K; chunk size is 2K or 4K. The indexer measurements use 2K chunks. A context setting is not a promise of the same speed at every depth. The launcher rejects changed engine source/binaries, incompatible CPU builds and changed model content. Inherited `DS4_*` tuning switches are cleared; use documented launcher options. `LD_PRELOAD` is refused.

The server provides the upstream API documented in [the engine manual](https://github.com/MadRocK-AI/ds4/blob/main/README_UPSTREAM.md); this companion adds no CUDA serving-fork features.

## Update and roll back

```sh
ds4-halo update
ds4-halo status --verify
ds4-halo rollback
```

Update fetches the companion's main branch and installs its engine pin, using the saved model and SDK configuration. Use `update --ref TAG` to select a published companion tag, or `update --companion /path/to/updated/ds4-on-halo --source /path/to/matching/ds4` for local source. An update reads/verifies the model again. It does not reset your source checkout or delete old installed versions.

Activation and the previous-version pointer are written together atomically. Rollback validates the retained previous companion and build before activation. Running servers keep their current executable until stopped; restart them to use the selected version. The installer/updater never kills a running server. If no previous installation exists, rollback reports that explicitly.

## SDK overlays and WSL

For an existing split SDK/header installation, pass `--sdk-env /absolute/path/to/sdk-env.json`. This JSON contains only process-local paths/flags; for example:

```json
{
  "ROCM_PATH": "/path/to/core-sdk",
  "HIP_PATH": "/path/to/core-sdk",
  "HIP_CLANG_PATH": "/path/to/core-sdk/lib/llvm/bin",
  "HIPCC_COMPILE_FLAGS_APPEND": "-isystem /path/to/rocwmma-2.2.1/include -isystem /path/to/math-sdk/include -isystem /path/to/core-sdk/include",
  "ROCM_LDLIBS": "-L/path/to/math-sdk/lib -Wl,-rpath,/path/to/math-sdk/lib -Wl,-rpath,/path/to/core-sdk/lib -lm -pthread -lhipblas -lhipblaslt -lrocblas",
  "LD_LIBRARY_PATH": "/path/to/math-sdk/lib:/path/to/core-sdk/lib:/path/to/core-sdk/lib/llvm/lib:/path/to/core-sdk/lib/rocm_sysdeps/lib"
}
```

Replace every path with an existing location. `HIPCC_LINK_FLAGS_APPEND` is also accepted. The installer saves these settings for launch and updates. `doctor` uses saved settings after installation. SDK probe/build success is not a GPU numerical or speed result.

Without a Halo GPU, WSL can verify the build/install flow explicitly:

```sh
bash install.sh --build-only --hipcc /path/to/hipcc --sdk-env /path/to/sdk-env.json
```

No model is required for build-only setup, and no server is started. For packaging checks without ROCm, use `--build-only --backend cpu`; such installations cannot launch model inference. A normal model-equipped installation can be prepared later by rerunning the installer on the target Halo. [Performance and numerical evidence](https://github.com/MadRocK-AI/ds4/blob/main/docs/HALO_PERFORMANCE.md) retain their original benchmark conditions.

## Troubleshooting

| Reported problem | Next step |
|---|---|
| `hipcc` or SDK headers/libraries missing | Supply the complete existing SDK, check `--hipcc` and the documented `--sdk-env` paths, then run `doctor`. |
| rocWMMA version mismatch | Put the recorded rocWMMA 2.2.1 headers first in the include paths. |
| GPU identity or `/dev/kfd` access refused | Check `rocminfo` and ROCm device permissions on the target Halo. Use `--build-only` only for packaging checks. |
| Model identity mismatch | Use the exact filename, byte count and SHA256 listed above; the installer requires that model. |
| Source or executable identity changed | Reinstall the intended clean pinned source; `status --verify` reports the managed build identity. |
| Build failed before activation | Inspect the retained `versions/.pending-*` directory. The previous active installation remains selected. |
| No previous version for rollback | Install an update first; rollback requires a retained previous installation. |

Include the companion version, `status --verify`, `doctor` output and the compiler error when reporting an installation issue. Remove personal paths before posting logs.
