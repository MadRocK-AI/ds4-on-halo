#!/usr/bin/env python3
"""Install and launch the pinned single-device Strix Halo engine on Linux."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import halo

ROOT = Path(__file__).resolve().parents[1]
SDK_KEYS = ("ROCM_PATH", "HIP_PATH", "HIP_CLANG_PATH", "HIPCC_COMPILE_FLAGS_APPEND",
            "HIPCC_LINK_FLAGS_APPEND", "ROCM_LDLIBS", "LD_LIBRARY_PATH")
MODEL = {"bytes": 86720111488,
         "sha256": "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"}
MANAGED = "# Managed by ds4-on-halo"
BINARIES = ("ds4", "ds4-server", "ds4-bench")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".halo-", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def prefix_default():
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "ds4-halo"


def clean_environment(sdk=None):
    require(not os.environ.get("LD_PRELOAD"), "Unset LD_PRELOAD before installing or launching DS4 Halo")
    env = {k: v for k, v in os.environ.items() if not k.startswith("DS4_")}
    env.update(sdk or {})
    return env


def sdk_settings(path):
    settings = {k: os.environ[k] for k in SDK_KEYS if k in os.environ}
    if path:
        value = read_json(path)
        require(isinstance(value, dict) and set(value) <= set(SDK_KEYS),
                "SDK environment file accepts only the documented ROCm environment names")
        require(all(isinstance(v, str) and "\0" not in v for v in value.values()),
                "SDK environment values must be strings without NUL bytes")
        settings.update(value)
    return settings


def hipcc_path(explicit):
    selected = explicit or shutil.which("hipcc")
    if not selected and Path("/opt/rocm/bin/hipcc").is_file():
        selected = "/opt/rocm/bin/hipcc"
    require(selected, "hipcc was not found; pass --hipcc /path/to/rocm/bin/hipcc")
    path = Path(selected).resolve(strict=True)
    require(path.is_file() and os.access(path, os.X_OK), "hipcc must be an existing executable")
    return path


def capture(argv, env=None, timeout=60):
    result = subprocess.run([str(x) for x in argv], env=env, capture_output=True,
                            text=True, timeout=timeout)
    require(result.returncode == 0, f"Command failed: {shlex.join([str(x) for x in argv])}\n{result.stderr.strip()}")
    return result.stdout


def hardware_check(env):
    rocminfo = shutil.which("rocminfo", path=env.get("PATH"))
    if not rocminfo and env.get("ROCM_PATH"):
        candidate = Path(env["ROCM_PATH"]) / "bin/rocminfo"
        if candidate.is_file():
            rocminfo = candidate
    require(rocminfo, "rocminfo is required to identify the GPU; install it with your ROCm SDK")
    agents = set(re.findall(r"\bName:\s*(gfx[0-9a-z]+)\b", capture([rocminfo], env)))
    require(agents == {"gfx1151"}, f"Expected a single gfx1151 GPU; rocminfo reports {sorted(agents)}")
    mem = re.search(r"^MemTotal:\s+(\d+)\s+kB", Path("/proc/meminfo").read_text(), re.M)
    require(mem is not None, "Cannot read Linux system memory")
    gib = int(mem.group(1)) / (1024 * 1024)
    require(gib >= 112, f"This model/setup requires a 128 GB Halo; Linux reports {gib:.2f} GiB total")
    require(os.access("/dev/kfd", os.R_OK | os.W_OK), "Current user cannot access /dev/kfd; check ROCm device permissions")
    return {"gpu": "gfx1151", "memory_gib": round(gib, 2)}


def sdk_probe(hipcc, env):
    # Compile/link only: the probe binary is never executed and loads no model.
    source = """#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>
#include <hipcub/hipcub.hpp>
#include <rocprim/rocprim.hpp>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma-version.hpp>
static_assert(ROCWMMA_VERSION_MAJOR == 2 && ROCWMMA_VERSION_MINOR == 2 && ROCWMMA_VERSION_PATCH == 1,
              "DS4 Halo requires the recorded rocWMMA 2.2.1 headers");
int main() { int n=0; return hipGetDeviceCount(&n); }
"""
    with tempfile.TemporaryDirectory(prefix="halo-sdk-") as directory:
        path = Path(directory)
        (path / "probe.cpp").write_text(source, encoding="utf-8")
        flags = shlex.split(env.get("ROCM_LDLIBS", "-lhipblas -lhipblaslt -lrocblas"))
        capture([hipcc, "-x", "hip", "-std=c++20", "--offload-arch=gfx1151",
                 path / "probe.cpp", "-o", path / "probe", *flags], env, timeout=180)
        output = capture(["ldd", "-r", path / "probe"], env)
        require("not found" not in output and "undefined symbol" not in output,
                "ROCm runtime libraries could not be resolved; check your SDK/library paths")
    return {"status": "PASS_COMPILE_LINK", "hipcc": str(hipcc),
            "hipcc_sha256": halo.digest(hipcc), "version": capture([hipcc, "--version"], env).strip()}


def preflight(backend, build_only, hipcc, sdk, prefix):
    require(platform.system() == "Linux" and platform.machine() == "x86_64",
            "DS4 Halo setup requires x86_64 Linux; use a Linux WSL shell for build-only checks")
    require(sys.version_info >= (3, 11), "Python 3.11 or later is required")
    require(backend != "cpu" or build_only, "CPU is supported only with --build-only; this installer never launches CPU model inference")
    for tool in ("git", "make", "cc", "ldd"):
        require(shutil.which(tool), f"Missing prerequisite: {tool}; install it before continuing")
    env = clean_environment(sdk)
    existing = prefix
    while not existing.exists():
        existing = existing.parent
    require(shutil.disk_usage(existing).free >= 4 * (1 << 30), "At least 4 GiB free space is required for the source/build")
    result = {"platform": platform.platform(), "scope": "BUILD_ONLY" if build_only else "HALO_HOST_AND_BUILD"}
    if backend == "rocm":
        result["hardware"] = {"status": "SKIPPED_BUILD_ONLY"} if build_only else hardware_check(env)
        result["sdk"] = sdk_probe(hipcc, env)
    return result


def model_record(path):
    model = path.resolve(strict=True)
    require(model.is_file() and model.stat().st_size == MODEL["bytes"],
            "Model must be the supported DeepSeek V4 Flash 0731 IQ2 GGUF (86,720,111,488 bytes)")
    with model.open("rb") as stream:
        require(stream.read(4) == b"GGUF", "Model is not a GGUF file")
    print("Verifying model SHA256; this reads the model once.", flush=True)
    require(halo.digest(model) == MODEL["sha256"], "Model SHA256 differs from the supported 0731 artifact")
    stat = model.stat()
    return {"path": str(model), "bytes": stat.st_size, "sha256": MODEL["sha256"], "mtime_ns": stat.st_mtime_ns}


def validate_model(record):
    require(record, "No model configured: reinstall with --model /path/to/the/supported.gguf")
    path = Path(record["path"])
    stat = path.stat()
    require(stat.st_size == record["bytes"], "Configured model size changed; reinstall to verify it")
    if stat.st_mtime_ns != record["mtime_ns"]:
        require(halo.digest(path) == record["sha256"], "Configured model content changed")
    return path


@contextmanager
def installation_lock(prefix):
    import fcntl
    prefix.mkdir(parents=True, exist_ok=True)
    with (prefix / ".install.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another install/update is in progress for this prefix") from None
        yield


def package_identity():
    files = [ROOT / "VERSION", ROOT / "engine.json", *sorted((ROOT / "scripts").glob("*.py"))]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def clone_engine(source, destination):
    if source:
        source = halo.verify_engine(source)
    halo.run(["git", "-c", "core.autocrlf=false", "clone", "--no-hardlinks", "--no-checkout",
              "--single-branch", source or halo.manifest()["repository"], destination])
    halo.run(["git", "-C", destination, "config", "core.autocrlf", "false"])
    halo.run(["git", "-C", destination, "checkout", "--detach", halo.pin()])
    halo.run(["git", "-C", destination, "remote", "remove", "origin"])
    halo.verify_engine(destination)


def launcher_text(prefix):
    return f"""#!/usr/bin/env python3
{MANAGED}
import json, os, sys
from pathlib import Path
try:
    prefix = Path({str(prefix)!r})
    state = json.loads((prefix / 'current.json').read_text(encoding='utf-8'))
    release = (prefix / state['release']).resolve()
    if not release.is_relative_to(prefix / 'versions'):
        raise ValueError('Invalid installed release path')
    script = release / 'companion/scripts/setup.py'
    os.execv(sys.executable, [sys.executable, str(script), '--prefix', str(prefix), *sys.argv[1:]])
except (OSError, ValueError, KeyError) as error:
    sys.exit(str(error))
"""


def install(args):
    prefix = args.prefix.expanduser().resolve()
    sdk = sdk_settings(args.sdk_env)
    hipcc = hipcc_path(args.hipcc) if args.backend == "rocm" else None
    require(args.build_only or args.model, "Pass --model /path/to/supported.gguf, or --build-only for offline compilation")
    require(args.jobs > 0, "--jobs must be positive")
    launcher = args.bin_dir.expanduser().resolve() / "ds4-halo"
    require(not launcher.is_symlink(), "Refusing to replace a symlink at the launcher path")
    require(not launcher.exists() or launcher.read_text(encoding="utf-8") == launcher_text(prefix),
            "Launcher path is occupied by another file/install; choose a different --bin-dir")
    print("Checking host and existing toolchain...", flush=True)
    checks = preflight(args.backend, args.build_only, hipcc, sdk, prefix)
    model = model_record(args.model) if args.model else None
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    require(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.]+)?", version), "Invalid VERSION")
    sdk_id = hashlib.sha256(json.dumps({"environment": sdk, "compiler": checks.get("sdk")},
                                       sort_keys=True).encode()).hexdigest()[:12]
    key = f"{version}-{halo.pin()[:12]}-{package_identity()[:12]}-{sdk_id}-{args.backend}"
    with installation_lock(prefix):
        versions = prefix / "versions"
        versions.mkdir(exist_ok=True)
        release = versions / key
        if not release.exists():
            stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=versions))
            print(f"Preparing pinned source in {stage}; failures leave this directory for inspection.", flush=True)
            clone_engine(args.source, stage / "engine")
            env = clean_environment(sdk)
            command = [sys.executable, ROOT / "scripts/halo.py", "build", "--engine", stage / "engine",
                       "--backend", args.backend, "--jobs", str(args.jobs)]
            if hipcc:
                command += ["--hipcc", hipcc]
            print("Building the engine...", flush=True)
            halo.run(command, env=env)
            for name in BINARIES:
                binary = stage / "engine" / name
                require(binary.is_file() and os.access(binary, os.X_OK), f"Build did not produce {name}")
                output = capture(["ldd", "-r", binary], env)
                require("not found" not in output and "undefined symbol" not in output, f"Unresolved runtime library in {name}")
            companion = stage / "companion"
            companion.mkdir()
            shutil.copytree(ROOT / "scripts", companion / "scripts", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name in ("VERSION", "engine.json", "LICENSE"):
                shutil.copyfile(ROOT / name, companion / name)
            build = {"schema": 1, "version": version, "engine_commit": halo.pin(), "backend": args.backend,
                     "package_identity": package_identity(), "checks": checks, "sdk_environment": sdk,
                     "binaries": {name: halo.digest(stage / "engine" / name) for name in BINARIES}}
            atomic_json(stage / "build.json", build)
            stage.rename(release)
        build = read_json(release / "build.json")
        validate_release(release, build, require_rocm=False)
        config = {"schema": 1, "release": str(release.relative_to(prefix)), "version": version,
                  "model": model, "sdk_environment": sdk, "hipcc": str(hipcc) if hipcc else None,
                  "bin_dir": str(launcher.parent), "build_only": args.build_only, "checks": checks}
        # Validate/write the launcher before the one atomic activation point.
        launcher.parent.mkdir(parents=True, exist_ok=True)
        if not launcher.exists():
            fd, filename = tempfile.mkstemp(prefix=".ds4-halo-", dir=launcher.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(launcher_text(prefix))
            os.chmod(filename, 0o755)
            os.replace(filename, launcher)
        previous = read_json(prefix / "current.json") if (prefix / "current.json").exists() else None
        if previous:
            previous_config = {k: v for k, v in previous.items() if k != "previous"}
            config["previous"] = previous.get("previous") if previous_config == config else previous_config
        atomic_json(prefix / "current.json", config)
    print(f"Installed {version} ({halo.pin()[:12]}). Launcher: {launcher}")
    if args.build_only:
        print("Build-only installation: GPU execution is untested; no model or server was started.")
    else:
        print(f"Start the server with: {launcher} serve")
    if str(launcher.parent) not in os.environ.get("PATH", "").split(os.pathsep):
        print(f"Add {shlex.quote(str(launcher.parent))} to PATH, or use the absolute launcher path.")


def loaded_install(prefix, previous=False):
    prefix = prefix.expanduser().resolve()
    config = read_json(prefix / "current.json")
    if previous:
        config = config.get("previous")
        require(config, "There is no previous installation to roll back to")
    release = (prefix / config["release"]).resolve()
    require(release.is_relative_to(prefix / "versions"), "Invalid installed release path")
    return config, release, read_json(release / "build.json")


def validate_release(release, build, require_rocm=True):
    require(build["engine_commit"] == halo.pin(), "Installed engine pin differs from its companion")
    require(build["package_identity"] == package_identity(), "Installed companion content changed; reinstall")
    require(not require_rocm or build["backend"] == "rocm", "CPU build-only installations cannot launch model inference")
    halo.verify_engine(release / "engine")
    for name, sha in build["binaries"].items():
        require(name in BINARIES and halo.digest(release / "engine" / name) == sha,
                f"Installed {name} differs from the verified build; reinstall")
    require(set(build["binaries"]) == set(BINARIES), "Incomplete installed binary inventory")


def launch(args):
    config, release, build = loaded_install(args.prefix)
    validate_release(release, build)
    model = validate_model(config["model"])
    require(args.context >= 2048 and args.context <= 131072, "Context must be between 2K and the documented 128K range")
    env = clean_environment(config["sdk_environment"])
    require(config["sdk_environment"] == build["sdk_environment"], "Installed SDK settings changed; reinstall with the intended toolchain")
    env["DS4_ROCM_HALO_PREFILL"] = "1"
    if args.command == "serve":
        require(1 <= args.port <= 65535, "Port must be between 1 and 65535")
        command = [release / "engine/ds4-server", "--rocm", "--gpu-devices", "0", "--gpu-vram", "100",
                   "--power", "100", "-m", model, "--ctx", str(args.context), "--prefill-chunk", str(args.chunk),
                   "--host", args.host, "--port", str(args.port)]
    else:
        command = [release / "engine/ds4", "--rocm", "--gpu-devices", "0", "--gpu-vram", "100",
                   "--power", "100", "-m", model, "--ctx", str(args.context), "--prefill-chunk", str(args.chunk)]
        if args.prompt is not None:
            command += ["-p", args.prompt]
    if args.dry_run:
        print(json.dumps({"argv": [str(x) for x in command], "environment": {"DS4_ROCM_HALO_PREFILL": "1"},
                          "scope": "Command preview only; no GPU or model execution"}, indent=2))
        return
    hardware_check(env)
    os.execve(str(command[0]), [str(x) for x in command], env)


def doctor(args):
    sdk = sdk_settings(args.sdk_env)
    explicit = args.hipcc
    if (args.prefix.expanduser() / "current.json").is_file():
        config, _, _ = loaded_install(args.prefix)
        sdk = {**config["sdk_environment"], **sdk}
        explicit = explicit or config["hipcc"]
    hipcc = hipcc_path(explicit) if args.backend == "rocm" else None
    report = preflight(args.backend, args.build_only, hipcc, sdk, args.prefix.expanduser().resolve())
    print(json.dumps(report, indent=2))


def update(args):
    require(args.jobs > 0, "--jobs must be positive")
    config, _, _ = loaded_install(args.prefix)
    clean_environment()
    with tempfile.TemporaryDirectory(prefix="halo-update-") as directory:
        companion = args.companion
        if not companion:
            companion = Path(directory) / "companion"
            repository = halo.manifest()["companion_repository"]
            halo.run(["git", "clone", "--single-branch", "--branch", args.ref, repository, companion])
        companion = companion.resolve(strict=True)
        require((companion / "scripts/setup.py").is_file(), "Updated companion has no setup.py")
        command = [sys.executable, companion / "scripts/setup.py", "--prefix", args.prefix, "install",
                   "--bin-dir", config["bin_dir"], "--jobs", str(args.jobs)]
        if config["hipcc"]:
            command += ["--hipcc", config["hipcc"]]
        else:
            command += ["--backend", "cpu"]
        if config["build_only"]:
            command += ["--build-only"]
        if config["model"]:
            command += ["--model", config["model"]["path"]]
        if args.source:
            command += ["--source", args.source]
        halo.run(command, env=clean_environment(config["sdk_environment"]))


def rollback(args):
    prefix = args.prefix.expanduser().resolve()
    with installation_lock(prefix):
        current = read_json(prefix / "current.json")
        previous = current.get("previous")
        require(previous, "There is no previous installation to roll back to")
        release = (prefix / previous["release"]).resolve()
        require(release.is_relative_to(prefix / "versions") and (release / "companion/scripts/setup.py").is_file(),
                "Previous release is missing; cannot roll back")
        # The previous companion validates its own engine pin and content.
        capture([sys.executable, release / "companion/scripts/setup.py", "--prefix", prefix, "status", "--previous", "--verify"])
        previous["previous"] = {k: v for k, v in current.items() if k != "previous"}
        atomic_json(prefix / "current.json", previous)
    print(f"Restored {previous['version']}. Restart your server to use it.")


def status(args):
    config, release, build = loaded_install(args.prefix, args.previous)
    if args.verify:
        validate_release(release, build, require_rocm=False)
    print(json.dumps({"version": config["version"], "engine_commit": build["engine_commit"],
                      "backend": build["backend"], "release": str(release), "model": config["model"]["path"] if config["model"] else None,
                      "build_only": config["build_only"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=prefix_default(), help="installation data directory")
    parser.add_argument("--version", action="version", version=(ROOT / "VERSION").read_text().strip())
    sub = parser.add_subparsers(dest="command", required=True)
    for name, action in (("install", install), ("doctor", doctor)):
        p = sub.add_parser(name, help="install the pinned engine" if name == "install" else "check host/SDK without inference")
        # Accept --prefix after the subcommand too, as used by install.sh.
        p.add_argument("--prefix", type=Path, default=argparse.SUPPRESS)
        p.add_argument("--hipcc", type=Path)
        p.add_argument("--sdk-env", type=Path, help="JSON file with process-local ROCm paths/flags")
        p.add_argument("--backend", choices=("rocm", "cpu"), default="rocm")
        p.add_argument("--build-only", action="store_true", help="skip hardware check; do not launch inference")
        if name == "install":
            p.add_argument("--source", type=Path, help="existing clean engine checkout at the exact pin; otherwise clone the repository")
            p.add_argument("--model", type=Path, help="existing supported 0731 GGUF; never downloaded by this installer")
            p.add_argument("--bin-dir", type=Path, default=Path.home() / ".local/bin")
            p.add_argument("--jobs", type=int, default=2)
        p.set_defaults(action=action)
    for name in ("serve", "run"):
        p = sub.add_parser(name, help="start the API server" if name == "serve" else "start the CLI")
        p.add_argument("--context", type=int, default=32768)
        p.add_argument("--chunk", choices=(2048, 4096), type=int, default=2048)
        p.add_argument("--dry-run", action="store_true")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
        else:
            p.add_argument("--prompt")
        p.set_defaults(action=launch)
    p = sub.add_parser("update", help="install a new companion pin; keep the previous installation")
    p.add_argument("--companion", type=Path, help="local updated companion checkout instead of downloading one")
    p.add_argument("--source", type=Path)
    p.add_argument("--ref", default="main", help="companion branch or release tag to fetch")
    p.add_argument("--jobs", type=int, default=2)
    p.set_defaults(action=update)
    p = sub.add_parser("rollback", help="reactivate the previous installation; does not stop a running server")
    p.set_defaults(action=rollback)
    p = sub.add_parser("status", help="show the active pin and installation")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--previous", action="store_true", help="inspect the retained previous installation")
    p.set_defaults(action=status)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(str(error))
