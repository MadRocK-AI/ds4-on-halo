#!/usr/bin/env python3
"""Local, pinned DS4 setup and a thin driver for upstream ds4-bench."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def fail(message):
    raise SystemExit(message)


def run(argv, **kwargs):
    return subprocess.run([str(x) for x in argv], check=True, **kwargs)


def git(repo, *args):
    return run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest():
    return json.loads((ROOT / "engine.json").read_text(encoding="utf-8"))


def pin():
    return manifest()["commit"]


def verify_engine(path):
    repo = path.resolve(strict=True)
    if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        fail("--engine must be the Git repository root")
    if git(repo, "rev-parse", "HEAD") != pin():
        fail("Engine HEAD does not match engine.json")
    run(["git", "-C", repo, "diff", "--exit-code", "HEAD", "--"],
        stdout=subprocess.DEVNULL)
    if git(repo, "ls-files", "--others", "--exclude-standard"):
        fail("Engine contains untracked files; review or remove them first")
    return repo


def verify_baseline(path):
    repo = path.resolve(strict=True)
    info = manifest()
    if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        fail("Baseline path must be a Git repository root")
    if git(repo, "rev-parse", "HEAD") != info["upstream_base"]:
        fail("Baseline HEAD must be the historical upstream base")
    changed = git(repo, "diff", "--name-only", "HEAD", "--").splitlines()
    if changed != ["ds4_bench.c"] or digest(repo / "ds4_bench.c") != info["bench_source_sha256"]:
        fail("Baseline must contain only the manifest-bound benchmark readback instrumentation")
    if git(repo, "ls-files", "--others", "--exclude-standard"):
        fail("Baseline contains untracked files")
    return repo


def baseline(args):
    source = verify_engine(args.source)
    destination = args.destination.resolve()
    if destination.exists() or destination.is_relative_to(source):
        fail("Baseline destination must be a new directory outside the source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "-c", "core.autocrlf=false", "clone", "--no-hardlinks",
         "--no-checkout", "--single-branch", source, destination])
    run(["git", "-C", destination, "remote", "remove", "origin"])
    run(["git", "-C", destination, "config", "core.autocrlf", "false"])
    run(["git", "-C", destination, "checkout", "--detach", manifest()["upstream_base"]])
    shutil.copyfile(source / "ds4_bench.c", destination / "ds4_bench.c")
    verify_baseline(destination)
    print(destination)


def bootstrap(args):
    source = verify_engine(args.source)
    destination = args.destination.resolve()
    if destination.exists() or destination == source or destination.is_relative_to(source):
        fail("Destination must be a new directory outside the source repository")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "-c", "core.autocrlf=false", "clone", "--no-hardlinks",
         "--no-checkout", "--single-branch", source, destination])
    run(["git", "-C", destination, "remote", "remove", "origin"])
    run(["git", "-C", destination, "config", "core.autocrlf", "false"])
    run(["git", "-C", destination, "checkout", "--detach", pin()])
    verify_engine(destination)
    print(destination)


def build(args):
    engine = verify_baseline(args.engine) if args.baseline else verify_engine(args.engine)
    if os.name != "posix":
        fail("Build in Linux or a Linux WSL shell; native Windows builds are unsupported")
    if not shutil.which("make"):
        fail("Existing make and C compiler are required")
    command = ["make", "-C", engine, "-j", args.jobs, args.backend]
    if args.backend == "rocm":
        hipcc = args.hipcc.resolve(strict=True) if args.hipcc else None
        if hipcc is None or not hipcc.is_file() or not os.access(hipcc, os.X_OK):
            fail("ROCm builds require an explicit existing executable --hipcc")
        command += [f"HIPCC={hipcc}", "ROCM_ARCH=gfx1151"]
    run(command)
    record = {"engine_commit": git(engine, "rev-parse", "HEAD"),
              "baseline_readback_instrumentation": args.baseline, "backend": args.backend,
              "bench_source_sha256": digest(engine / "ds4_bench.c"),
              "binary_sha256": digest(engine / "ds4-bench"), "command": [str(x) for x in command]}
    if args.backend == "rocm":
        record.update(hipcc_sha256=digest(hipcc), hipcc_version=run([hipcc, "--version"],
                      capture_output=True, text=True).stdout.strip())
    (Path(git(engine, "rev-parse", "--absolute-git-dir")) / "halo-build.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("Build finished. GPU correctness and performance remain pending.")


def config(path):
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in ("model", "prompt", "model_sha256", "model_bytes", "hardware", "power_policy"):
        if not cfg.get(key):
            fail(f"Missing config field: {key}")
    for key in ("model", "prompt"):
        v = cfg[key]
        if not isinstance(v,str) or not (PurePosixPath(v).is_absolute() or PureWindowsPath(v).is_absolute()):
            fail(f"{key} must be an explicit absolute path")
    for key in ("hardware", "power_policy"):
        if not isinstance(cfg[key],str):
            fail(f"{key} must describe the actual conditions in text")
    sha = cfg["model_sha256"]
    if not isinstance(sha,str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        fail("model_sha256 must be a lowercase SHA256")
    for key in ("context_start", "context_max", "context_alloc", "step_incr", "prefill_chunk", "gen_tokens"):
        if type(cfg.get(key)) is not int or cfg[key] <= 0:
            fail(f"Expected positive integer: {key}")
    if cfg["context_start"] > cfg["context_max"]:
        fail("context_start exceeds context_max")
    if cfg["context_alloc"] <= cfg["context_max"] + cfg["gen_tokens"]:
        fail("context_alloc must exceed context_max + gen_tokens")
    if type(cfg["model_bytes"]) is not int or cfg["model_bytes"] <= 0:
        fail("model_bytes must be a positive integer")
    declared = cfg.get("environment", {})
    if not isinstance(declared, dict) or any(not k.startswith("DS4_") or not isinstance(v, str)
                                            for k, v in declared.items()):
        fail("[environment] accepts only explicit DS4_ names with string values")
    controlled = {"DS4_ROCM_HALO_PREFILL", "DS4_BENCH_TIMING_CSV", "DS4_BENCH_DUMP_PAYLOAD_DIR"}
    if controlled.intersection(declared):
        fail("Do not override the arm or diagnostic output environment in config")
    return cfg


def bench(args):
    engine = verify_baseline(args.engine) if args.arm == "base" else verify_engine(args.engine)
    cfg = config(args.config)
    model = Path(cfg["model"]).resolve(strict=True)
    prompt = Path(cfg["prompt"]).resolve(strict=True)
    binary = engine / "ds4-bench"
    if not model.is_file() or not prompt.is_file() or not binary.is_file():
        fail("Model, prompt and built ds4-bench must be existing regular files")
    if model.stat().st_size != cfg["model_bytes"]:
        fail("Model size differs from the declared artifact")
    if args.output.exists() or args.output.resolve().is_relative_to(engine):
        fail("--output must be a new directory outside the engine (one directory per observation)")
    if os.environ.get("LD_PRELOAD"):
        fail("Remove LD_PRELOAD before qualification; this engine uses its normal backend")
    command = [str(binary), "--rocm", "--gpu-devices", "0", "--gpu-vram", "100",
               "-m", str(model), "--prompt-file", str(prompt),
               "--power", "100", "--ctx-start", str(cfg["context_start"]), "--ctx-max", str(cfg["context_max"]),
               "--ctx-alloc", str(cfg["context_alloc"]), "--step-incr", str(cfg["step_incr"]),
               "--prefill-chunk", str(cfg["prefill_chunk"]), "--gen-tokens", str(cfg["gen_tokens"]),
               "--csv", str(args.output.resolve() / "official.csv")]
    if args.logits:
        command += ["--dump-frontier-logits-dir", str(args.output.resolve() / "logits")]
    env = {k: v for k, v in os.environ.items() if not k.startswith("DS4_")}
    env.update(cfg.get("environment", {}))
    env["DS4_ROCM_HALO_PREFILL"] = str(int(args.arm == "halo"))
    env["DS4_BENCH_TIMING_CSV"] = str(args.output.resolve() / "timings.csv")
    if args.payloads:
        env["DS4_BENCH_DUMP_PAYLOAD_DIR"] = str(args.output.resolve() / "payloads")
    if args.dry_run:
        print(json.dumps({"argv": command, "environment": {k:v for k,v in env.items() if k.startswith("DS4_")},
                          "model_hash_check": "pending unless --verify-model on the real run"}, indent=2))
        return
    if os.name != "posix":
        fail("GPU observations require a Linux Halo executor")
    build_record = json.loads((Path(git(engine, "rev-parse", "--absolute-git-dir")) /
                               "halo-build.json").read_text(encoding="utf-8"))
    if build_record.get("backend") != "rocm" or build_record.get("engine_commit") != git(engine, "rev-parse", "HEAD") or \
       build_record.get("bench_source_sha256") != digest(engine / "ds4_bench.c") or \
       build_record.get("binary_sha256") != digest(binary):
        fail("Build record differs from the current ROCm executable; run the companion build again")
    hash_verified = False
    if args.verify_model:
        if digest(model) != cfg["model_sha256"]:
            fail("Model SHA256 mismatch")
        hash_verified = True
    args.output.mkdir(parents=True)
    if args.logits:
        (args.output / "logits").mkdir()
    if args.payloads:
        (args.output / "payloads").mkdir()
        env["DS4_BENCH_DUMP_PAYLOAD_DIR"] = str(args.output.resolve() / "payloads")
    metadata = {"engine_commit": git(engine, "rev-parse", "HEAD"),
                "baseline_readback_instrumentation": args.arm == "base",
                "bench_source_sha256": digest(engine / "ds4_bench.c"), "binary_sha256": digest(binary), "argv": command,
                "arm": args.arm, "config": cfg, "model_sha256_verified_this_run": hash_verified,
                "prompt_sha256": digest(prompt), "system": platform.system(),
                "kernel": platform.release(), "machine": platform.machine(),
                "ds4_environment": {k: v for k, v in env.items() if k.startswith("DS4_")},
                "build": build_record, "status": "RUNNING", "diagnostic_logits": args.logits,
                "diagnostic_payloads": args.payloads}
    path = args.output / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output / "stdout.log").open("wb") as out, (args.output / "stderr.log").open("wb") as err:
        result = subprocess.run(command, env=env, stdout=out, stderr=err)
    metadata.update(returncode=result.returncode, status="PASS_PROCESS" if result.returncode == 0 else "FAIL_PROCESS")
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    if result.returncode:
        fail(f"ds4-bench failed ({result.returncode}); inspect {args.output}")
    print("Observation recorded. Process success alone does not qualify numerical correctness.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("bootstrap", help="clone an explicit local pinned source into a new directory")
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    p.set_defaults(action=bootstrap)
    p = sub.add_parser("baseline", help="prepare historical core plus benchmark-only readbacks")
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    p.set_defaults(action=baseline)
    p = sub.add_parser("build", help="use existing compilers; installs nothing")
    p.add_argument("--engine", required=True, type=Path)
    p.add_argument("--backend", choices=("cpu", "rocm"), required=True)
    p.add_argument("--hipcc", type=Path)
    p.add_argument("--baseline", action="store_true", help="build the verified historical control")
    p.add_argument("--jobs", type=int, default=2)
    p.set_defaults(action=build)
    p = sub.add_parser("bench", help="run one fresh upstream ds4-bench observation")
    p.add_argument("--engine", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--arm", choices=("base", "native", "halo"), required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify-model", action="store_true")
    p.add_argument("--logits", action="store_true")
    p.add_argument("--payloads", action="store_true", help="full state/logits/tokens; large local readbacks")
    p.set_defaults(action=bench)
    args = parser.parse_args()
    if getattr(args, "jobs", 1) < 1:
        parser.error("--jobs must be positive")
    args.action(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
