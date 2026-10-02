#!/usr/bin/env python3
"""Compare complete upstream benchmark payloads; never infer quality from text."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message):
    raise ValueError(message)


def load(root):
    m = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if m.get("status") != "PASS_PROCESS" or not m.get("diagnostic_payloads"):
        fail(f"{root}: completed process and full payload readbacks required")
    if not m.get("model_sha256_verified_this_run"):
        fail(f"{root}: model identity was only declared; run qualification with --verify-model")
    pin = json.loads((Path(__file__).resolve().parents[1] / "engine.json").read_text(encoding="utf-8"))
    expected = pin["upstream_base"] if m["arm"] == "base" else pin["commit"]
    if m["engine_commit"] != expected or m["bench_source_sha256"] != pin["bench_source_sha256"]:
        fail(f"{root}: observation source identity differs from this companion")
    if m["arm"] not in ("base", "native", "halo") or m.get("returncode") != 0:
        fail(f"{root}: invalid process/arm identity")
    if m["ds4_environment"].get("DS4_ROCM_HALO_PREFILL") != str(int(m["arm"] == "halo")):
        fail(f"{root}: arm selector does not match recorded environment")
    cfg = m["config"]
    for key in ("context_start", "context_max", "context_alloc", "step_incr", "gen_tokens"):
        if type(cfg.get(key)) is not int or cfg[key] <= 0:
            fail(f"{root}: invalid positive configuration field {key}")
    if cfg["context_start"] > cfg["context_max"] or cfg["context_alloc"] <= cfg["context_max"] + cfg["gen_tokens"]:
        fail(f"{root}: invalid context bounds")
    frontiers = []
    t = cfg["context_start"]
    while True:
        frontiers.append(t)
        if t == cfg["context_max"]:
            break
        t = min(t + cfg["step_incr"], cfg["context_max"])
    required = set()
    for t in frontiers:
        phases = ["prefill", "decode"] + (["restored"] if t < frontiers[-1] else [])
        for phase in phases:
            for suffix in ["state", "logits.f32", "tokens.i32", "manifest.json"]:
                required.add(f"frontier_{t:06d}.{phase}.{suffix}")
    files = {p.name: p for p in (root / "payloads").iterdir() if p.is_file()}
    if set(files) != required:
        fail(f"{root}: missing/extra payload files: {sorted(required.symmetric_difference(files))[:8]}")
    hashes = {}
    vocabulary = None
    actual_snapshot_restore = len(frontiers) > 1
    for t in frontiers:
        phases = ["prefill", "decode"] + (["restored"] if t < frontiers[-1] else [])
        for phase in phases:
            stem = f"frontier_{t:06d}.{phase}"
            extent = json.loads(files[stem + ".manifest.json"].read_text(encoding="utf-8"))
            tokens = cfg["gen_tokens"] if phase == "decode" else t
            expected_pos = t + cfg["gen_tokens"] if phase == "decode" else t
            for key, expected in {"frontier": t, "phase": phase, "token_count": tokens,
                                  "session_pos": expected_pos, "context_alloc": cfg["context_alloc"]}.items():
                if extent.get(key) != expected:
                    fail(f"{root}: incorrect {key} in {stem}")
            vocab = extent.get("logits_count")
            state = extent.get("state_bytes")
            if type(vocab) is not int or vocab <= 0 or type(state) is not int or state <= 0:
                fail(f"{root}: invalid API readback extents in {stem}")
            if phase == "restored":
                kind = extent.get("restore_kind")
                if kind not in ("snapshot", "replay"):
                    fail(f"{root}: unspecified restoration method in {stem}")
                actual_snapshot_restore &= kind == "snapshot" and extent.get("snapshot_bytes") == state
            elif extent.get("restore_kind") != "none" or extent.get("snapshot_bytes") != 0:
                fail(f"{root}: unexpected restore claim in {stem}")
            if vocabulary is None:
                vocabulary = vocab
            if vocab != vocabulary:
                fail(f"{root}: inconsistent vocabulary size")
            for suffix, count in {"state": state, "logits.f32": vocab * 4, "tokens.i32": tokens * 4}.items():
                name = stem + "." + suffix
                path = files[name]
                if path.stat().st_size != count:
                    fail(f"{path}: readback is incomplete or contains extra data")
                hashes[name] = {"bytes": count, "sha256": digest(path)}
    for t in frontiers[:-1]:
        for suffix in ["state", "logits.f32", "tokens.i32"]:
            if hashes[f"frontier_{t:06d}.prefill.{suffix}"] != hashes[f"frontier_{t:06d}.restored.{suffix}"]:
                fail(f"{root}: restore changed {suffix} at frontier {t}")
    return m, hashes, actual_snapshot_restore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--require-restore", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    a, ah, ar = load(args.left)
    b, bh, br = load(args.right)
    if a["arm"] == b["arm"]:
        fail("Use distinct arms for cross-engine qualification")
    for key in ["config", "prompt_sha256", "system", "kernel", "machine"]:
        if a[key] != b[key]:
            fail(f"Comparison changes {key}")
    def flags(m):
        excluded = {"DS4_ROCM_HALO_PREFILL", "DS4_BENCH_TIMING_CSV", "DS4_BENCH_DUMP_PAYLOAD_DIR"}
        return {k:v for k,v in m["ds4_environment"].items() if k not in excluded}
    if flags(a) != flags(b):
        fail("Comparison changes DS4 runtime flags")
    if args.require_restore and not (ar and br):
        fail("Serialized snapshot restore was not exercised at every intermediate frontier; use an incremental snapshot case")
    differences = [name for name in sorted(ah) if ah[name] != bh[name]]
    result = {"status": "FAIL_BINARY_EQUALITY" if differences else "PASS_BINARY_EQUALITY_ON_TESTED_CASE",
              "arms": [a["arm"], b["arm"]], "engine_commits": [a["engine_commit"], b["engine_commit"]],
              "binary_sha256": [a["binary_sha256"], b["binary_sha256"]],
              "model_sha256": a["config"]["model_sha256"], "prompt_sha256": a["prompt_sha256"],
              "config": a["config"], "restore_exercised": ar and br,
              "differences": differences, "payloads": ah}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            fail("--output must be a new file")
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return bool(differences)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as e:
        sys.exit(str(e))
