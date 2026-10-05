"""Distribution lifecycle tests with small files; no inference or GPU qualification."""
import argparse
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import setup


@unittest.skipUnless(os.name == "posix", "Linux installer lifecycle")
class InstallationLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.prefix = self.root / "data with spaces"
        self.bin = self.root / "bin"
        self.args = argparse.Namespace(prefix=self.prefix, sdk_env=None, hipcc=None,
                                       backend="cpu", model=None, build_only=True,
                                       source=None, bin_dir=self.bin, jobs=2)

    def tearDown(self):
        self.tmp.cleanup()

    def mock_build(self, failure=False):
        stack = ExitStack()
        stack.enter_context(patch.object(setup, "preflight", return_value={"scope": "BUILD_ONLY"}))
        stack.enter_context(patch.object(setup.halo, "verify_engine", side_effect=lambda x: x))
        def clone(source, destination):
            destination.mkdir()
            for name in setup.BINARIES:
                p = destination / name
                p.write_bytes(b"small distribution test binary")
                p.chmod(0o755)
        stack.enter_context(patch.object(setup, "clone_engine", side_effect=clone))
        stack.enter_context(patch.object(setup, "capture", return_value="no missing libraries"))
        run = stack.enter_context(patch.object(setup.halo, "run"))
        if failure:
            run.side_effect = subprocess.CalledProcessError(2, "make")
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack, run

    def test_failed_build_keeps_active_install_and_launcher(self):
        self.prefix.mkdir()
        self.bin.mkdir()
        before = b'{"release":"versions/old","version":"previous"}\n'
        (self.prefix / "current.json").write_bytes(before)
        launcher = self.bin / "ds4-halo"
        launcher.write_text(setup.launcher_text(self.prefix), encoding="utf-8")
        original_launcher = launcher.read_bytes()
        stack, _ = self.mock_build(failure=True)
        with stack, self.assertRaises(subprocess.CalledProcessError):
            setup.install(self.args)
        self.assertEqual((self.prefix / "current.json").read_bytes(), before)
        self.assertEqual(launcher.read_bytes(), original_launcher)

    def test_reinstall_reuses_verified_build(self):
        stack, run = self.mock_build()
        with stack:
            setup.install(self.args)
            config1, release1, _ = setup.loaded_install(self.prefix)
            setup.install(self.args)
            config2, release2, _ = setup.loaded_install(self.prefix)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(release1, release2)
            self.assertEqual(config1["release"], config2["release"])
            self.assertIsNone(config2["previous"])

    def test_tampered_reused_binary_is_refused_without_activation(self):
        stack, _ = self.mock_build()
        with stack:
            setup.install(self.args)
            _, release, _ = setup.loaded_install(self.prefix)
            (release / "engine/ds4-server").write_bytes(b"replacement binary")
            before = (self.prefix / "current.json").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "differs from the verified build"):
                setup.install(self.args)
            self.assertEqual((self.prefix / "current.json").read_bytes(), before)

    def test_unrelated_launcher_is_not_overwritten(self):
        self.bin.mkdir()
        path = self.bin / "ds4-halo"
        path.write_text("unrelated file", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "occupied"):
            setup.install(self.args)
        self.assertEqual(path.read_text(), "unrelated file")
        self.assertFalse(self.prefix.exists())

    def test_concurrent_mutation_is_refused(self):
        with setup.installation_lock(self.prefix):
            with self.assertRaisesRegex(RuntimeError, "in progress"):
                with setup.installation_lock(self.prefix):
                    self.fail("lock acquired twice")

    def test_cpu_install_cannot_launch_model(self):
        stack, _ = self.mock_build()
        with stack:
            setup.install(self.args)
            args = argparse.Namespace(prefix=self.prefix, context=32768, command="serve")
            with self.assertRaisesRegex(RuntimeError, "CPU build-only"):
                setup.launch(args)

    def test_release_cannot_escape_managed_directory(self):
        self.prefix.mkdir()
        setup.atomic_json(self.prefix / "current.json", {"release": "../../outside"})
        with self.assertRaisesRegex(RuntimeError, "Invalid installed release path"):
            setup.loaded_install(self.prefix)

    def test_failed_activation_preserves_previous_pointer(self):
        old = {"release": "versions/old", "version": "previous"}
        self.prefix.mkdir()
        setup.atomic_json(self.prefix / "current.json", old)
        stack, _ = self.mock_build()
        real_atomic = setup.atomic_json
        def fail_activation(path, value):
            if path.name == "current.json":
                raise OSError("simulated disk failure")
            real_atomic(path, value)
        with stack, patch.object(setup, "atomic_json", side_effect=fail_activation):
            with self.assertRaises(OSError):
                setup.install(self.args)
        self.assertEqual(setup.read_json(self.prefix / "current.json"), old)


class LaunchContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / "model ; literal.gguf"
        self.model.write_bytes(b"GGUFsynthetic file")
        stat = self.model.stat()
        self.record = {"path": str(self.model), "bytes": stat.st_size,
                       "mtime_ns": stat.st_mtime_ns, "sha256": hashlib.sha256(self.model.read_bytes()).hexdigest()}

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_size_model_change_is_refused(self):
        self.model.write_bytes(b"GGUFmodified!!file")
        self.assertEqual(self.model.stat().st_size, self.record["bytes"])
        os.utime(self.model, ns=(self.record["mtime_ns"] + 1000000000, self.record["mtime_ns"] + 1000000000))
        with self.assertRaisesRegex(RuntimeError, "content changed"):
            setup.validate_model(self.record)

    def test_actual_exec_uses_literal_argv_and_clears_inherited_selectors(self):
        args = argparse.Namespace(prefix=self.root, command="run", context=32768,
                                  chunk=2048, prompt='literal $(command) ; text', dry_run=False)
        cfg = {"model": self.record, "sdk_environment": {}}
        build = {"sdk_environment": {}}
        with patch.object(setup, "loaded_install", return_value=(cfg, self.root, build)), \
             patch.object(setup, "validate_release"), patch.object(setup, "hardware_check"), \
             patch.dict(os.environ, {"DS4_UNQUALIFIED_SWITCH": "1"}, clear=True), \
             patch.object(setup.os, "execve") as execute:
            setup.launch(args)
        argv = execute.call_args.args[1]
        env = execute.call_args.args[2]
        self.assertIn(str(self.model), argv)
        self.assertEqual(argv[-2:], ["-p", args.prompt])
        self.assertEqual(env["DS4_ROCM_HALO_PREFILL"], "1")
        self.assertNotIn("DS4_UNQUALIFIED_SWITCH", env)

    def test_dry_run_never_checks_or_executes_gpu(self):
        args = argparse.Namespace(prefix=self.root, command="serve", context=131072,
                                  chunk=2048, host="127.0.0.1", port=8000, dry_run=True)
        cfg = {"model": self.record, "sdk_environment": {}}
        with patch.object(setup, "loaded_install", return_value=(cfg, self.root, {"sdk_environment": {}})), \
             patch.object(setup, "validate_release"), patch.object(setup, "hardware_check") as hardware, \
             patch.object(setup.os, "execve") as execute, redirect_stdout(io.StringIO()) as output:
            setup.launch(args)
        command = json.loads(output.getvalue())["argv"]
        self.assertIn("131072", command)
        self.assertEqual(command[-4:], ["--host", "127.0.0.1", "--port", "8000"])
        hardware.assert_not_called()
        execute.assert_not_called()

    def test_library_preload_is_refused(self):
        with patch.dict(os.environ, {"LD_PRELOAD": "/tmp/library.so"}):
            with self.assertRaisesRegex(RuntimeError, "LD_PRELOAD"):
                setup.clean_environment()


if __name__ == "__main__":
    unittest.main()
