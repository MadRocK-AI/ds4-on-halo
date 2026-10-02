"""Parser unit tests with tiny synthetic files; no model/GPU correctness claim."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PIN = json.loads((ROOT / 'engine.json').read_text(encoding='utf-8'))
SCRIPT = ROOT / 'scripts/compare_payloads.py'


class PayloadContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.a = self.fixture('base')
        self.b = self.fixture('halo')

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, arm, incremental=True):
        root = self.root / arm
        root.mkdir()
        cfg = dict(context_start=8, context_max=16 if incremental else 8,
                   context_alloc=64,step_incr=8,gen_tokens=2,model_sha256='0'*64)
        m = dict(status='PASS_PROCESS',returncode=0,diagnostic_payloads=True,
                 model_sha256_verified_this_run=True,arm=arm,config=cfg,
                 engine_commit=PIN['upstream_base'] if arm=='base' else PIN['commit'],
                 bench_source_sha256=PIN['bench_source_sha256'],binary_sha256='1'*64,
                 prompt_sha256='2'*64,system='UNIT',kernel='UNIT',machine='UNIT',
                 ds4_environment={'DS4_ROCM_HALO_PREFILL':str(int(arm=='halo'))})
        (root/'metadata.json').write_text(json.dumps(m),encoding='utf-8')
        payload = root/'payloads';payload.mkdir()
        for t in ([8,16] if incremental else [8]):
            for phase in ['prefill','decode']+(['restored'] if incremental and t==8 else []):
                stem = f'frontier_{t:06d}.{phase}'
                tokens = 2 if phase=='decode' else t
                manifest = dict(frontier=t,phase=phase,state_bytes=32,logits_count=4,
                                token_count=tokens,session_pos=t+2 if phase=='decode' else t,
                                context_alloc=64,restore_kind='snapshot' if phase=='restored' else 'none',
                                snapshot_bytes=32 if phase=='restored' else 0)
                for suffix, data in [('state',b'x'*32),('logits.f32',b'l'*16),('tokens.i32',b't'*tokens*4)]:
                    (payload/(stem+'.'+suffix)).write_bytes(data)
                (payload/(stem+'.manifest.json')).write_text(json.dumps(manifest),encoding='utf-8')
        return root

    def compare(self, success=True, *options):
        r = subprocess.run([sys.executable,str(SCRIPT),str(self.a),str(self.b),*options],
                           text=True,capture_output=True,timeout=10)
        self.assertEqual(r.returncode==0,success,r.stdout+r.stderr)
        return r

    def edit_metadata(self, key, value):
        p=self.b/'metadata.json';m=json.loads(p.read_text());m[key]=value;p.write_text(json.dumps(m))

    def test_snapshot_restore_and_cross_arm_equality(self):
        result=json.loads(self.compare(True,'--require-restore').stdout)
        self.assertTrue(result['restore_exercised'])
        self.assertEqual(len(result['payloads']),15)

    def test_truncated_equal_logits_are_rejected(self):
        for r in [self.a,self.b]:
            (r/'payloads/frontier_000008.prefill.logits.f32').write_bytes(b'l'*4)
        self.compare(False)

    def test_prompt_token_truncation(self):
        (self.b/'payloads/frontier_000008.prefill.tokens.i32').write_bytes(b't'*4)
        self.compare(False)

    def test_cross_arm_state_difference(self):
        (self.b/'payloads/frontier_000016.decode.state').write_bytes(b'y'*32)
        result=json.loads(self.compare(False).stdout)
        self.assertEqual(result['status'],'FAIL_BINARY_EQUALITY')

    def test_restore_state_difference(self):
        (self.b/'payloads/frontier_000008.restored.state').write_bytes(b'y'*32)
        self.compare(False)

    def test_missing_file(self):
        (self.b/'payloads/frontier_000016.decode.state').unlink()
        self.compare(False)

    def test_source_pin_mismatch(self):
        self.edit_metadata('engine_commit','3'*40);self.compare(False)

    def test_unverified_model(self):
        self.edit_metadata('model_sha256_verified_this_run',False);self.compare(False)

    def test_selector_mismatch(self):
        self.edit_metadata('ds4_environment',{'DS4_ROCM_HALO_PREFILL':'0'});self.compare(False)

    def test_zero_step_is_rejected(self):
        m=json.loads((self.b/'metadata.json').read_text());m['config']['step_incr']=0
        self.edit_metadata('config',m['config']);self.compare(False)

    def test_replay_is_not_snapshot_restore(self):
        p=self.b/'payloads/frontier_000008.restored.manifest.json'
        m=json.loads(p.read_text());m.update(restore_kind='replay',snapshot_bytes=0);p.write_text(json.dumps(m))
        self.compare(False,'--require-restore')

    def test_fresh_case_cannot_claim_restore(self):
        shutil.rmtree(self.a);shutil.rmtree(self.b)
        self.a=self.fixture('base',False);self.b=self.fixture('halo',False)
        self.compare(True);self.compare(False,'--require-restore')


if __name__=='__main__':
    unittest.main()
