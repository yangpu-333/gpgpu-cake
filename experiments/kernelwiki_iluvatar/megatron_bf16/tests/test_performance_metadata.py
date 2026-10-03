"""Negative evidence cases must never become claimed BF16 throughput gains."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bf16_performance_reader',ROOT/'summarize_performance.py')
reader=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class PerformanceMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder=Path(self.temporary.name)/'optimized-fixture-perf-26002'
        self.folder.mkdir()
        self.contract=json.loads((ROOT/'contracts/diagnostic.json').read_text())
        self.contract_sha='a'*64
        self.candidate_sha='b'*64
        self.argv=self.contract['training_argv']+['--train-iters','120','--seed','26002','--log-interval','10']
        self.command={'case':self.folder.name,'mode':'optimized','seed':26002,'steps':120,
            'audit':False,'profile':False,'exit_code':0,'contract_sha256':self.contract_sha,
            'whole_process_elapsed_seconds':20,'started_utc':'2026-10-03T08:00:00+00:00',
            'finished_utc':'2026-10-03T08:01:00+00:00',
            'command':['python','-u','/private/suite/launch_training.py','--megatron-root','/private/native',
                '--contract','/private/suite/contracts/0008-main.json','--entry-evidence',
                '/private/suite/evidence/'+self.folder.name+'/entry.json','--candidate','/private/suite/candidates/0008/candidate.py']+self.argv}
        manifest=json.loads((reader.FORMAL/'source_manifest.json').read_text())
        self.handle={'active':True,'enabled':True,'provenance_verified':True,
            'candidate_sha256':self.candidate_sha,'candidate_expected_sha256':self.candidate_sha,
            'runtime_world_size':1,'verified_devices':{'0':'BI-V150'},
            'counts':{'ce_candidate_calls':120,'ce_native_calls':0,'residual_native_calls':480,'fallback_reasons':{}},
            'config_at_installation':{'bf16':True,'fp16':False,'normalization':'RMSNorm'},
            'observations':[{'logits_dtype':'torch.bfloat16','loss_dtype':'torch.float32','autocast':False,
                'logits_shape':[128,2,1024]} for _ in range(4)]}
        self.entry={'status':'completed','contract_sha256':self.contract_sha,
            'native_sources_sha256':manifest['files'],'megatron_commit':manifest['commit'],
            'integration_sha256':reader.integration(),'launcher_sha256':reader.integration()['launch_training.py'],
            'training_argv':self.argv,'formal_entry':'/private/native/pretrain_gpt.py',
            'runtime_compatibility':{'torch':'2.4.1','optional_backends_disabled_in_process':['transformer_engine','apex']},
            'profile_hooks':[],'iluvatar_kernels':True,'kernel_adapters':[self.handle],
            'models':[{'class':'megatron.core.models.gpt.gpt_model.GPTModel','parameters':13701632,
                'layers':4,'hidden_size':512,'vocab_size':1024,'bf16':True,'fp16':False}]}
        lines=[]
        for i in range(10,121,10):
            lines.append('iteration %d/120 | consumed samples: %d | elapsed time per iteration (ms): 20.0 | global batch size: 2 | lm loss: 7.0 | number of skipped iterations: 0 | number of nan iterations: 0 |'%(i,i*2))
        (self.folder/'stdout.log').write_text('\n'.join(lines))
        (self.folder/'stderr.log').write_text('')

    def verify(self):
        (self.folder/'command.json').write_text(json.dumps(self.command))
        (self.folder/'entry.json').write_text(json.dumps(self.entry))
        return reader.validate_case(self.folder,'optimized',26002,self.contract,self.contract_sha,self.candidate_sha)

    def test_complete_coverage_is_validated(self):
        result=self.verify()
        self.assertEqual(result['mean_iteration_ms'],20)
        self.assertEqual(result['tokens_per_second'],12800)

    def test_autocast_is_not_true_bf16_kernel_evidence(self):
        self.handle['observations'][0]['autocast']=True
        with self.assertRaisesRegex(ValueError,'dtype/shape'):
            self.verify()

    def test_api_fallback_rejects_claimed_kernel_coverage(self):
        self.handle['counts']['ce_candidate_calls']=119
        self.handle['counts']['ce_native_calls']=1
        with self.assertRaisesRegex(ValueError,'coverage'):
            self.verify()

    def test_timing_with_profile_is_rejected(self):
        self.command['profile']=True
        with self.assertRaisesRegex(ValueError,'instrumentation'):
            self.verify()

    def test_changed_native_source_is_rejected(self):
        self.entry['native_sources_sha256']['pretrain_gpt.py']='c'*64
        with self.assertRaisesRegex(ValueError,'native source'):
            self.verify()

    def test_hidden_checkpoint_flag_is_rejected(self):
        self.command['command']+=['--save','/tmp/checkpoint']
        with self.assertRaisesRegex(ValueError,'unexpected invocation'):
            self.verify()

    def test_fp32_logits_are_not_bf16_claim(self):
        self.handle['observations'][0]['logits_dtype']='torch.float32'
        with self.assertRaisesRegex(ValueError,'dtype/shape'):
            self.verify()


if __name__=='__main__':
    unittest.main()
