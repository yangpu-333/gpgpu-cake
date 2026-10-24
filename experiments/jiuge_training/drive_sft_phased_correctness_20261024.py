"""One bounded observer of CPU-only correction; never start a GPU experiment."""
from pathlib import Path
import hashlib
import json
import zipfile
import drive_sft_phased_acceptance_20261024 as observer


def main():
    base=Path(__file__).resolve().parent
    observer.LOCAL=base/'results/sft-phased-correctness-driver-20261024-03'
    observer.ROOT=observer.PREP+'/sft-phased-correctness-postprocess-20261024-03'
    code=observer.main()
    if code:return code
    bundle=observer.LOCAL/'final-key-results.zip'
    from pinning_key_results import verify
    checked=verify(bundle,observer.LOCAL/'independent-key-verification.json',64*1024**2)
    with zipfile.ZipFile(bundle) as z:
        root='sft-phased-correctness-postprocess-20261024-03/'
        primary=json.loads(z.read(root+'primary-archive-verification.json'));sem=json.loads(z.read(root+'semantics-verification.json'));pol=json.loads(z.read(root+'resume-policy-review.json'))
        assert primary['passed'] and primary['primary_complete_raw_values_verified'] and primary['checkpoint_resume_tested'] and len(primary['runs'])==10
        assert sem['passed'] and sem['sft_mask_gate'] and sem['full_model_native_checkpoint_resume_verified'] and pol['passed'] and pol['negative_gate_regressions_rejected']==12
        for row in primary['runs']:
            name=f'sft-phased-acceptance-20261024-02/numeric/{row["arm"]}/rank-{row["rank"]}/all-state-manifest.json'
            data=z.read(name);assert hashlib.sha256(data).hexdigest()==row['manifest_sha256']
            m=json.loads(data);assert m['comparison_dtype']=='CPU float64' and m['atol']==3e-4 and m['rtol']==1e-3 and all(x['passed'] for x in m['tensors'])
        result={'passed':True,'GPU_used':False,'full_state_and_native_resume_semantics_passed':True,'ACT_raw_chunks_verified':primary['unique_raw_chunks_verified'],'numeric_runs':primary['runs'],'native_dummy_step_gate_error_corrected_in_CPU_review':True,'formal_performance_passed':False,'formal_performance_started':False,'key_bundle_sha256':checked['bundle_sha256'],'raw_tensors_downloaded':False}
        with (base/'results/sft-phased-correctness-independent-review-20261024-03.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    return 0


if __name__=='__main__':raise SystemExit(main())
