"""Derive only the formal phase; preserve every frozen scientific source."""
from sft_phased_formal_20261025 import configure, evidence_gate


def generated():
    core = configure()
    source = (core.PREP/'continue_sft_phased_acceptance_20261024.py').read_text()
    start = source.index('    try:\n        update(\'preflight\'')
    formal = source.index("        update('starting-formal',formal_performance_started=True)", start)
    prefix, suffix = source[:start], source[formal:]
    prefix = core.replace_exact(prefix, [
        ("sft-phased-acceptance-20261024.lock", "sft-phased-formal-20261025.lock"),
        ("'run_sft_phased_acceptance_20261024.py'", "'run_sft_phased_formal_20261025.py'"),
    ])
    suffix = core.replace_exact(suffix, [
        ("'phased20261024'+round_name", "'phased20261025'+round_name"),
        ("'sft-phased-acceptance-final-20261024-01.zip'", "'sft-phased-formal-final-20261025-01.zip'"),
    ])
    setup = '''    try:
        update('preflight', initial_resource_gate=resources())
        if int(Path('/sys/fs/cgroup/memory.max').read_text()) < 300*1024**3:
            raise ValueError('Original300GiB allocation and memory guards required')
        if remaining() < 10800:
            raise TimeoutError('Original3h reserve before any formal work required')
        from pinning_key_results import export, KEY_SUFFIXES
        KEY_SUFFIXES.add('.sh')
        admission = evidence_gate()
        admitted = ROOT/'formal-admission.json'
        with admitted.open('x') as stream:
            json.dump(admission, stream, indent=2)
        from sft_phased_formal_20261025 import EVIDENCE
        copied = ROOT/'correctness-evidence'
        copied.mkdir()
        for path in EVIDENCE.iterdir():
            if path.is_file():
                shutil.copyfile(path, copied/path.name)
        from sft_phased_formal_20261025 import CPU
        for name in ('primary-archive-verification.json','semantics-verification.json',
                     'resume-policy-review.json','checkpoint-sha256.json'):
            shutil.copyfile(CPU/name, copied/name)
        state.update(correctness_complete_before_formal=True,
                     new_correctness_GPU_runs=0, old_performance_reused=False,
                     inherited_raw_chunks=18601, raw_SHA_gate='closed Oct24 receipt; unchanged manifests')
'''
    return prefix+setup+suffix


if __name__ == '__main__':
    exec(compile(generated(), '<frozen-formal-only-continuation>', 'exec'),
         {'__name__': '__main__', '__file__': __file__, 'evidence_gate': evidence_gate})
