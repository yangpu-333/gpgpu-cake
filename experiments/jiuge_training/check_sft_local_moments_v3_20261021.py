"""Run the frozen CPU full-state/resume fixtures against direct-zero creation."""
import hashlib
from pathlib import Path

EXPECTED = '6cd4dc7b695f4f7fd8fe62fbdee315f03886ae846b0744e3c2a187c0fcfc09f0'


def generated():
    source = (Path(__file__).parent/'check_sft_local_moments_20261021.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != EXPECTED:
        raise ValueError('Closed CPU fixture source differs')
    text = source.decode()
    changes = [
        ('sft-moment-placement-preparation-20261021-01', 'sft-moment-direct-zero-preparation-20261021-01'),
        ('from sft_local_moments_20261021 import install,node_for_current_cpus,relocate_moment',
         'from sft_local_moments_v3_20261021 import install,node_for_current_cpus,relocate_moment'),
        ("PREP/'sft_local_moments_20261021.py'", "PREP/'sft_local_moments_v3_20261021.py'"),
        ("    uninstall = install(torch,events)",
         "    native_zeros_like = torch.zeros_like\n    uninstall = install(torch,events)\n    assert torch.zeros_like is native_zeros_like"),
        ("    value = {'passed':True", "    assert torch.zeros_like is native_zeros_like\n"
         "    assert sum(e['initialization_path']=='direct_local_zero' for e in events)==8\n"
         "    assert sum(e['initialization_path']=='native_resume_copy' for e in events)==8\n"
         "    assert all(e['copied_source_moment_bytes']==0 and not e['native_source_moment_created'] for e in events if e['initialization_path']=='direct_local_zero')\n"
         "    value = {'passed':True"),
        ("'placement_events':len(x['placement_events']),", "'placement_events':len(x['placement_events']),"
         "'direct_zero_events':sum(e['initialization_path']=='direct_local_zero' for e in x['placement_events']),"
         "'native_resume_copy_events':sum(e['initialization_path']=='native_resume_copy' for e in x['placement_events']),"),
        ('jiuge7b-sft-local-moments-CPU-regression-v1', 'jiuge7b-sft-direct-zero-moments-CPU-regression-v3'),
    ]
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Exact frozen fixture derivation required: '+before)
        text = text.replace(before, after)
    return text


if __name__ == '__main__':
    exec(compile(generated(), '<direct-zero-frozen-CPU-fixtures>', 'exec'),
         {'__name__':'__main__', '__file__':__file__})
