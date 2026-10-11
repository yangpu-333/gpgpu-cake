"""Add the missing native oracle snippet to the not-yet-released skill draft.

Only generated source/wiki prose changes. Every experimental evidence byte and
the original packager are preserved; no CUDA code is executed by this repair.
"""
import argparse
import hashlib
import json
from pathlib import Path

SNIPPET = '''\n## Native weighted activation oracle\n\nOn the matching vendor runtime, use the native implementation as the forward/backward oracle before replacing it:\n\n```python
import torch
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl

x = torch.randn(257, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
w = torch.rand(257, 1, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = weighted_bias_swiglu_impl(x, None, w, False)
dx, dw = torch.autograd.grad(y, (x, w), torch.randn_like(y))
assert dx.dtype == dw.dtype == torch.bfloat16
```
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skill', type=Path, required=True)
    args = parser.parse_args()
    root = args.skill.resolve(strict=True)
    bundle = root / 'evidence/bi-v150-corex42-stage35-retry01'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for row in manifest['files']:
        assert sha(bundle / row['path']) == row['sha256']
    changes = []
    for relative in ('sources/experiments/bi-v150-corex42-stage35-retry01.md',
                     'wiki/kernels/weighted-swiglu-heldout-megatron-bi-v150.md'):
        path = root / relative
        before = sha(path)
        text = path.read_text(encoding='utf-8')
        assert '## Native weighted activation oracle' not in text
        path.write_text(text + SNIPPET, encoding='utf-8', newline='\n')
        changes.append({'path': relative, 'before_sha256': before, 'after_sha256': sha(path)})
    print(json.dumps({'status': 'unreleased_draft_repaired', 'changes': changes,
                      'all_original_evidence_bytes_unchanged': True, 'gpu_work_executed': False}))


if __name__ == '__main__':
    main()
