"""Observe actual compiled-runner calls in a separate, untimed formal launch."""
import argparse
from collections import Counter
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dispatch-evidence', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.dispatch_evidence.exists():
        parser.error('refusing to overwrite dispatch evidence')
    from triton.compiler.compiler import CompiledKernel
    original = CompiledKernel.__getitem__
    launches = []
    def observe(kernel, grid):
        runner = original(kernel, grid)
        metadata = getattr(kernel, 'metadata', {})
        name = metadata.get('name', 'unknown') if isinstance(metadata, dict) else getattr(metadata, 'name', 'unknown')
        def launch(*positional, **keywords):
            result = runner(*positional, **keywords)
            launches.append({'name':str(name), 'grid':list(grid),
                             'asm_keys':sorted(getattr(kernel, 'asm', {}))})
            return result
        return launch
    CompiledKernel.__getitem__ = observe
    status = 'failed'
    try:
        launcher = Path(__file__).with_name('launch_pretrain.py')
        sys.argv = [str(launcher), *remaining]
        runpy.run_path(str(launcher), run_name='__main__')
        status = 'completed'
    finally:
        CompiledKernel.__getitem__ = original
        args.dispatch_evidence.write_text(json.dumps({'status':status, 'launches':launches,
            'counts':dict(Counter(row['name'] for row in launches)),
            'scope':'Successful CompiledKernel runner invocations during formal training; separate untimed process. No hardware instruction equivalence claim.'}, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
