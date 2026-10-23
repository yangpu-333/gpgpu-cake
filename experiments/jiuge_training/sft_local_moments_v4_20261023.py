"""Preserve the native initializer globals and its Dynamo-disabled wrapper.

The frozen V3 math and two factory replacements are unchanged. V3 copied a
wrapper's globals while inspect.getsource silently unwrapped its source.
"""
import hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = BASE/'sft_local_moments_v3_20261021.py'
EXPECTED = '73fff646de66d5ec36ce2d3376bea771b8fe2e74e5f2b55673d5f72b9e3c48e4'


def generated():
    payload = SOURCE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXPECTED:
        raise ValueError('Frozen V3 source differs')
    text = payload.decode()
    changes = (
        ('    namespace = dict(original.__globals__)',
         '    native = inspect.unwrap(original)\n'
         '    if Path(inspect.getsourcefile(native)).resolve() != Path(inspect.getsourcefile(inspect.getmodule(native).AdamW)).resolve():\n'
         "        raise ValueError('Native initializer provenance differs')\n"
         '    namespace = dict(native.__globals__)'),
        ("    return namespace['_init_group']",
         "    result = namespace['_init_group']\n"
         '    if native is not original:\n'
         "        if not getattr(original,'_torchdynamo_disable',False) or original.__globals__.get('__name__') != 'torch._dynamo.eval_frame':\n"
         "            raise ValueError('Only the observed native Dynamo-disabled wrapper is supported')\n"
         "        result = native.__globals__['torch']._dynamo.disable(result)\n"
         '    return result'),
        ('    wrapped._jiuge_local_moments_direct_zero_v3 = True',
         '    wrapped._jiuge_local_moments_direct_zero_v3 = True\n'
         '    wrapped._jiuge_native_initializer_globals_v4 = True'),
    )
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Exact initializer globals repair required: '+before)
        text = text.replace(before, after)
    return text


exec(compile(generated(), str(SOURCE), 'exec'), globals())
