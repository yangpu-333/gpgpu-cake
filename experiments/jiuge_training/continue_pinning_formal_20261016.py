"""Same frozen formal protocol, on the renewed allocation and original GPUs."""
import hashlib
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    payload = (PREP / 'continue_pinning_formal.py').read_bytes()
    if hashlib.sha256(payload).hexdigest() != '41a2968b000b9f0d8e6c559233804733b694ea4c8a7e4dd195fd67f1738b4773':
        raise ValueError('Frozen formal protocol changed')
    text = payload.decode()
    for before, after in (
        ('sg-huids25-261015-0d9b8-default0-0', 'sg-huids25-261016-24e4c-default0-0'),
        ('pinning-formal-20261015-01', 'pinning-formal-20261016-01'),
        ("CODE = ('continue_pinning_formal.py',", "CODE = ('continue_pinning_formal_20261016.py', 'continue_pinning_formal.py',")):
        if text.count(before) != 1:
            raise ValueError('Allocation derivation no longer exact')
        text = text.replace(before, after)
    # Emit only outside timed cases, through the existing SSH stdout channel.
    # The local coordinator can show progress without remote polling.
    before = "                with (root / f'{case}.log').open('x') as stream:"
    after = "                print(json.dumps({'event': 'formal_case_start', 'case': case, 'utc': datetime.now(timezone.utc).isoformat()}), flush=True)\n" + before
    if text.count(before) != 1:
        raise ValueError('Formal case-start boundary changed')
    text = text.replace(before, after)
    before = "                state['completed_cases'].append(case)\n                save()"
    after = before + "\n                print(json.dumps({'event': 'formal_case_complete', 'case': case, 'completed_cases': len(state['completed_cases']), 'utc': datetime.now(timezone.utc).isoformat()}), flush=True)"
    if text.count(before) != 1:
        raise ValueError('Formal case-completion boundary changed')
    text = text.replace(before, after)
    before = '        save()\n        return result'
    after = "        save()\n        print(json.dumps({'event': 'formal_round_complete', 'round': name, 'promotion_passed': result['round_promotion_passed'], 'repeat_required': result['repeat_required'], 'gain_vs_control': result['gain_vs_control']}), flush=True)\n        return result"
    if text.count(before) != 1:
        raise ValueError('Formal round-completion boundary changed')
    text = text.replace(before, after)
    exec(compile(text, str(PREP / 'continue_pinning_formal_20261016.generated.py'), 'exec'),
         {'__name__': '__main__', '__file__': __file__})


if __name__ == '__main__':
    main()
