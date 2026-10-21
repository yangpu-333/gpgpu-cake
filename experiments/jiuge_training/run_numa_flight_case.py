"""Derive separate 35-step native-recorder diagnostics; never formal timing."""
import hashlib
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    payload = (PREP / 'run_numa_case.py').read_bytes()
    # The intermediate exact derivation still verifies the frozen native runner.
    expected = (PREP / 'cpu-numa-flight-source-manifest.json').read_text()
    import json
    if hashlib.sha256(payload).hexdigest() != json.loads(expected)['run_numa_case.py']:
        raise ValueError('Diagnostic reference source changed')
    text = payload.decode().replace("str(PREP/'numa_entry.py'),'--evidence'",
                                  "str(PREP/'numa_flight_entry.py'),'--flight-root',str(folder),'--evidence'")
    text = text.replace("    command += ['--numa-mode'", "    if args.steps!=35 or args.seq_length!=2048 or not args.capture_windows or args.full_state_audit or args.reference or args.capture_inputs:\n        raise ValueError('Separate exact 35-step diagnostic required')\n    command += ['--numa-mode'")
    needle = '    record.update(numa_mode=args.numa_mode'
    if text.count(needle) != 1:
        raise ValueError('Diagnostic environment injection changed')
    text = text.replace(needle, "    environment.update(TORCH_NCCL_TRACE_BUFFER_SIZE='100000',TORCH_NCCL_ENABLE_TIMING='1',TORCH_NCCL_TRACE_CPP_STACK='0')\\n" + needle)
    text = text.replace("'numa_entry.py','run_numa_case.py'):", "'numa_entry.py','run_numa_case.py','numa_observation_v2_entry.py','numa_flight_entry.py','run_numa_flight_case.py'):")
    exec(compile(text, str(PREP / 'run_numa_flight_case.derived.py'), 'exec'),
         {'__name__': '__main__', '__file__': __file__})


if __name__ == '__main__':
    main()
