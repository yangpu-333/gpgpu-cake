"""Continue with a fresh evidence directory after the native load-order failure."""
import hashlib
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    payload = (PREP / 'continue_pinning_recovery.py').read_bytes()
    if hashlib.sha256(payload).hexdigest() != '2f04eb3ce094eeba841cb0e1aba7ce9c752b779210c4d9a18c802d7392d9b39a':
        raise ValueError('Prior recovery controller changed')
    text = payload.decode()
    changes = (
        ('sg-huids25-261016-3d89f-default0-0', 'sg-huids25-261016-24e4c-default0-0', 1),
        ('pinning-recovery-20261016-01', 'pinning-recovery-20261016-04', 1),
        ('pinning-semantics-20261016-01', 'pinning-semantics-20261016-04', 1),
        ('pinning_resume_compat_entry.py', 'pinning_resume_compat_entry_v2.py', 1),
        ('run_pinning_resume_compat.py', 'run_pinning_resume_compat_v2.py', 2),
        ("CODE = ('continue_pinning_recovery.py',", "CODE = ('continue_pinning_recovery_v2.py', 'continue_pinning_recovery.py', 'checkpoint_hdo_load_compat.py', 'run_pinning_resume_compat.py',", 1),
        ('for i in (1, 2)', 'for i in (7, 8)', 1),
        ('resume-03', 'resume-06', 3),
        ('checkpoint-common-compat-gloo-20261016-rank-', 'checkpoint-common-compat-gloo-20261016-v3-rank-', 1),
        ("'pinning-native-resume-02',", "'pinning-native-resume-02', 'pinning-native-resume-03', 'pinning-native-resume-04', 'pinning-native-resume-05', 'pinning-optimized-resume-05',", 1),
        ("if not proof['run_returned_successfully'] or proof['redirected_calls'] != 1:",
         "if not proof['run_returned_successfully'] or proof['common_load_calls'] < 1 or proof['sharded_load_calls'] < 1 or len(proof['hdo_placeholder_initializations']) != 1:", 1),
        ("                    raise ValueError('Actual real-model checkpoint compatibility was not observed')",
         """                    raise ValueError('Actual real-model checkpoint compatibility was not observed')
            for rank in (0, 1):
                actual = json.loads((PREP / 'runs' / name / f'rank-{rank}/optimizer-state.json').read_text())['optimizers']
                if (len(actual) != 1 or actual[0]['adam_state_entries'] != 199 or
                        actual[0]['elements_with_both_adam_moments'] != 3834518016 or
                        actual[0]['elements_with_master_param'] != 3834518016 or
                        actual[0]['min_step'] != 3 or actual[0]['max_step'] != 3):
                    raise ValueError('Real resumed optimizer step/coverage differs; do not start the next arm')""", 1),
    )
    for before, after, count in changes:
        if text.count(before) != count:
            raise ValueError('Versioned recovery derivation no longer exact: ' + before)
        text = text.replace(before, after)
    exec(compile(text, str(PREP / 'continue_pinning_recovery_v2.generated.py'), 'exec'),
         {'__name__': '__main__', '__file__': __file__})


if __name__ == '__main__':
    main()
