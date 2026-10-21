"""Derive the bounded key-only coordinator for the renewed ACT allocation."""
import hashlib
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    payload = (base / 'drive_pinning_validation.py').read_bytes()
    if hashlib.sha256(payload).hexdigest() != 'a173135d45e860b802fd3fdcd4aa4026a864f8f78d265872285ba5fa3c7d5f95':
        raise ValueError('Original coordinator changed')
    text = payload.decode()
    replacements = (
        ('32405', '31457', 2),
        ("'-o', 'ConnectTimeout=12',", "'-o', 'ConnectTimeout=12', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6', '-o', 'UserKnownHostsFile='+str(BASE/'results/act-known-hosts-20261016'),", 2),
        ('results/pinning-driver-20261015-02', 'results/pinning-driver-20261016-05', 1),
        ('sg-huids25-261015-0d9b8-default0-0', 'sg-huids25-261016-24e4c-default0-0', 1),
        ('pinning-followup-20261015-02', 'pinning-recovery-20261016-04', 1),
        ('pinning-complete-key-results-20261015-v2', 'pinning-complete-key-results-20261016', 1),
        ('local-pinning-complete-key-verification-20261015', 'local-pinning-complete-key-verification-20261016', 1),
        ('pinning-formal-admission-20261015', 'pinning-formal-admission-20261016', 1),
        ('pinning-semantics-20261015-02', 'pinning-semantics-20261016-04', 1),
        ('-u continue_pinning_formal.py --admission', '-u continue_pinning_formal_20261016.py --admission', 1),
        ('pinning-formal-20261015-01', 'pinning-formal-20261016-01', 2),
        ('pinning-formal-key-results-20261015', 'pinning-formal-key-results-20261016', 3),
        ('shutil.copyfile(__file__, folder / Path(__file__).name)',
         "shutil.copyfile(__file__, folder / Path(__file__).name)\n    (folder / 'original-driver-source.py').write_bytes(_original_driver_source)", 1),
    )
    for before, after, count in replacements:
        if text.count(before) != count:
            raise ValueError('Coordinator derivation no longer exact: ' + before)
        text = text.replace(before, after)
    exec(compile(text, str(base / 'drive_pinning_recovery.generated.py'), 'exec'),
         {'__name__': '__main__', '__file__': __file__, '_original_driver_source': payload})


if __name__ == '__main__':
    main()
