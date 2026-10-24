"""Use the identical frozen runner with the untimed observer recovery entry."""
from sft_phased_acceptance_recovery_20261024 import configure


def main():
    configure()
    import run_sft_phased_acceptance_20261024 as previous
    original=previous.derive_generated
    def derive(text):
        result=original(text)
        before="str(PREP/'sft_phased_acceptance_entry_20261024.py')"
        if result.count(before)!=1:
            raise ValueError('Exact entry substitution required')
        return result.replace(before,"str(PREP/'sft_phased_acceptance_entry_v2_20261024.py')")
    previous.derive_generated=derive
    previous.main()


if __name__=='__main__':
    main()
