"""Only bind the frozen Oct24 formal runner to new source/admission paths."""
from sft_phased_formal_20261025 import configure
configure()
import run_sft_phased_acceptance_20261024 as original

derive = original.derive_generated


def generated(text):
    value = derive(text)
    before = "str(PREP/'sft_phased_acceptance_entry_20261024.py'),'--evidence'"
    assert value.count(before) == 1
    return value.replace(before, "str(PREP/'sft_phased_formal_entry_20261025.py'),'--evidence'")


if __name__ == '__main__':
    original.derive_generated = generated
    original.main()
