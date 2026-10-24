"""Same live CPU math/guards plus own-archive cache-advice CPU fixture."""
from sft_phased_acceptance_recovery_20261024 import configure
import subprocess,sys
if __name__=='__main__':
 core=configure()
 text=(core.PREP/'probe_sft_phased_acceptance_20261024.py').read_text()
 text=text.replace('sft-phased-acceptance-approved-20261024-01.json','sft-phased-acceptance-approved-20261024-02.json')
 exec(compile(text,'<same-live-CPU-math-gates-v2>','exec'),{'__name__':'__main__','__file__':__file__})
 # CPU fixture has an immutable output; each new container uses a distinct fixture root.
 text=(core.PREP/'check_sft_audit_cache_20261024.py').read_text().replace('sft-audit-cache-CPU-20261024-01','sft-audit-cache-CPU-'+__import__('socket').gethostname())
 exec(compile(text,'<live-owned-archive-cache-CPU-test>','exec'),{'__name__':'__main__','__file__':__file__})
