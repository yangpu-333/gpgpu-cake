"""New observer directories; never revive the closed failed V1 observer."""
import drive_sft_phased_acceptance_20261024 as previous
previous.LOCAL=previous.BASE/'results/sft-phased-acceptance-driver-20261024-02'
previous.ROOT=previous.PREP+'/sft-phased-acceptance-20261024-02'
if __name__=='__main__':
    raise SystemExit(previous.main())
