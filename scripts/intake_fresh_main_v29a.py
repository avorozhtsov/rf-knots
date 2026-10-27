"""Future intake retains full replay, with separately corrected UID identity check."""
import argparse
from pathlib import Path
import intake_fresh_main_v29 as parent
from audit_fresh_main_eligibility_v29 import validate_fresh

ORIGINAL_PINS=parent.code_pins


def pins():
    return {**ORIGINAL_PINS(),**{str(p):parent.sha(p) for p in [Path(__file__),
        Path('scripts/audit_fresh_main_eligibility_v29.py')]}}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['out','parent-bank','parent-audit']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--first-batch',type=int,required=True);p.add_argument('--batches',type=int,required=True)
    a=p.parse_args();assert a.first_batch>=1224;parent.code_pins=pins
    parent.intake_main(a.out,a.parent_bank,a.parent_audit,a.first_batch,a.batches)
    print(dict(status='fresh_MAIN_audited',eligible=len(validate_fresh(a.out/'bank'))))
