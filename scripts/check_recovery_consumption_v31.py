"""Separate READY from actual consumption of the recovered305..308 path IDs."""
import argparse
from pathlib import Path
from check_connected_learning_live_v30 import check
from check_connected_runtime_v30a import verify_started
from build_mcts_training_supervision import sha, atomic
from run_continuous_train_mcts_v1 import read

RECOVERY = Path('artifacts/random-TRAIN-stream-v30/window-000305-000308/storage-recovery-completed-v31.json')
RECEIPT = Path('artifacts/connected-learning-live-v30/recovery-305-308-consumption-v31.json')


def consumed_ids(recovery_ids, active_ids, drawn_ids):
    return set(recovery_ids) & set(active_ids) & set(drawn_ids)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--write-receipt', action='store_true')
    parser.add_argument('--write-resume-receipt', action='store_true')
    args = parser.parse_args()
    recovery = read(RECOVERY); assert recovery['status'] == 'passed'
    result = check()
    state = read(Path('artifacts/connected-learning-live-v30/state.json'))
    verify_started(result['LIVE'], result['durable'], state['status'], 2056741)
    ref = read(Path('artifacts/continuous-r2-current-training.json'))
    spec = read(Path(ref['spec']))
    progress = read(Path(ref['out'])/'progress.json')
    metrics = progress['metrics']
    assert progress['live_optimizer_steps'] == result['LIVE'], 'metadata changed during check; take a fresh coherent snapshot'
    assert metrics['consumed_random_bank_sha256'] == spec['random_positive_bank_sha256']
    used = consumed_ids(recovery['new_path_ids'], spec['fresh_random_ids'],
        metrics['connected_consumption']['distinct_new_path_ids'])
    assert sha(Path('artifacts/paired-learning-48h100k-v3/baseline.pt')) == 'a2bf6f7b975a134352176e9d8c36059f9e9584bfd4bf47a0614747bf90541d1c'
    status = 'VERIFIED_RECOVERED_RANDOM_IN_ADAMW' if used else 'PENDING_NORMAL_HOURLY_INTAKE_OR_FIRST_DRAW'
    result.pop('v30_random_path_ids', None)
    result.update(status=status, recovered_new_paths=len(recovery['new_path_ids']),
        recovered_ids_in_active_spec=len(set(recovery['new_path_ids']) & set(spec['fresh_random_ids'])),
        recovered_ids_actually_consumed=sorted(used), active_random_bank=spec['random_positive_bank'],
        active_random_bank_sha256=spec['random_positive_bank_sha256'], recovery_sha256=sha(RECOVERY),
        progress_sha256=sha(Path(ref['out'])/'progress.json'), progress_timestamp=progress['timestamp'])
    if args.write_resume_receipt:
        before = read(Path('artifacts/compact-storage-v31/pre-resume-2056741-semantics.json'))
        assert before['saved_step'] == 2056741
        assert before['saved_sha256'] == 'e9cf0673d3c615520a42edf88d065c8ed288a0189a1a31396bc7f79071793deb'
        assert sha(Path('artifacts/continuous-r2-connected-v30-020/step-2056741.pt')) == before['saved_sha256']
        assert spec['out'] == 'artifacts/continuous-r2-connected-v30-020', 'initial resume proof, not a later round'
        old, new = before['source_counts'], metrics['connected_consumption']
        delta = {key:new[key]-old[key] for key in ('retained','random','new_retained','new_random')}
        assert delta['retained'] + delta['random'] == result['LIVE'] - 2056741 > 0
        assert delta['retained'] == delta['new_retained'] and delta['random'] == delta['new_random']
        assert abs(delta['retained']-delta['random']) <= 1
        result.update(resume_native_updates=result['LIVE']-2056741, resume_source_count_delta=delta,
            preserved_A_sha256=before['A_sha256'], pre_resume_B_gate=before['gate'])
        resume_receipt = Path('artifacts/connected-learning-live-v30/resume-2056741-v31-verification.json')
        assert not resume_receipt.exists()
        atomic(resume_receipt, result)
    if args.write_receipt:
        assert used, 'READY/PID or old random IDs are not acceptance'
        assert not RECEIPT.exists(), 'preserve original verification receipt'
        atomic(RECEIPT, result)
    print(result)


if __name__ == '__main__': main()
