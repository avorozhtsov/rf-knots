"""One reviewed pre-optimizer transport recovery, not a general force-resume."""
from pathlib import Path
import sys
import connected_learning_supervisor_v30 as supervisor
from probe_compact_inheritance_v31a import main as probe

ROOT = supervisor.ROOT
REVIEW = ROOT/'transport-review-2056741-v31a.json'
LAUNCH = ROOT/'transport-resume-2056741-v31a-launch.json'


def main():
    assert not REVIEW.exists() and not LAUNCH.exists(), 'one reviewed transport launch only'
    assert not (ROOT/'failure.json').exists()
    for fragment in ('scripts/train_connected_learning_live_', 'scripts/connected_learning_supervisor_v30.py',
                     'scripts/collect_connected_learning_', 'scripts/run_continuous_train_mcts_v2.py'):
        assert not supervisor.has_process(fragment)
    actual = supervisor.read(ROOT/'state.json')
    assert actual['status'] == 'STOPPED_LEARNER' and actual['exit_code'] == 1
    assert actual['stop_reason'] == 'storage_gate' and actual['round'] == 20
    assert not actual['supply_stopped'] and not actual['main_intake_stopped']
    _, parent, manifest, entry, saved = supervisor.validate_parent()
    before = supervisor.read(Path('artifacts/compact-storage-v31/pre-resume-2056741-semantics.json'))
    assert entry['step'] == saved['step'] == before['saved_step'] == 2056741
    assert entry['sha256'] == before['saved_sha256'] == 'e9cf0673d3c615520a42edf88d065c8ed288a0189a1a31396bc7f79071793deb'
    progress = supervisor.read(parent/'progress.json')
    assert progress['live_optimizer_steps'] <= 2056741
    log = (parent/'runtime.log').read_text()
    assert 'FileNotFoundError' in log and 'main-intake-001563-001566/bank/path-00000.json' in log
    assert 's,order,frozen=parent.setup(path)' in log
    assert supervisor.sha(Path('artifacts/paired-learning-48h100k-v3/baseline.pt')) == before['A_sha256']
    assert supervisor.read(Path('artifacts/random-TRAIN-stream-v30/window-000305-000308/storage-recovery-completed-v31.json'))['status'] == 'passed'
    probe()
    state_sha = supervisor.sha(ROOT/'state.json')
    supervisor.atomic(REVIEW, dict(status='reviewed_pre_optimizer_transport_failure', failed_exit_code=1,
        native_updates=0, actual_failed_state=actual, actual_failed_state_sha256=state_sha,
        runtime_log_sha256=supervisor.sha(parent/'runtime.log'), full_checkpoint=entry,
        pre_resume_semantics_sha256=supervisor.sha(Path('artifacts/compact-storage-v31/pre-resume-2056741-semantics.json')),
        independent_reader_probe='passed child and grandchild with overwritten PYTHONPATH', no_search_retry=True))
    original_read = supervisor.read
    used = False
    def read_reviewed(path):
        nonlocal used
        data = original_read(path)
        if Path(path) == ROOT/'state.json' and not used:
            assert supervisor.sha(Path(path)) == state_sha and data == actual
            used = True
            return {**data, 'exit_code':0, 'reviewed_transport_failure_exit_code':1,
                'reviewed_transport_receipt':str(REVIEW)}
        return data
    supervisor.atomic(LAUNCH, dict(status='one_shot_reviewed_launch', review_sha256=supervisor.sha(REVIEW),
        full_step=2056741, full_sha256=entry['sha256'], no_smoke_no_commission_no_reset=True))
    supervisor.read = read_reviewed
    sys.argv = [supervisor.__file__, '--resume']
    try:
        supervisor.main()
    except BaseException as error:
        supervisor.atomic(ROOT/'transport-resume-2056741-v31a-failure.json', dict(error=repr(error), no_retry=True))
        raise


if __name__ == '__main__': main()
