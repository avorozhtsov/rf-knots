"""Opt-in immutable-snapshot v28 learner. NEVER changes the v27 teacher pointer.

An audited new bank is admitted only in a new spec/output at full-state handoff.
The source/path/shard in minute metrics proves actual consumption, not just READY.
"""
from pathlib import Path
import train_continuous_random_view_r2_v24 as old
import positive_sampler_v28 as sampler


def joint_step(*args, **kwargs):
    result = old.positive.joint_step(*args, **kwargs)
    s = sampler.LAST
    return {**result, 'positive_sampling_source': s['source'],
            'random_positive_draw': int(s['source'] == 'random'),
            'positive_exact_view_word': s['word'], 'positive_exact_view_strands': s['strands'],
            'positive_observation_shard': s['shard'], 'positive_admitted_path_id': s['path_id']}


def setup(path):
    settings, order, frozen = old.setup(path)
    sampler.initialize(settings['positive_bank'], settings['random_positive_bank'])
    old.positive.select = sampler.select
    old.guard.core.base.train_step = joint_step
    old.guard.core.base.CONFIG['positive_group_sampling'] = 'source50_group_view_semanticpath_roundrobin_v28'
    for f in [Path(__file__), Path(sampler.__file__), Path('research/460-v28-connected-learning-protocol.md')]:
        frozen[str(f)] = old.guard.core.base.sha(f)
    return settings, order, frozen


if __name__ == '__main__':
    old.guard.core.PROTOCOL = Path('research/460-v28-connected-learning-protocol.md')
    old.guard.core.setup = setup
    old.guard.core.numerical_gate = old.guard.gate
    old.guard.core.main()
