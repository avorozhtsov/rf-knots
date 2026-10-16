"""Hourly R2 with canonical finite-positive F gate and exact failure evidence."""
import inspect
import math
import time
from pathlib import Path

import train_continuous_hourly_r2_v15 as core

ORIGINAL_SETUP = core.setup
LOW_COUNT = 0
MIN_OBSERVED = float('inf')
LAST_REPORT = 0.


def validate(metrics):
    keys = ['F_loss', 'F_encoder_grad_norm', 'aux_scale', 'aux_encoder_grad_norm',
            'rehearsal_scale', 'rehearsal_encoder_grad_norm', 'combined_grad_norm']
    if any(not math.isfinite(metrics[k]) for k in keys):
        raise ValueError('nonfinite F or auxiliary numerical state')
    f = metrics['F_encoder_grad_norm']
    if metrics['F_loss'] < 0 or f <= 1e-12:
        raise ValueError('negative loss or zero/vanishing F numerical state')
    for k, limit in [('aux', .100001), ('rehearsal', .200001)]:
        if metrics[k+'_scale'] < 0 or metrics[k+'_encoder_grad_norm'] < 0:
            raise ValueError('negative auxiliary scale or norm')
        if metrics[k+'_scale'] * metrics[k+'_encoder_grad_norm'] > limit*f:
            raise ValueError('effective auxiliary gradient ratio exceeded')


def setup(path):
    settings, order, frozen = ORIGINAL_SETUP(path)
    frozen[str(Path(__file__))] = core.base.sha(Path(__file__))
    evidence = Path('artifacts/continuous-r2-diagnostic-v16/failure.json')
    frozen[str(evidence)] = core.base.sha(evidence)
    if frozen[str(evidence)] != '1a295e4cb76e918786f47dd13f270401acb5daac903503a10ecb5f67af6c206f':
        raise ValueError('measured correction evidence changed')
    # These fields describe guard behavior only, not the training objective.
    core.base.CONFIG['F_encoder_numerical_min'] = 1e-12
    return settings, order, frozen


def gate(metrics):
    global LOW_COUNT, MIN_OBSERVED, LAST_REPORT
    out = Path(core.generic.SETTINGS['out'])
    caller = inspect.currentframe().f_back
    context = caller.f_locals
    try:
        validate(metrics)
    except ValueError as error:
        core.base.atomic(out/'failure.json', dict(error=str(error), metrics=metrics,
            attempted_optimizer_step=context.get('step', -1)+1,
            durable_step=context.get('m', {}).get('optimizer_steps'), caller_line=caller.f_lineno,
            root_index=context.get('root', {}).get('root_index'),
            group_index=context.get('group_index'), resumable_failed_weights=False))
        raise
    f = metrics['F_encoder_grad_norm']
    if f <= 1e-6:
        LOW_COUNT += 1
        MIN_OBSERVED = min(MIN_OBSERVED, f)
        now = time.monotonic()
        if now-LAST_REPORT >= 60:
            core.base.atomic(out/'small-gradient-monitor.json', dict(timestamp=time.time(),
                low_gradient_gate_observations_this_process=LOW_COUNT,
                min_positive_F_grad_this_process=MIN_OBSERVED,
                metrics=metrics, note='finite positive TRAIN gradient; not a utility claim'))
            LAST_REPORT = now


if __name__ == '__main__':
    core.PROTOCOL = Path('research/404-finite-small-f-gradient-guard-correction-v17.md')
    core.setup = setup
    core.numerical_gate = gate
    core.main()
