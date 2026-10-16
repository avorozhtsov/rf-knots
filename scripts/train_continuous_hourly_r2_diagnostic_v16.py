"""One fail-closed diagnostic reproduction with exact offending F metrics."""
import inspect
import math
from pathlib import Path

import train_continuous_hourly_r2_v15 as core

ORIGINAL_SETUP = core.setup
ORIGINAL_GATE = core.numerical_gate


def setup(path):
    settings, order, frozen = ORIGINAL_SETUP(path)
    frozen[str(Path(__file__))] = core.base.sha(Path(__file__))
    return settings, order, frozen


def gate(metrics):
    try:
        ORIGINAL_GATE(metrics)
    except ValueError as error:
        context = inspect.currentframe().f_back.f_locals
        f = metrics.get('F_encoder_grad_norm')
        loss = metrics.get('F_loss')
        out = Path(core.generic.SETTINGS['out'])
        core.base.atomic(out/'failure.json', dict(
            schema='rf-knots-v16-exact-numerical-failure-v1', error=str(error),
            metrics=metrics, attempted_optimizer_step=context.get('step', -1)+1,
            durable_step=context.get('m', {}).get('optimizer_steps'),
            caller_line=inspect.currentframe().f_back.f_lineno,
            root_index=context.get('root', {}).get('root_index'),
            group_index=context.get('group_index'),
            predicates=dict(loss_finite=math.isfinite(loss), loss_nonnegative=loss>=0,
                            gradient_finite=math.isfinite(f), gradient_above_1e_minus6=f>1e-6),
            resumable_failed_weights=False))
        raise


if __name__ == '__main__':
    core.PROTOCOL = Path('research/402-hourly-v15-numerical-failure-and-diagnostic-v16.md')
    core.setup = setup
    core.numerical_gate = gate
    core.main()
