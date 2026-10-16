"""Independent v2 replay audit with the prospectively shorter rollout horizon."""

import audit_reference_guided_root_f_pilot_v1 as A


ORIGINAL_AUDIT_EPISODE = A.audit_episode


def audit_episode_short(instance, episode, config):
    revised = {**config, "horizon": 24}
    return ORIGINAL_AUDIT_EPISODE(instance, episode, revised)


if __name__ == "__main__":
    A.ROOT = A.ROOT.with_name("f-reference-guided-root-pilot-v2")
    A.SEEDS = (20271338, 20271339)
    A.audit_episode = audit_episode_short
    A.__file__ = __file__
    A.main()
