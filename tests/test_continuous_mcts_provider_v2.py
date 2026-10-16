import run_continuous_train_mcts_v2 as provider


def test_aligned_handoff_continues_root_and_weight_order():
    provider.OFFSET = 192
    tasks = [provider.task(i, 256) for i in range(32)]
    assert {r for r, _, _ in tasks} == set(range(48, 56))
    for root in range(48, 56):
        assert {(w, seed) for r, w, seed in tasks if r == root} == {
            (0, 20277420), (0, 20277421), (1, 20277420), (1, 20277421)}


def test_global_pass_boundary_gets_new_independent_seeds_without_omission():
    provider.OFFSET = 992
    before = [provider.task(i, 256) for i in range(32)]
    after = [provider.task(i, 256) for i in range(32, 64)]
    assert {r for r, _, _ in before} == set(range(248, 256))
    assert {r for r, _, _ in after} == set(range(8))
    assert {seed for _, _, seed in before} == {20277420, 20277421}
    assert {seed for _, _, seed in after} == {20277422, 20277423}
