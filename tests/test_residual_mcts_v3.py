import run_residual_train_mcts_v3 as provider


def test_balanced_first_batch_and_complete_unique_queue():
    rows = [{"stage": s} for s, count in enumerate([20, 15, 18, 35])
            for _ in range(count)]
    provider.ORDER = provider.stage_order(rows)
    assert sorted(provider.ORDER) == list(range(88))
    tasks = [provider.task(p, 88) for p in range(352)]
    assert len({r for r, _, _ in tasks}) == 88
    assert [rows[provider.ORDER[i]]["stage"] for i in range(8)] == [0, 1, 2, 3] * 2
    for r in provider.ORDER[:8]:
        assert {(w, seed) for q, w, seed in tasks[:32] if q == r} == {
            (0, 20277440), (0, 20277441), (1, 20277440), (1, 20277441)}


def test_no_seed_reuse_when_replicating_an_entire_pass():
    provider.ORDER = list(range(88))
    assert provider.task(351, 88) == (87, 1, 20277441)
    assert provider.task(352, 88) == (0, 0, 20277442)
