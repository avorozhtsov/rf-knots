import run_residual_train_mcts_v4 as provider


def test_same_balanced_roots_fresh_replication_seeds():
    rows = [{"stage": s} for s, count in enumerate([20, 15, 18, 35])
            for _ in range(count)]
    provider.ORDER = provider.stage_order(rows)
    assert [rows[provider.ORDER[i]]["stage"] for i in range(8)] == [0, 1, 2, 3] * 2
    for r in provider.ORDER[:8]:
        assert {(w, seed) for q, w, seed in [provider.task(p, 88) for p in range(32)] if q == r} == {
            (0, 20277442), (0, 20277443), (1, 20277442), (1, 20277443)}
