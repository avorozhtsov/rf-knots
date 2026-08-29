from rf_knots.embedding_augmentation import (
    MaterializedSyntheticPair,
    SyntheticPairSpec,
)
from rf_knots.embedding_shards import (
    iter_shard_batches,
    load_materialized_shard,
    write_materialized_shard,
)


def test_compact_shard_round_trip(tmp_path) -> None:
    spec = SyntheticPairSpec(0, "hard", "cc", 1, 2, 2, float("inf"), 0, 1, 2, 1)
    rows = [
        MaterializedSyntheticPair(spec, "cc", (1, -1), (2,), 3, 4, 2),
        MaterializedSyntheticPair(spec, "cc", (1,), (2, -2, 1), 3, 4, 4),
    ]
    path = tmp_path / "shard.pt"
    write_materialized_shard(path, rows)
    payload = load_materialized_shard(path)
    batches = list(iter_shard_batches(payload, 2))
    assert len(batches) == 1
    assert batches[0]["left_word"].tolist() == [[1, -1], [1, 0]]
    assert batches[0]["right_length"].tolist() == [1, 3]
    assert batches[0]["lower"].tolist() == [2, 2]
