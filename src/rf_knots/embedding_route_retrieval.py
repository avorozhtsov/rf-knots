"""Route-aware retrieval over a replay-verified unknotting proof graph.

The embedding is only a candidate generator.  Every proposed graph bridge must
still be constructed and replay-verified before it changes an upper bound.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder
from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.embedding_supervision import decode_ukb0


@dataclass(frozen=True)
class GraphRepresentation:
    node_id: int
    representation_id: int
    word: tuple[int, ...]
    strands: int
    word_length: int
    knot_id: str | None
    split: str
    u_upper: int


@dataclass(frozen=True)
class HiddenRouteEdge:
    source_node: int
    target_node: int
    source_representation: int
    target_representation: int
    source_split: str
    target_split: str
    source_u: int
    target_u: int
    edge_id: int
    provenance: str


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {str(key): str(value) for key, value in connection.execute("SELECT key,value FROM meta")}


def validate_snapshot_contract(
    *,
    proof_path: str | Path,
    pairs_path: str | Path,
    identification_path: str | Path,
    lower_bounds_path: str | Path,
) -> dict[str, str]:
    """Fail closed when graph-derived inputs refer to different snapshots."""
    paths = {
        "proof": Path(proof_path),
        "pairs": Path(pairs_path),
        "identification": Path(identification_path),
        "lower_bounds": Path(lower_bounds_path),
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    pairs = sqlite3.connect(f"file:{paths['pairs']}?mode=ro", uri=True)
    identification = sqlite3.connect(f"file:{paths['identification']}?mode=ro", uri=True)
    try:
        pairs_meta = _metadata(pairs)
        identification_meta = _metadata(identification)
    finally:
        pairs.close()
        identification.close()
    expected = pairs_meta.get("proof_sha256")
    if expected != hashes["proof"]:
        raise ValueError(
            "embedding pairs and proof graph SHA mismatch: "
            f"expected {expected}, observed {hashes['proof']}"
        )
    identified_proof = identification_meta.get("graph_snapshot_sha256")
    if identified_proof != hashes["proof"]:
        raise ValueError(
            "identification and proof graph SHA mismatch: "
            f"expected {identified_proof}, observed {hashes['proof']}"
        )
    return hashes


def load_graph_representations(
    proof_path: str | Path, pairs_path: str | Path
) -> tuple[GraphRepresentation, ...]:
    """Select one exact sidecar representation for every graph mirror-orbit node."""
    proof = sqlite3.connect(f"file:{Path(proof_path)}?mode=ro", uri=True)
    pairs = sqlite3.connect(f"file:{Path(pairs_path)}?mode=ro", uri=True)
    try:
        representative_by_mirror: dict[bytes, tuple] = {}
        for row in pairs.execute(
            "SELECT representation_id,mirror_key,encoding,strands,word_length,knot_id,split "
            "FROM representations ORDER BY representation_id"
        ):
            representative_by_mirror.setdefault(bytes(row[1]), row)
        result = []
        for node_id, rep_key, u_upper in proof.execute(
            "SELECT k.node_id,k.rep_key,n.u_upper_bound FROM node_keys k "
            "JOIN nodes n USING(node_id) WHERE k.key_kind=0 ORDER BY k.node_id"
        ):
            row = representative_by_mirror.get(bytes(rep_key))
            if row is None or u_upper is None:
                continue
            decoded = decode_ukb0(int(row[0]), bytes(row[2]), row[5], str(row[6]))
            result.append(
                GraphRepresentation(
                    node_id=int(node_id),
                    representation_id=decoded.representation_id,
                    word=decoded.word,
                    strands=decoded.strands,
                    word_length=int(row[4]),
                    knot_id=decoded.knot_id,
                    split=decoded.split,
                    u_upper=int(u_upper),
                )
            )
    finally:
        proof.close()
        pairs.close()
    return tuple(result)


def load_hidden_exact_cc1_edges(
    proof_path: str | Path, pairs_path: str | Path
) -> tuple[HiddenRouteEdge, ...]:
    """Load evaluation-only CC=1 edges and orient them toward decreasing U."""
    proof = sqlite3.connect(f"file:{Path(proof_path)}?mode=ro", uri=True)
    pairs = sqlite3.connect(f"file:{Path(pairs_path)}?mode=ro", uri=True)
    try:
        node_by_mirror = {
            bytes(rep_key): (int(node_id), int(u_upper))
            for rep_key, node_id, u_upper in proof.execute(
                "SELECT k.rep_key,k.node_id,n.u_upper_bound FROM node_keys k "
                "JOIN nodes n USING(node_id) WHERE k.key_kind=0 AND n.u_upper_bound IS NOT NULL"
            )
        }
        representation_rows = {
            int(row[0]): (bytes(row[1]), str(row[2]))
            for row in pairs.execute(
                "SELECT representation_id,mirror_key,split FROM representations"
            )
        }
        rows = []
        for left, right, edge_id, provenance in pairs.execute(
            "SELECT left_representation,right_representation,edge_id,provenance "
            "FROM excluded_cross_split_pairs WHERE metric='cc' "
            "AND distance_lower=1 AND distance_upper=1 ORDER BY edge_id"
        ):
            left_mirror, left_split = representation_rows[int(left)]
            right_mirror, right_split = representation_rows[int(right)]
            left_node = node_by_mirror.get(left_mirror)
            right_node = node_by_mirror.get(right_mirror)
            if left_node is None or right_node is None:
                continue
            if left_node[1] == right_node[1] + 1:
                source = (left_node, int(left), left_split)
                target = (right_node, int(right), right_split)
            elif right_node[1] == left_node[1] + 1:
                source = (right_node, int(right), right_split)
                target = (left_node, int(left), left_split)
            else:
                continue
            rows.append(
                HiddenRouteEdge(
                    source_node=source[0][0],
                    target_node=target[0][0],
                    source_representation=source[1],
                    target_representation=target[1],
                    source_split=source[2],
                    target_split=target[2],
                    source_u=source[0][1],
                    target_u=target[0][1],
                    edge_id=int(edge_id),
                    provenance=str(provenance),
                )
            )
    finally:
        proof.close()
        pairs.close()
    return tuple(rows)


def load_gap_sources(
    identification_path: str | Path, lower_bounds_path: str | Path
) -> tuple[tuple[int, str, int, int], ...]:
    """Return identified graph nodes whose replayed U upper bound exceeds a lower bound."""
    connection = sqlite3.connect(
        f"file:{Path(identification_path)}?mode=ro", uri=True
    )
    connection.execute(
        "ATTACH DATABASE ? AS lower_bounds", (str(Path(lower_bounds_path)),)
    )
    try:
        rows = connection.execute(
            "SELECT v.node_id,m.knot_id,v.u_upper,b.u_lower FROM graph_vertex_knot_map m "
            "JOIN graph_vertices v USING(rep_key) "
            "JOIN lower_bounds.effective_lower_bounds b ON b.knot_id=m.knot_id "
            "WHERE v.u_upper>b.u_lower ORDER BY v.u_upper-b.u_lower DESC,v.node_id"
        )
        best_by_knot: dict[str, tuple[int, str, int, int]] = {}
        for node_id, knot_id, u_upper, u_lower in rows:
            candidate = (int(node_id), str(knot_id), int(u_upper), int(u_lower))
            previous = best_by_knot.get(candidate[1])
            if previous is None or candidate[2] < previous[2]:
                best_by_knot[candidate[1]] = candidate
        return tuple(
            sorted(
                best_by_knot.values(),
                key=lambda row: (-(row[2] - row[3]), -row[2], row[1]),
            )
        )
    finally:
        connection.close()


def load_checkpoint_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[ScalableBraidEmbeddingEncoder, float, dict]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    config = dict(payload.get("config", {}))
    model = ScalableBraidEmbeddingEncoder(
        width=int(config.get("width", 96)),
        depth=int(config.get("depth", 4)),
        embedding_dim=int(config.get("embedding_dim", 64)),
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    objective = MetricEmbeddingObjective(
        cc_geometry=str(config.get("cc_geometry", "chord")),
        cc_rank_weight=float(config.get("cc_rank_weight", 0.0)),
        cc_rank_margin=float(config.get("cc_rank_margin", 0.25)),
    )
    objective.load_state_dict(payload["objective"])
    return model, float(objective.scale("cc").detach()), payload


def embed_graph_representations(
    model: ScalableBraidEmbeddingEncoder,
    representations: tuple[GraphRepresentation, ...],
    *,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Embed length-bucketed graph representations and return CPU float32 rows."""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    embeddings: Tensor | None = None
    ordered = sorted(
        range(len(representations)),
        key=lambda index: representations[index].word_length,
    )
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            indices = ordered[start : start + batch_size]
            maximum = max((representations[index].word_length for index in indices), default=0)
            padded_length = max(maximum, 1)
            words = torch.zeros((len(indices), padded_length), dtype=torch.long, device=device)
            lengths = torch.empty(len(indices), dtype=torch.long, device=device)
            strands = torch.empty(len(indices), dtype=torch.long, device=device)
            for row_index, representation_index in enumerate(indices):
                representation = representations[representation_index]
                if representation.word:
                    words[row_index, : len(representation.word)] = torch.tensor(
                        representation.word, dtype=torch.long, device=device
                    )
                lengths[row_index] = len(representation.word)
                strands[row_index] = representation.strands
            batch = model(words, lengths, strands)["knot"].float().cpu()
            if embeddings is None:
                embeddings = torch.empty(
                    (len(representations), batch.shape[1]), dtype=torch.float32
                )
            embeddings[torch.tensor(indices, dtype=torch.long)] = batch
    if embeddings is None:
        raise ValueError("no graph representations to embed")
    return embeddings


def deterministic_rank(scores: Tensor, node_ids: Tensor, target_index: int) -> int:
    """One-based descending rank with node id as a stable tie breaker."""
    target_score = scores[target_index]
    target_node = node_ids[target_index]
    better = (scores > target_score) | ((scores == target_score) & (node_ids < target_node))
    return int(better.sum().item()) + 1


def summarize_ranks(ranks: list[int], pool_sizes: list[int]) -> dict:
    if not ranks:
        return {"cases": 0}
    sorted_ranks = sorted(ranks)
    result: dict[str, float | int] = {
        "cases": len(ranks),
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / len(ranks),
        "median_rank": sorted_ranks[(len(sorted_ranks) - 1) // 2],
        "mean_pool_size": sum(pool_sizes) / len(pool_sizes),
    }
    for cutoff in (1, 10, 100, 1000):
        result[f"recall_at_{cutoff}"] = sum(rank <= cutoff for rank in ranks) / len(ranks)
        result[f"random_expected_recall_at_{cutoff}"] = sum(
            min(cutoff / pool_size, 1.0) for pool_size in pool_sizes
        ) / len(pool_sizes)
    return result


def fit_ridge(features: Tensor, targets: Tensor, *, alpha: float = 1.0) -> dict[str, Tensor]:
    """Fit a deterministic standardized ridge probe with an unpenalized intercept."""
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("ridge features and targets have incompatible shapes")
    if len(features) == 0 or alpha < 0:
        raise ValueError("ridge training data must be nonempty and alpha non-negative")
    features = features.double()
    targets = targets.double()
    mean = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(1e-8)
    standardized = (features - mean) / scale
    target_mean = targets.mean()
    identity = torch.eye(standardized.shape[1], dtype=torch.float64)
    weights = torch.linalg.solve(
        standardized.T @ standardized + alpha * identity,
        standardized.T @ (targets - target_mean),
    )
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "target_mean": target_mean,
        "weights": weights,
    }


def predict_ridge(model: dict[str, Tensor], features: Tensor) -> Tensor:
    standardized = (features.double() - model["feature_mean"]) / model["feature_scale"]
    return (model["target_mean"] + standardized @ model["weights"]).float()


def regression_metrics(targets: Tensor, predictions: Tensor) -> dict[str, float | int]:
    if targets.ndim != 1 or predictions.shape != targets.shape or len(targets) == 0:
        raise ValueError("regression metric inputs must be nonempty aligned vectors")
    targets = targets.float()
    predictions = predictions.float()
    error = predictions - targets
    unequal = targets[:, None] != targets[None, :]
    upper = torch.triu(unequal, diagonal=1)
    target_order = torch.sign(targets[:, None] - targets[None, :])[upper]
    predicted_order = torch.sign(predictions[:, None] - predictions[None, :])[upper]
    correct = (target_order == predicted_order).float()
    correct = torch.where(predicted_order == 0, correct.new_tensor(0.5), correct)
    centered_target = targets - targets.mean()
    centered_prediction = predictions - predictions.mean()
    denominator = centered_target.norm() * centered_prediction.norm()
    correlation = (
        float((centered_target @ centered_prediction) / denominator)
        if float(denominator) > 0
        else 0.0
    )
    return {
        "rows": len(targets),
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "pearson": correlation,
        "rounded_accuracy": float((predictions.round() == targets).float().mean()),
        "pairwise_order_accuracy": float(correct.mean()) if len(correct) else 0.0,
    }
