"""Frozen pair-similarity gate for validation-neighbour quarantine."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder

SCHEMA = "rf-knots-embedding-pair-audit-gate-v0"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audit_encoder(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[ScalableBraidEmbeddingEncoder, dict]:
    """Load a calibration checkpoint as an inference-only audit encoder."""
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if payload.get("schema") != "rf-knots-embedding-calibration-v0":
        raise ValueError(f"unsupported audit checkpoint schema {payload.get('schema')!r}")
    config = payload["config"]
    model = ScalableBraidEmbeddingEncoder(
        width=int(config["width"]),
        depth=int(config["depth"]),
        embedding_dim=int(config["embedding_dim"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


@dataclass(frozen=True)
class FrozenPairAuditGate:
    """Reference pair bank with an orientation-invariant cosine threshold."""

    reference_left: Tensor
    reference_right: Tensor
    threshold: float
    enabled: bool
    checkpoint_sha256: str
    protocol_sha256: str

    def to(self, device: torch.device) -> FrozenPairAuditGate:
        return FrozenPairAuditGate(
            self.reference_left.to(device),
            self.reference_right.to(device),
            self.threshold,
            self.enabled,
            self.checkpoint_sha256,
            self.protocol_sha256,
        )

    def score(self, left: Tensor, right: Tensor) -> Tensor:
        """Return max symmetric pair cosine similarity to the reference bank."""
        if left.ndim != 2 or right.shape != left.shape:
            raise ValueError("audit endpoints must have equal [batch, dimension] shape")
        if self.reference_left.ndim != 2 or self.reference_left.shape != self.reference_right.shape:
            raise ValueError("audit reference endpoints have inconsistent shape")
        if left.shape[1] != self.reference_left.shape[1]:
            raise ValueError("audit embedding dimensions differ")
        direct = (
            left @ self.reference_left.T + right @ self.reference_right.T
        ) / 2.0
        swapped = (
            left @ self.reference_right.T + right @ self.reference_left.T
        ) / 2.0
        return torch.maximum(direct, swapped).amax(dim=1)

    def rejects(self, left: Tensor, right: Tensor) -> Tensor:
        if not self.enabled:
            return torch.zeros(left.shape[0], dtype=torch.bool, device=left.device)
        return self.score(left, right) >= self.threshold


def load_pair_audit_gate(path: str | Path) -> FrozenPairAuditGate:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported pair audit gate schema {payload.get('schema')!r}")
    return FrozenPairAuditGate(
        payload["reference_left"],
        payload["reference_right"],
        float(payload["threshold"]),
        bool(payload["enabled"]),
        str(payload["checkpoint_sha256"]),
        str(payload["protocol_sha256"]),
    )
