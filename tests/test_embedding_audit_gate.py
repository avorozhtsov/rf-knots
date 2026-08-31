import torch

from rf_knots.embedding_audit_gate import FrozenPairAuditGate


def test_pair_score_is_orientation_invariant() -> None:
    references_left = torch.tensor([[1.0, 0.0]])
    references_right = torch.tensor([[0.0, 1.0]])
    gate = FrozenPairAuditGate(references_left, references_right, 0.9, True, "c", "p")
    direct = gate.score(references_left, references_right)
    swapped = gate.score(references_right, references_left)
    assert torch.allclose(direct, torch.ones(1))
    assert torch.allclose(swapped, torch.ones(1))


def test_disabled_gate_rejects_nothing() -> None:
    references = torch.tensor([[1.0, 0.0]])
    gate = FrozenPairAuditGate(references, references, 0.5, False, "c", "p")
    assert not bool(gate.rejects(references, references).any())
