import torch

from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder


def test_scalable_encoder_reads_variable_length_batch() -> None:
    model = ScalableBraidEmbeddingEncoder(width=16, depth=2, embedding_dim=8)
    word = torch.tensor([[1, -2, 1], [1, 0, 0]])
    output = model(word, torch.tensor([3, 1]), torch.tensor([3, 2]))
    assert set(output) == {"knot", "isotopy", "layout"}
    assert output["knot"].shape == (2, 8)
    assert torch.allclose(output["knot"].norm(dim=-1), torch.ones(2), atol=1e-5)
