import torch

from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder


def test_embedding_is_invariant_to_unrelated_batch_padding() -> None:
    torch.manual_seed(7)
    model = ScalableBraidEmbeddingEncoder(width=16, depth=1, embedding_dim=8).eval()
    word = torch.tensor([[1, -2, 1]], dtype=torch.long)
    single = model(word, torch.tensor([3]), torch.tensor([3]))["knot"][0]
    padded = torch.tensor([[1, -2, 1, 0, 0], [1, 2, 1, 2, 1]], dtype=torch.long)
    batched = model(padded, torch.tensor([3, 5]), torch.tensor([3, 3]))["knot"][0]
    torch.testing.assert_close(single, batched, atol=1e-6, rtol=1e-6)
