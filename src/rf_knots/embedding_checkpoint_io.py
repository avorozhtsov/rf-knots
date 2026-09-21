"""Pinned initial-state loading shared by offline embedding evaluators."""

import hashlib
import io

import torch


def load_initial_checkpoint(run, manifest, architecture):
    """Use saved initial tensors; legacy manifests explicitly return None."""
    expected = manifest.get("initial_checkpoint_sha256")
    if expected is None:
        return None
    path = run / "initial.pt"
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("initial checkpoint SHA mismatch")
    initial = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    if initial["architecture"] != architecture:
        raise ValueError("initial checkpoint architecture mismatch")
    return initial
