"""Isolated CPU inference probe; never imported by the frozen protocol97 runner."""

import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
from pretrained_raster_tiles import PretrainedRasterTiles, _compile_checked_tiles, checked_word


class TileCacheProbe:
    """Bounded exact-input token reuse, restricted to the audited native trunk.

    This deliberately hashes every weight on every call, including .data writes.
    It is a correctness prototype, not an optimized production cache.
    """

    def __init__(self, encoder, capacity=512):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("positive integer cache capacity required")
        self.encoder, self.capacity = encoder, capacity
        self.entries = OrderedDict()
        self.namespace = None
        self.invalidations = self.logical_tiles = self.computed_tiles = self.hits = 0

    def signature(self):
        from pgx_mcts_bench.networks import MaskedAxialCylinderResidualBlock, MaskedGroupNorm

        net = self.encoder
        if type(net) is not PretrainedRasterTiles:
            raise ValueError("only the existing exact tile encoder is supported")
        allowed = {
            PretrainedRasterTiles,
            torch.nn.Conv2d,
            torch.nn.ModuleList,
            MaskedAxialCylinderResidualBlock,
            MaskedGroupNorm,
        }
        if torch.is_grad_enabled() or any(m.training for m in net.modules()):
            raise ValueError("cache requires eval mode and disabled gradients")
        if any(type(m) not in allowed for m in net.modules()):
            raise ValueError("unsupported trunk module")
        if any(
            type(b) is not MaskedAxialCylinderResidualBlock or b.wrap_strands for b in net.blocks
        ):
            raise ValueError("require the audited nonwrapping masked axial trunk")
        config = [(net.rows, net.columns, net.tile_batch, net.width)]
        for name, module in net.named_modules():
            # Include native norm epsilon/groups and convolution configuration.
            attrs = {
                k: v
                for k, v in vars(module).items()
                if not k.startswith("_")
                and k != "largest_tile_batch"
                and isinstance(v, (int, float, str, bool, tuple))
            }
            config.append((name, type(module).__qualname__, repr(module), attrs))
        h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
        for name, tensor in net.state_dict().items():
            if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                raise ValueError("probe supports CPU float32 only")
            if not torch.isfinite(tensor).all():
                raise ValueError("nonfinite model tensor")
            h.update(name.encode())
            h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.detach().contiguous().numpy().tobytes())
        return h.hexdigest()

    def __call__(self, word, strands):
        namespace = self.signature()
        if namespace != self.namespace:
            self.invalidations += int(self.namespace is not None)
            self.entries.clear()
            self.namespace = namespace
        net = self.encoder
        word = checked_word(word, strands, "cpu")
        outputs = []
        for begin in range(0, len(word), net.tile_batch):
            centers = torch.arange(begin, min(begin + net.tile_batch, len(word)))
            tiles, origins = _compile_checked_tiles(
                word, strands, centers, rows=net.rows, columns=net.columns
            )
            upper = word[centers].abs() - 1 - origins
            if not ((tiles == 0) | (tiles == 1)).all():
                raise ValueError("binary tile key contract changed")
            keys = [
                (t.to(torch.uint8).numpy().tobytes(), int(r))
                for t, r in zip(tiles, upper, strict=True)
            ]
            found, misses = {}, OrderedDict()
            for index, key in enumerate(keys):
                if key in found or key in misses:
                    continue
                if key in self.entries:
                    found[key] = self.entries[key]
                    self.entries.move_to_end(key)
                else:
                    misses[key] = index
            if misses:
                indices = torch.tensor(list(misses.values()))
                hidden = net.spatial(tiles[indices])
                batch, rows, column = torch.arange(len(indices)), upper[indices], net.columns // 2
                tokens = torch.cat(
                    (hidden[batch, :, rows, column], hidden[batch, :, rows + 1, column]), dim=1
                )
                if not torch.isfinite(tokens).all():
                    raise ValueError("nonfinite local token")
                for key, token in zip(misses, tokens, strict=True):
                    found[key] = token.clone()
                    self.entries[key] = found[key]
                    while len(self.entries) > self.capacity:
                        self.entries.popitem(last=False)
            self.logical_tiles += len(keys)
            self.computed_tiles += len(misses)
            self.hits += len(keys) - len(misses)
            outputs.append(torch.stack([found[key] for key in keys]))
        return torch.cat(outputs) if outputs else net.input.weight.new_empty((0, 2 * net.width))


def synthetic_word(length, strands):
    word = list(range(1, strands))
    if length < len(word) or (length - len(word)) % 2:
        raise ValueError("invalid synthetic fixture")
    for i in range((length - len(word)) // 2):
        g = 1 + (37 * i + 11) % (strands - 1)
        word.extend((g, -g))
    return word


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    from pgx_mcts_bench.networks import MaskedAxialCylinderResidualBlock

    checkpoint = Path("artifacts/pretrained-raster-tile-transfer-v1/spatial-trunk.pt")
    expected = "397ca781bb38cee215b53cc068219aed80325c1f84a177d829b879bc80a20771"
    if file_sha(checkpoint) != expected:
        raise ValueError("transferred local checkpoint changed")
    paths = [
        Path(__file__),
        Path("scripts/pretrained_raster_tiles.py"),
        checkpoint,
        args.pgx_root / "src/pgx_mcts_bench/networks.py",
    ]
    pins = {str(p.resolve()): file_sha(p) for p in paths}
    torch.set_num_threads(1)
    teacher = SimpleNamespace(
        variant="axial",
        recurrent_steps=0,
        max_strands=12,
        input=torch.nn.Conv2d(4, 64, 1, bias=False),
        blocks=torch.nn.ModuleList([MaskedAxialCylinderResidualBlock(64) for _ in range(4)]),
    )
    net = PretrainedRasterTiles(teacher).eval()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    probe = TileCacheProbe(net)
    results = []
    with torch.no_grad():
        initial = probe.signature()
        fixtures = [
            ([], 1),
            ([1], 2),
            ([1, 1, 1], 2),
            ([1, -2, 1, 2, -1], 3),
            (synthetic_word(64, 33), 33),
            (synthetic_word(256, 65), 65),
        ]
        for word, strands in fixtures:
            variants = [("original", word), ("repeat", word)]
            if word:
                cc = word.copy()
                cc[len(word) // 2] *= -1
                variants.extend([("crossing_change", cc), ("cyclic_shift", word[1:] + word[:1])])
            for variant, w in variants:
                started = time.perf_counter()
                reference = net(w, strands)
                reference_seconds = time.perf_counter() - started
                before = probe.computed_tiles
                started = time.perf_counter()
                actual = probe(w, strands)
                probe_seconds = time.perf_counter() - started
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
                results.append(
                    dict(
                        length=len(w),
                        strands=strands,
                        variant=variant,
                        max_abs_error=float((actual - reference).abs().max()) if w else 0,
                        bitwise_equal=torch.equal(actual, reference),
                        computed_tiles=probe.computed_tiles - before,
                        reference_seconds=reference_seconds,
                        probe_seconds=probe_seconds,
                    )
                )
        if probe.signature() != initial:
            raise ValueError("probe changed model weights or configuration")
    if any(file_sha(path) != value for path, value in pins.items()):
        raise ValueError("source/checkpoint changed during probe")
    report = dict(
        status="passed",
        scope="local-token engineering probe; not search utility",
        inputs=pins,
        cases=results,
        capacity=probe.capacity,
        logical_tiles=probe.logical_tiles,
        computed_tiles=probe.computed_tiles,
        cache_hits=probe.hits,
        retained_entries=len(probe.entries),
        weights_unchanged=True,
        training_updates=0,
        protocol97_modified=False,
        limitations=[
            "Synthetic fixtures; no TEST or MCTS states",
            "Single timing sample under concurrent workload; not a speedup claim",
            "Full-network, trajectory, and all-move parity still unverified",
        ],
    )
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {k: report[k] for k in ("status", "logical_tiles", "computed_tiles", "cache_hits")}
        )
    )


if __name__ == "__main__":
    main()
