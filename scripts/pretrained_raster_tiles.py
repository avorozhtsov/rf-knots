"""Actual pretrained Q raster trunk on bounded tiles; not a complete solver."""

import copy
import math

import torch
from torch import nn


def checked_word(word, strands, device="cpu"):
    raw = torch.as_tensor(word, device=device)
    if raw.numel() and (raw.is_floating_point() or raw.is_complex() or raw.dtype == torch.bool):
        raise ValueError("Artin generators must be exact integers")
    word = raw.to(dtype=torch.long)
    if word.ndim != 1 or not isinstance(strands, int) or strands < 1:
        raise ValueError("require one signed Artin word and positive strand count")
    if word.numel() and ((word.abs() < 1).any() or (word.abs() >= strands).any()):
        raise ValueError("generator outside active Artin strands")
    return word


def compile_tiles(word, strands, centers, *, rows=12, columns=7):
    """Compile only requested crossing-centered crops, without allocating N*L."""
    word = checked_word(word, strands, getattr(word, "device", "cpu"))
    return _compile_checked_tiles(word, strands, centers, rows=rows, columns=columns)


def _compile_checked_tiles(word, strands, centers, *, rows, columns):
    # The word has already been validated once. Revalidating the whole word for
    # every bounded tile batch would turn the nominally linear pass quadratic.
    centers = torch.as_tensor(centers, dtype=torch.long, device=word.device)
    if rows < 2 or columns < 1 or columns % 2 != 1 or centers.ndim != 1:
        raise ValueError("invalid tile geometry")
    length = word.numel()
    if centers.numel() and (not length or (centers < 0).any() or (centers >= length).any()):
        raise ValueError("tile center outside word")
    origins = (word[centers].abs() - 1 - (rows // 2 - 1)).clamp(0, max(0, strands - rows))
    tile = torch.zeros((len(centers), 4, rows, columns), device=word.device)
    if not len(centers):
        return tile, origins
    row = torch.arange(rows, device=word.device)[None, :] + origins[:, None]
    active = (row < strands).to(tile.dtype)
    tile[:, 1] = active[:, :, None]
    tile[:, 3] = active[:, :, None]
    offsets = torch.arange(columns, device=word.device) - columns // 2
    selected = word[(centers[:, None] + offsets[None, :]) % length]
    upper = selected.abs() - 1 - origins[:, None]
    batch = torch.arange(len(centers), device=word.device)[:, None].expand_as(upper)
    col = torch.arange(columns, device=word.device)[None, :].expand_as(upper)
    for side in (0, 1):
        local_row = upper + side
        valid = (local_row >= 0) & (local_row < rows)
        b, r, c = batch[valid], local_row[valid], col[valid]
        tile[b, 0, r, c] = float(side)
        tile[b, 1, r, c] = ((selected[valid] > 0) if side == 0 else (selected[valid] < 0)).float()
        tile[b, 2, r, c] = float(1 - side)
    return tile, origins


def head_context(word, strands, head, *, rows=12):
    """Cheap head-relative coordinates, independent of learned raster inference."""
    if not isinstance(head, int) or rows < 2:
        raise ValueError("invalid discrete head or row geometry")
    word = checked_word(word, strands, getattr(word, "device", "cpu"))
    length = word.numel()
    if not length:
        return torch.empty((0, 16), device=word.device)
    position = torch.arange(length, device=word.device)
    delta = ((position - int(head)) % length).float() / length
    frequency = torch.tensor([1, 2, 4, 8], device=word.device)
    phase = 2 * math.pi * delta[:, None] * frequency[None, :]
    g = word.abs()
    origin = (g - 1 - (rows // 2 - 1)).clamp(0, max(0, strands - rows))
    geometry = torch.stack(
        (
            g.float() / max(strands - 1, 1),
            torch.log1p((g - 1).float()),
            torch.log1p((strands - g - 1).float()),
            (g - 1 - origin).float() / (rows - 1),
            torch.full_like(delta, math.log1p(length)),
            torch.full_like(delta, math.log1p(strands)),
            (origin > 0).float(),
            (origin + rows < strands).float(),
        ),
        dim=1,
    )
    return torch.cat((phase.sin(), phase.cos(), geometry), dim=1)


class PretrainedRasterTiles(nn.Module):
    """Copy only the exact spatial trunk, with one shared parameter set per word.

    Tokens are head-independent. Fixed crops preserve native normalization scope
    on <=12 strands, not full-raster equivalence at larger strand counts.
    """

    def __init__(self, representation, *, rows=12, columns=7, tile_batch=64):
        super().__init__()
        if (
            representation.variant != "axial"
            or representation.recurrent_steps
            or any(b.wrap_strands for b in representation.blocks)
        ):
            raise ValueError("require ordinary nonrecurrent axial Q trunk")
        if rows != representation.max_strands or columns < 1 or columns % 2 != 1 or tile_batch < 1:
            raise ValueError("invalid native crop or tile batch")
        self.input = copy.deepcopy(representation.input)
        self.blocks = copy.deepcopy(representation.blocks)
        self.rows, self.columns, self.tile_batch = rows, columns, tile_batch
        self.width = self.input.out_channels
        self.largest_tile_batch = 0

    def spatial(self, tile):
        active = tile[:, 3:4]
        hidden = torch.relu(self.input(tile))
        for block in self.blocks:
            hidden = block(hidden, active)
        return hidden

    def forward(self, word, strands):
        word = checked_word(word, strands, self.input.weight.device)
        tokens = []
        for begin in range(0, word.numel(), self.tile_batch):
            centers = torch.arange(
                begin, min(begin + self.tile_batch, len(word)), device=word.device
            )
            tile, origins = _compile_checked_tiles(
                word, strands, centers, rows=self.rows, columns=self.columns
            )
            self.largest_tile_batch = max(self.largest_tile_batch, len(centers))
            hidden = self.spatial(tile)
            batch = torch.arange(len(centers), device=word.device)
            upper = word[centers].abs() - 1 - origins
            col = self.columns // 2
            tokens.append(
                torch.cat((hidden[batch, :, upper, col], hidden[batch, :, upper + 1, col]), 1)
            )
        return torch.cat(tokens) if tokens else self.input.weight.new_empty((0, 2 * self.width))
