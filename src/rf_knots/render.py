"""Drawing braid words, so that training progress can be looked at rather than
summarised into a single scalar.

Convention: the braid is drawn **top to bottom**, one row per letter, with strand
1 on the left. A positive letter ``+i`` (sigma_i) is drawn with the strand coming
from position `i` passing **over** the strand from position `i+1`; a negative
letter passes under. The closure -- which is the knot the word actually
represents -- is indicated by the arcs on the right joining each strand's bottom
back to its top.
"""

from __future__ import annotations

Word = tuple[int, ...]

_OVER = "X"
_UNDER = "x"


def braid_ascii(word: Word, n: int) -> str:
    """Compact terminal rendering. `X` is an over-crossing, `x` an under-crossing.

    Strand columns are two characters apart, and the crossing symbol sits between
    the two strands it exchanges.
    """
    if n < 1:
        raise ValueError("a braid needs at least one strand")
    width = 2 * n - 1
    rows = []
    for letter in word:
        if letter == 0:
            continue
        index = abs(letter) - 1
        if index + 1 >= n:
            raise ValueError(f"letter {letter} outside B_{n}")
        cells = ["|" if column % 2 == 0 else " " for column in range(width)]
        cells[2 * index] = " "
        cells[2 * index + 1] = _OVER if letter > 0 else _UNDER
        cells[2 * index + 2] = " "
        rows.append("".join(cells))
    if not rows:
        rows = ["|" if column % 2 == 0 else " " for column in range(width)]
        return "".join(rows)
    return "\n".join(rows)


def braid_svg(
    word: Word,
    n: int,
    *,
    row_height: int = 22,
    strand_gap: int = 26,
    margin: int = 16,
    closure: bool = True,
    title: str | None = None,
) -> str:
    """Standalone SVG of the closed braid.

    Each letter is one row. The under-strand is drawn with a gap at the crossing
    so the over/under information is visible, which is the whole point: a
    crossing change (the unknotting move) is exactly a swap of those two.
    """
    if n < 1:
        raise ValueError("a braid needs at least one strand")
    letters = [int(x) for x in word if int(x) != 0]
    rows = max(len(letters), 1)
    arc_width = strand_gap * 1.2 if closure else 0
    width = 2 * margin + (n - 1) * strand_gap + arc_width
    height = 2 * margin + rows * row_height

    def x_of(strand: int) -> float:
        return margin + strand * strand_gap

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="ui-monospace,monospace">',
        '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">',
    ]

    for row, letter in enumerate(letters):
        top = margin + row * row_height
        bottom = top + row_height
        index = abs(letter) - 1
        if index + 1 >= n:
            raise ValueError(f"letter {letter} outside B_{n}")
        for strand in range(n):
            if strand in (index, index + 1):
                continue
            parts.append(
                f'<path d="M {x_of(strand):.1f} {top:.1f} L {x_of(strand):.1f} {bottom:.1f}"/>'
            )
        left, right = x_of(index), x_of(index + 1)
        middle_y = (top + bottom) / 2
        # The strand that ends up on the right, and the one that ends up on the
        # left. `letter > 0` means the left-to-right strand crosses over.
        over_start, over_end = (left, right) if letter > 0 else (right, left)
        under_start, under_end = (right, left) if letter > 0 else (left, right)
        parts.append(
            f'<path d="M {under_start:.1f} {top:.1f} '
            f"L {(under_start + under_end) / 2 - 5:.1f} {middle_y - 5:.1f}"
            f'"/>'
        )
        parts.append(
            f'<path d="M {(under_start + under_end) / 2 + 5:.1f} {middle_y + 5:.1f} '
            f'L {under_end:.1f} {bottom:.1f}"/>'
        )
        parts.append(
            f'<path d="M {over_start:.1f} {top:.1f} L {over_end:.1f} {bottom:.1f}"/>'
        )

    if not letters:
        for strand in range(n):
            parts.append(
                f'<path d="M {x_of(strand):.1f} {margin:.1f} '
                f'L {x_of(strand):.1f} {margin + row_height:.1f}"/>'
            )

    if closure:
        right_edge = margin + (n - 1) * strand_gap
        for strand in range(n):
            reach = right_edge + arc_width * (strand + 1) / (n + 1)
            top = margin
            bottom = height - margin
            parts.append(
                f'<path stroke-dasharray="3 3" opacity="0.45" '
                f'd="M {x_of(strand):.1f} {top:.1f} '
                f"C {reach:.1f} {top - 10:.1f} {reach:.1f} {bottom + 10:.1f} "
                f'{x_of(strand):.1f} {bottom:.1f}"/>'
            )

    parts.append("</g>")
    if title is not None:
        parts.append(
            f'<title>{title}</title>'
        )
    parts.append("</svg>")
    return "".join(parts)


def word_label(word: Word, n: int) -> str:
    """`B3: s1 s2^-1 s1` -- the same format used by `reference.format_word`."""
    letters = [int(x) for x in word if int(x) != 0]
    if not letters:
        return f"B{n}: e"
    body = " ".join(f"s{abs(x)}" + ("^-1" if x < 0 else "") for x in letters)
    return f"B{n}: {body}"
