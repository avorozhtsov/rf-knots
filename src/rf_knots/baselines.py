"""Adapters for fixed external unknot-recognition and simplification baselines.

Optional dependencies are imported only inside adapter calls.  This keeps the
core JAX environment small while making benchmark failures explicit instead of
silently replacing a missing baseline with a weaker one.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass


class BaselineUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class BaselineResult:
    name: str
    status: str
    elapsed_seconds: float
    input_crossings: int
    output_crossings: int | None
    detail: str = ""


def _spherogram_link(word: Iterable[int]):
    try:
        from spherogram import Link
    except ImportError as error:
        raise BaselineUnavailable("install the 'snappy' Python package") from error
    return Link(braid_closure=[int(x) for x in word if int(x)])


def run_snappy(word: Iterable[int], strands: int, mode: str = "global") -> BaselineResult:
    """Run Spherogram's documented simplifier; success is a positive witness only."""
    letters = tuple(int(x) for x in word if int(x))
    if not letters and strands == 1:
        return BaselineResult("snappy-spherogram", "unknot", 0.0, 0, 0, "empty one-braid")
    started = time.monotonic()
    link = _spherogram_link(letters)
    link.simplify(mode)
    elapsed = time.monotonic() - started
    remaining = len(link.crossings)
    return BaselineResult(
        "snappy-spherogram",
        "unknot" if remaining == 0 else "inconclusive",
        elapsed,
        len(letters),
        remaining,
        f"Link.simplify({mode!r}); failure to simplify is not a nontriviality proof",
    )


def _pd_tsv(word: Iterable[int]) -> str:
    link = _spherogram_link(word)
    return "".join("\t".join(str(int(x)) for x in crossing) + "\n"
                   for crossing in link.PD_code())


def run_reapr(
    word: Iterable[int],
    strands: int,
    executable: str = "knoodlesimplify",
    timeout: float = 60.0,
) -> BaselineResult:
    """Run Knoodle's ReAPR CLI in streaming mode on an unsigned PD code."""
    letters = tuple(int(x) for x in word if int(x))
    if not letters and strands == 1:
        return BaselineResult("reapr", "unknot", 0.0, 0, 0, "empty one-braid")
    resolved = shutil.which(executable)
    if resolved is None:
        raise BaselineUnavailable(
            "knoodlesimplify was not found; install Knoodle or pass its executable path"
        )
    payload = _pd_tsv(letters)
    started = time.monotonic()
    completed = subprocess.run(
        [resolved, "--streaming-mode", "--quiet", "--simplify-level=reapr"],
        input=payload,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode:
        raise RuntimeError(f"ReAPR exited {completed.returncode}: {completed.stderr.strip()}")
    numeric = [line for line in completed.stdout.splitlines()
               if line.strip() and line.strip() not in {"k", "s"}]
    remaining = len(numeric)
    return BaselineResult("reapr", "unknot" if remaining == 0 else "simplified",
                          elapsed, len(letters), remaining,
                          "knoodlesimplify --simplify-level=reapr")


def run_regina(word: Iterable[int], strands: int) -> BaselineResult:
    """Run Regina's complete solid-torus recognition on the knot complement."""
    letters = tuple(int(x) for x in word if int(x))
    if not letters and strands == 1:
        return BaselineResult("regina", "unknot", 0.0, 0, 0, "empty one-braid")
    try:
        import regina
    except ImportError as error:
        raise BaselineUnavailable("install the 'regina' Python package") from error
    # Spherogram supplies the well-tested braid-to-planar-diagram conversion.
    pd = [[int(value) + 1 for value in crossing]
          for crossing in _spherogram_link(letters).PD_code()]
    started = time.monotonic()
    link = regina.Link.fromPD(pd)
    if link.countComponents() != 1:
        raise ValueError("Regina adapter received a link instead of a knot")
    is_unknot = bool(link.complement().isSolidTorus())
    elapsed = time.monotonic() - started
    return BaselineResult("regina", "unknot" if is_unknot else "nontrivial",
                          elapsed, len(letters), None,
                          "exact solid-torus recognition of the knot complement")
