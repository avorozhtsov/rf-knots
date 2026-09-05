"""The first MCTS training loop: a size-independent net that narrows the search.

The premise of note 22 is that *the network* narrows the search, the way a policy
and value net narrow AlphaGo's MCTS.  Everything here exists to run that loop and
measure whether it works, not to prove in advance that it would.

Shape of the loop
-----------------

* **Environment** -- `rf_knots.reference`, exact and in-process.  A search state is
  `(word, strands, crossing_changes_used, moves_used)`; the terminal is the empty
  one-strand braid; the budget is `(moves, crossing_changes)` and a crossing change
  is only offered while its budget is unspent.
* **Search** -- one MCTS tree per instance, `simulations` node expansions, priors
  from `SiteHead` and a leaf value from `p(solve within the remaining budget)`.
  A simulation that reaches the empty braid *is* the witness, and the path is
  replayed through `UnknotWitness.verify()` before it is counted.
  Root selection is either PUCT or Gumbel top-`m` with sequential halving; both
  are implemented because at the simulation counts this project can afford the
  question is empirical.
* **Targets** -- root and internal visit counts for the policy, the observed
  "did this subtree reach the unknot" bit for the value, and the set-valued
  `A_plus` loss of §5.3 wherever a replayed witness exhibits a good site.  Sites
  outside `A_plus` are unexplored, never negative: the `A_plus` term is
  `site_head.set_valued_loss`, which has no way to express a negative.

Size independence
-----------------

`ScalableBraidEmbeddingEncoder(positional_geometry="necklace")` is constructed
explicitly -- the default geometry is the legacy one and is *not* `Z_L`-equivariant
-- and `SiteHead` carries the same 49,833 parameters at every braid size.
`tests/test_mcts_loop.py` runs this loop with one `state_dict` at two very
different `(max_len, max_strands)` and fails if any parameter count moves.
"""

from __future__ import annotations

import functools
import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, ActionSpec
from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder
from rf_knots.evidence import UnknotWitness
from rf_knots.site_head import SiteHead, set_valued_loss, sites_from_actions

Word = tuple[int, ...]

SOLVED: tuple[Word, int] = ((), 1)


# ---------------------------------------------------------------- instances


@dataclass(frozen=True, eq=False)
class Instance:
    """One search problem: a braid, a crossing-change cap, and the room to move.

    `growth_len` and `growth_strands` are the *slack* the `ActionSpec` gets over
    the instance's own shape.  They are a first-class part of the problem, not a
    tuning knob hidden in the search: §2.3 requires that the search be able to
    grow the diagram at all, and `growth_len = 0` would make `INSERT` and
    `STABILIZE` illegal everywhere.
    """

    name: str
    word: Word
    strands: int
    cap_crossing_changes: int
    cap_moves: int = 32
    growth_len: int = 2
    growth_strands: int = 1
    source: str = ""
    crossings: int | None = None
    #: Unique within a run.  Bank row names repeat -- two different `4_1`
    #: presentations sit in the same easy pool -- so a report keyed on `name`
    #: silently merges instances.
    uid: str = ""

    @functools.cached_property
    def spec(self) -> ActionSpec:
        return ActionSpec(len(self.word) + self.growth_len, self.strands + self.growth_strands)

    @property
    def label(self) -> str:
        return self.uid or self.name

    def to_dict(self) -> dict:
        return {
            "uid": self.label,
            "name": self.name,
            "word": list(self.word),
            "strands": self.strands,
            "cap_crossing_changes": self.cap_crossing_changes,
            "cap_moves": self.cap_moves,
            "growth_len": self.growth_len,
            "growth_strands": self.growth_strands,
            "source": self.source,
            "crossings": self.crossings,
        }


def curriculum_rows(path: str | Path) -> list[dict]:
    """The rows of one of the fast-6 cohort's curriculum banks, in registered order.

    The bank's own row order is a vetted easy-first ordering with the answers
    attached ("ten cycles of four low-strand rows then one simple high-strand
    bridge"), so it is preserved rather than re-sorted.
    """
    payload = json.loads(Path(path).read_text())
    return list(payload["rows"])


def instances_from_rows(
    rows: Iterable[dict],
    *,
    source: str,
    cap_moves: int = 32,
    growth_len: int = 2,
    growth_strands: int = 1,
) -> list[Instance]:
    out: list[Instance] = []
    for index, row in enumerate(rows):
        upper = row.get("certified_unknotting_upper_bound")
        if upper is None:
            continue
        out.append(
            Instance(
                name=str(row["name"]),
                word=tuple(int(x) for x in row["word"]),
                strands=int(row["strands"]),
                cap_crossing_changes=int(upper),
                cap_moves=cap_moves,
                growth_len=growth_len,
                growth_strands=growth_strands,
                source=source,
                crossings=row.get("crossings"),
                uid=f"{source}:{index:03d}:{row['name']}",
            )
        )
    return out


# ---------------------------------------------------------------- search state


@dataclass(frozen=True)
class SearchState:
    word: Word
    strands: int
    used_cc: int
    used_moves: int

    @property
    def key(self) -> tuple[Word, int]:
        return (self.word, self.strands)


def successors_of(
    state: SearchState, instance: Instance, spec: ActionSpec
) -> list[tuple[int, SearchState]]:
    """Legal children, with crossing changes offered only while budget remains."""
    if state.used_moves >= instance.cap_moves:
        return []
    allow = state.used_cc < instance.cap_crossing_changes
    out: list[tuple[int, SearchState]] = []
    for action, word, strands in reference.successors(spec, state.word, state.strands, allow):
        kind = spec.decode(action)[0]
        used_cc = state.used_cc + (1 if kind == CROSSING_CHANGE else 0)
        out.append((action, SearchState(word, strands, used_cc, state.used_moves + 1)))
    return out


# ---------------------------------------------------------------- evaluators


@dataclass
class Evaluation:
    priors: np.ndarray
    value: float


class Evaluator:
    """What the search asks for at a leaf: a prior over children and a value."""

    batched: bool = False

    def evaluate(
        self, requests: Sequence[tuple[Instance, SearchState, Sequence[int]]]
    ) -> list[Evaluation]:
        raise NotImplementedError


class UniformEvaluator(Evaluator):
    """No network at all: a flat prior and a constant value.

    This is the search with its heuristic removed, which is the control the whole
    architecture line is arguing against.
    """

    def __init__(self, value: float = 0.5) -> None:
        self.value = float(value)
        self.calls = 0

    def evaluate(
        self, requests: Sequence[tuple[Instance, SearchState, Sequence[int]]]
    ) -> list[Evaluation]:
        self.calls += len(requests)
        return [
            Evaluation(np.full(len(actions), 1.0 / max(len(actions), 1)), self.value)
            for _, _, actions in requests
        ]


# ---------------------------------------------------------------- the network


class SitePolicyValueNet(nn.Module):
    """`ScalableBraidEmbeddingEncoder` + `SiteHead` + a scalar survival head.

    The value head is deliberately the simple one: a single
    `p(solve within the remaining budget)`.  §5.1's monotone head over the whole
    budget vector is the design; this is enough to get the loop running and is
    the piece to upgrade once it does.

    Every parameter dimension here is drawn from `(width, embedding_dim,
    num_kinds, budget_dim, boundary_clip)` and the encoder's fixed vocabularies.
    Nothing is a function of `max_len` or `max_strands`.
    """

    def __init__(
        self,
        *,
        width: int = 96,
        depth: int = 4,
        embedding_dim: int = 64,
        maximum_strands: int = 1024,
        value_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = ScalableBraidEmbeddingEncoder(
            width=width,
            depth=depth,
            embedding_dim=embedding_dim,
            maximum_strands=maximum_strands,
            # Explicit, and the reason is measured: the default geometry moves the
            # `knot` embedding by up to 0.377 under a cyclic word shift, which is
            # a no-op on the knot.  The necklace geometry moves it by 2.2e-16.
            positional_geometry="necklace",
        )
        self.head = SiteHead(width=width, embedding_dim=embedding_dim, budget_dim=2)
        # 3 globals + (remaining moves, remaining crossing changes) + log1p of each
        # + a length/strand pair that is scalar at every size.
        self.value_input = 3 * embedding_dim + 6
        self.value = nn.Sequential(
            nn.Linear(self.value_input, value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, 1),
        )

    @staticmethod
    def value_features(
        globals_: dict[str, torch.Tensor],
        budgets: torch.Tensor,
        lengths: torch.Tensor,
        strands: torch.Tensor,
    ) -> torch.Tensor:
        parts = [globals_["knot"], globals_["isotopy"], globals_["layout"]]
        scalars = torch.cat(
            [
                budgets,
                torch.log1p(budgets.clamp_min(0.0)),
                torch.log1p(lengths.to(budgets.dtype)).unsqueeze(-1),
                torch.log1p(strands.to(budgets.dtype)).unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.cat([*parts, scalars], dim=-1)

    def forward(
        self,
        word: torch.Tensor,
        lengths: torch.Tensor,
        strands: torch.Tensor,
        sites: torch.Tensor,
        site_mask: torch.Tensor,
        budgets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder.encode_sites(word, lengths, strands)
        globals_ = {name: encoded[name] for name in ("knot", "isotopy", "layout")}
        logits = self.head(
            encoded["tokens"],
            encoded["mask"],
            globals_,
            sites,
            site_mask,
            budgets=budgets,
            strands=strands,
        )
        effective = self.head.effective_site_mask(encoded["mask"], sites, site_mask)
        value = self.value(self.value_features(globals_, budgets, lengths, strands)).squeeze(-1)
        return logits, value, effective

    def parameter_report(self) -> dict[str, int]:
        return {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "site_head": sum(p.numel() for p in self.head.parameters()),
            "value_head": sum(p.numel() for p in self.value.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


def build_batch(
    requests: Sequence[tuple[Instance, SearchState, Sequence[int]]],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Pad a set of `(instance, state, candidate actions)` into one network batch.

    The word tensor is padded with zero, which is the encoder's `padding_idx`, and
    the site tensor is padded with rows the `site_mask` marks illegal.  Nothing
    about the batch depends on any instance sharing an `ActionSpec` with any other
    -- that is the property the whole architecture line was built for.
    """
    batch = len(requests)
    max_len = max(max(len(state.word) for _, state, _ in requests), 1)
    max_sites = max(max(len(actions) for _, _, actions in requests), 1)
    word = torch.zeros((batch, max_len), dtype=torch.long)
    lengths = torch.zeros(batch, dtype=torch.long)
    strands = torch.zeros(batch, dtype=torch.long)
    sites = torch.zeros((batch, max_sites, 4), dtype=torch.long)
    site_mask = torch.zeros((batch, max_sites), dtype=torch.bool)
    budgets = torch.zeros((batch, 2), dtype=torch.float32)
    for index, (instance, state, actions) in enumerate(requests):
        length = len(state.word)
        if length:
            word[index, :length] = torch.tensor(state.word, dtype=torch.long)
        lengths[index] = max(length, 0)
        strands[index] = state.strands
        if actions:
            rows = sites_from_actions(actions, instance.spec)
            sites[index, : len(actions)] = rows
            site_mask[index, : len(actions)] = True
        budgets[index, 0] = instance.cap_moves - state.used_moves
        budgets[index, 1] = instance.cap_crossing_changes - state.used_cc
    return {
        "word": word.to(device),
        "lengths": lengths.to(device),
        "strands": strands.to(device),
        "sites": sites.to(device),
        "site_mask": site_mask.to(device),
        "budgets": budgets.to(device),
    }


class NetworkEvaluator(Evaluator):
    """The learned prior and value, evaluated in one batch per search step."""

    batched = True

    def __init__(
        self,
        net: SitePolicyValueNet,
        device: torch.device | str = "cpu",
        *,
        micro_batch: int = 64,
    ) -> None:
        self.net = net
        self.device = torch.device(device)
        self.micro_batch = int(micro_batch)
        self.calls = 0
        self.forwards = 0

    @torch.no_grad()
    def evaluate(
        self, requests: Sequence[tuple[Instance, SearchState, Sequence[int]]]
    ) -> list[Evaluation]:
        self.calls += len(requests)
        was_training = self.net.training
        self.net.eval()
        out: list[Evaluation] = []
        # Length-sorted micro-batches: padding falls from ~50% to nearly nothing,
        # which is the measured difference between 312 and 3,758 examples/s.
        order = sorted(range(len(requests)), key=lambda i: len(requests[i][1].word))
        results: list[Evaluation | None] = [None] * len(requests)
        for start in range(0, len(order), self.micro_batch):
            chunk = order[start : start + self.micro_batch]
            batch = build_batch([requests[i] for i in chunk], self.device)
            logits, value, effective = self.net(
                batch["word"],
                batch["lengths"],
                batch["strands"],
                batch["sites"],
                batch["site_mask"],
                batch["budgets"],
            )
            self.forwards += 1
            log_pi = SiteHead.masked_log_policy(logits, effective)
            probability = log_pi.exp().cpu().numpy()
            survival = torch.sigmoid(value).cpu().numpy()
            for row, index in enumerate(chunk):
                count = len(requests[index][2])
                prior = probability[row, :count].astype(np.float64)
                total = prior.sum()
                if not np.isfinite(total) or total <= 0:
                    prior = np.full(count, 1.0 / max(count, 1))
                else:
                    prior = prior / total
                results[index] = Evaluation(prior, float(survival[row]))
        if was_training:
            self.net.train()
        out = [r if r is not None else Evaluation(np.zeros(0), 0.0) for r in results]
        return out


# ---------------------------------------------------------------- the tree


class Node:
    __slots__ = (
        "state",
        "actions",
        "children_states",
        "children",
        "prior",
        "visits",
        "total",
        "dead",
        "count",
        "expanded",
        "parent",
        "parent_index",
    )

    def __init__(self, state: SearchState, parent: Node | None, parent_index: int) -> None:
        self.state = state
        self.parent = parent
        self.parent_index = parent_index
        self.actions: tuple[int, ...] = ()
        self.children_states: tuple[SearchState, ...] = ()
        self.children: list[Node | None] = []
        self.prior = np.zeros(0)
        self.visits = np.zeros(0)
        self.total = np.zeros(0)
        self.dead = np.zeros(0, dtype=bool)
        self.count = 0
        self.expanded = False


@dataclass
class SearchResult:
    instance: Instance
    solved: bool
    witness: UnknotWitness | None
    actions: tuple[int, ...] | None
    simulations: int
    expansions: int
    root: Node | None
    seed: int
    root_selection: str


@dataclass
class _Pending:
    tree: Tree
    node: Node
    child_index: int
    state: SearchState
    actions: tuple[int, ...]
    children: tuple[SearchState, ...]
    path: list[tuple[Node, int]]


class Tree:
    """One MCTS search over one instance.

    Deterministic single-player search, so the transposition guard is the same one
    a best-first beam uses: a child whose `(word, strands)` already exists in this
    tree at no greater crossing-change cost is dropped at expansion.  It is applied
    identically to every arm, which is what makes "at equal simulations" mean the
    same thing for the network arms and for the classical control.
    """

    def __init__(
        self,
        instance: Instance,
        *,
        seed: int,
        c_puct: float = 1.5,
        root_selection: str = "puct",
        gumbel_m: int = 16,
        gumbel_c_visit: float = 50.0,
        gumbel_c_scale: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_fraction: float = 0.0,
        budget: int = 1024,
    ) -> None:
        if root_selection not in ("puct", "gumbel"):
            raise ValueError(f"unknown root selection {root_selection!r}")
        self.instance = instance
        self.spec = instance.spec
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.c_puct = float(c_puct)
        self.root_selection = root_selection
        self.gumbel_m = int(gumbel_m)
        self.gumbel_c_visit = float(gumbel_c_visit)
        self.gumbel_c_scale = float(gumbel_c_scale)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_fraction = float(dirichlet_fraction)
        self.budget = int(budget)

        start = SearchState(instance.word, instance.strands, 0, 0)
        self.root = Node(start, None, -1)
        self.seen: dict[tuple[Word, int], int] = {start.key: 0}
        self.expansions = 0
        self.simulations = 0
        self.solved_path: tuple[int, ...] | None = None
        self.finished = start.key == SOLVED
        self.nodes = 1
        # Gumbel state, built when the root is first expanded.
        self._gumbel: dict | None = None

    # -- selection ---------------------------------------------------------

    def _puct(self, node: Node) -> int:
        total = max(node.count, 1)
        q = np.where(node.visits > 0, node.total / np.maximum(node.visits, 1), 0.0)
        u = self.c_puct * node.prior * math.sqrt(total) / (1.0 + node.visits)
        score = np.where(node.dead, -np.inf, q + u)
        if not np.isfinite(score).any():
            return int(np.argmax(node.visits * 0))
        return int(np.argmax(score))

    def _init_gumbel(self) -> None:
        """Gumbel top-`m` at the root, then sequential halving over the budget.

        Danihelka et al. 2022.  The reason it is here rather than assumed away is
        the one the milestone states: at the simulation counts this project can
        afford, PUCT's `sqrt(N)/(1+n)` term never gets enough visits for the
        exploration bonus to mean anything, and sequential halving spends the same
        budget on a shortlist instead.  Which one wins is measured, not assumed.
        """
        node = self.root
        count = len(node.actions)
        logits = np.log(np.maximum(node.prior, 1e-12))
        gumbel = self.rng.gumbel(size=count)
        m = min(self.gumbel_m, count)
        considered = [int(a) for a in np.argsort(-(logits + gumbel))[:m]]
        self._gumbel = {
            "logits": logits,
            "gumbel": gumbel,
            "alive": considered,
            "phases": max(1, int(math.ceil(math.log2(max(m, 2))))),
            "phase": 0,
            "target": 0,
        }
        self._gumbel["target"] = self._phase_target()

    def _phase_target(self) -> int:
        state = self._gumbel
        assert state is not None
        alive = max(len(state["alive"]), 1)
        share = self.budget / (state["phases"] * alive)
        return int(state["target"]) + max(1, int(share))

    def _gumbel_root(self) -> int:
        assert self._gumbel is not None
        state = self._gumbel
        node = self.root
        alive: list[int] = state["alive"]
        for _ in range(state["phases"] + 2):
            live = [a for a in alive if not node.dead[a]]
            if not live:
                # Everything on the shortlist is a dead end; fall back to PUCT so
                # the budget is not thrown away.
                return self._puct(node)
            if len(live) == 1:
                return live[0]
            under = [a for a in live if node.visits[a] < state["target"]]
            if under:
                return min(under, key=lambda a: node.visits[a])
            max_visit = float(node.visits.max()) if node.count else 0.0
            q = np.where(node.visits > 0, node.total / np.maximum(node.visits, 1), 0.0)
            sigma = (self.gumbel_c_visit + max_visit) * self.gumbel_c_scale * q
            score = state["logits"] + state["gumbel"] + sigma
            live.sort(key=lambda a: -score[a])
            state["alive"] = live[: max(len(live) // 2, 1)]
            alive = state["alive"]
            state["phase"] += 1
            state["target"] = self._phase_target()
        return self._puct(node)

    def select_leaf(self) -> _Pending | None:
        """Descend to an unexpanded child; return what needs evaluating."""
        if self.finished:
            return None
        node = self.root
        path: list[tuple[Node, int]] = []
        while True:
            if not node.expanded:
                # Only the root can be unexpanded on entry.
                return _Pending(
                    self, node, -1, node.state, (), (), path
                )
            if len(node.actions) == 0:
                # Dead end: back up a loss and stop this simulation.
                if node.parent is not None:
                    node.parent.dead[node.parent_index] = True
                self._backup(path, 0.0)
                self.simulations += 1
                return None
            if node is self.root and self.root_selection == "gumbel":
                index = self._gumbel_root()
            else:
                index = self._puct(node)
            path.append((node, index))
            child = node.children[index]
            if child is None:
                return _Pending(
                    self,
                    node,
                    index,
                    node.children_states[index],
                    (),
                    (),
                    path,
                )
            node = child

    # -- expansion ---------------------------------------------------------

    def candidates(
        self, state: SearchState
    ) -> tuple[tuple[int, ...], tuple[SearchState, ...], bool]:
        """Children after the transposition guard, plus "is this the unknot"."""
        if state.key == SOLVED:
            return (), (), True
        actions: list[int] = []
        children: list[SearchState] = []
        for action, child in successors_of(state, self.instance, self.spec):
            if child.key == SOLVED:
                return (action,), (child,), False
            previous = self.seen.get(child.key)
            if previous is not None and previous <= child.used_cc:
                continue
            actions.append(action)
            children.append(child)
        return tuple(actions), tuple(children), False

    def apply(
        self,
        pending: _Pending,
        evaluation: Evaluation | None,
        solved_value: float = 1.0,
    ) -> None:
        node = pending.node
        index = pending.child_index
        state = pending.state
        if index >= 0:
            child = Node(state, node, index)
            node.children[index] = child
        else:
            child = node
        self.nodes += 1
        self.expansions += 1
        self.simulations += 1
        self.seen[state.key] = min(self.seen.get(state.key, 10**9), state.used_cc)

        if state.key == SOLVED:
            child.expanded = True
            self._record_solution(child)
            self._backup(pending.path, solved_value)
            return

        actions, children, _ = pending.actions, pending.children, False
        child.actions = actions
        child.children_states = children
        child.children = [None] * len(actions)
        count = len(actions)
        if evaluation is None or count == 0:
            child.prior = np.full(count, 1.0 / max(count, 1))
            value = 0.0
        else:
            prior = np.asarray(evaluation.priors, dtype=np.float64)[:count]
            if prior.size != count or not np.isfinite(prior).all() or prior.sum() <= 0:
                prior = np.full(count, 1.0 / max(count, 1))
            else:
                prior = prior / prior.sum()
            child.prior = prior
            value = float(evaluation.value)
        child.visits = np.zeros(count)
        child.total = np.zeros(count)
        child.dead = np.zeros(count, dtype=bool)
        child.expanded = True
        if count == 0 and child.parent is not None:
            # A dead end must stop competing for the budget, or PUCT re-selects it
            # forever at `Q = 0` and the arm is measured on a search that spent its
            # simulations re-discovering the same wall.
            child.parent.dead[child.parent_index] = True

        if child is self.root:
            if self.dirichlet_fraction > 0 and count:
                noise = self.rng.dirichlet(np.full(count, self.dirichlet_alpha))
                child.prior = (
                    1 - self.dirichlet_fraction
                ) * child.prior + self.dirichlet_fraction * noise
            if self.root_selection == "gumbel":
                self._init_gumbel()

        # An immediate terminal child is taken at once: the search has the witness.
        for position, candidate in enumerate(children):
            if candidate.key == SOLVED:
                terminal = Node(candidate, child, position)
                terminal.expanded = True
                child.children[position] = terminal
                self._record_solution(terminal)
                self._backup(pending.path + [(child, position)], solved_value)
                return
        self._backup(pending.path, value)

    def _record_solution(self, node: Node) -> None:
        actions: list[int] = []
        current = node
        while current.parent is not None:
            actions.append(current.parent.actions[current.parent_index])
            current = current.parent
        self.solved_path = tuple(reversed(actions))
        self.finished = True

    def _backup(self, path: Sequence[tuple[Node, int]], value: float) -> None:
        for node, index in path:
            node.visits[index] += 1
            node.total[index] += value
            node.count += 1


def run_searches(
    instances: Sequence[Instance],
    evaluator: Evaluator,
    *,
    simulations: int,
    seeds: Sequence[int] = (0,),
    root_selection: str = "puct",
    c_puct: float = 1.5,
    gumbel_m: int = 16,
    dirichlet_fraction: float = 0.0,
    time_cap: float | None = None,
    log_every: float = 15.0,
    log: bool = False,
    label: str = "",
) -> list[SearchResult]:
    """Run every `(instance, seed)` tree in lockstep, one batched evaluation a step.

    Lockstep is what makes the network affordable: every tree contributes one leaf
    per simulation, so a step is a single forward over the whole cohort rather than
    one forward per tree.
    """
    trees = [
        Tree(
            instance,
            seed=seed * 9973 + index,
            c_puct=c_puct,
            root_selection=root_selection,
            gumbel_m=gumbel_m,
            dirichlet_fraction=dirichlet_fraction,
            budget=simulations,
        )
        for index, instance in enumerate(instances)
        for seed in seeds
    ]
    owners = [
        (instance, seed)
        for instance in instances
        for seed in seeds
    ]
    start = time.monotonic()
    last_log = start
    for step in range(simulations):
        pendings: list[_Pending] = []
        for tree in trees:
            if tree.finished or tree.simulations >= simulations:
                continue
            pending = tree.select_leaf()
            if pending is None:
                continue
            actions, children, is_solved = tree.candidates(pending.state)
            pending.actions = actions
            pending.children = children
            pendings.append(pending)
        if not pendings:
            break
        requests = [
            (p.tree.instance, p.state, p.actions)
            for p in pendings
            if p.state.key != SOLVED and p.actions
        ]
        evaluations: dict[int, Evaluation] = {}
        if requests:
            results = evaluator.evaluate(requests)
            cursor = 0
            for position, p in enumerate(pendings):
                if p.state.key != SOLVED and p.actions:
                    evaluations[position] = results[cursor]
                    cursor += 1
        for position, p in enumerate(pendings):
            p.tree.apply(p, evaluations.get(position))
        now = time.monotonic()
        if log and now - last_log >= log_every:
            done = sum(t.finished for t in trees)
            print(
                f"[{label}] sim {step + 1}/{simulations} solved {done}/{len(trees)} "
                f"{now - start:.0f}s",
                flush=True,
            )
            last_log = now
        if time_cap is not None and now - start > time_cap:
            if log:
                print(f"[{label}] time cap {time_cap:.0f}s hit at sim {step + 1}", flush=True)
            break

    out: list[SearchResult] = []
    for tree, (instance, seed) in zip(trees, owners, strict=True):
        witness = None
        actions = tree.solved_path
        if actions is not None:
            witness = UnknotWitness.from_actions(
                instance.word, instance.strands, instance.spec, actions
            )
            witness.verify()
        out.append(
            SearchResult(
                instance=instance,
                solved=actions is not None,
                witness=witness,
                actions=actions,
                simulations=tree.simulations,
                expansions=tree.expansions,
                root=tree.root,
                seed=seed,
                root_selection=root_selection,
            )
        )
    return out


# ---------------------------------------------------------------- classical arm


def classical_beam(
    instance: Instance,
    *,
    simulations: int,
    seed: int = 0,
) -> SearchResult:
    """Best-first on `(crossing changes used, word length)`, the M0.3 control.

    One expansion here is one expansion there, so the budget currency is the same
    as the tree search's `simulations`.  The random tie-break is what makes the
    arm seed-dependent, which is what `witness_rate@(cap, dose, seeds)` requires.
    """
    import heapq

    spec = instance.spec
    rng = np.random.default_rng(seed)
    start = SearchState(instance.word, instance.strands, 0, 0)
    seen = {start.key: 0}
    heap: list[tuple[int, int, float, int, SearchState, tuple[int, ...]]] = [
        (0, len(start.word), 0.0, 0, start, ())
    ]
    counter = 1
    expansions = 0
    while heap and expansions < simulations:
        _, _, _, _, state, path = heapq.heappop(heap)
        expansions += 1
        for action, child in successors_of(state, instance, spec):
            if child.key == SOLVED:
                actions = path + (action,)
                witness = UnknotWitness.from_actions(
                    instance.word, instance.strands, spec, actions
                )
                witness.verify()
                return SearchResult(
                    instance, True, witness, actions, expansions, expansions, None, seed, "beam"
                )
            previous = seen.get(child.key)
            if previous is not None and previous <= child.used_cc:
                continue
            seen[child.key] = child.used_cc
            heapq.heappush(
                heap,
                (
                    child.used_cc,
                    len(child.word),
                    float(rng.random()),
                    counter,
                    child,
                    path + (action,),
                ),
            )
            counter += 1
    return SearchResult(instance, False, None, None, expansions, expansions, None, seed, "beam")


# ---------------------------------------------------------------- training data


@dataclass
class PolicySample:
    """One state, its candidate sites, and what the loop learned to want there."""

    instance: Instance
    state: SearchState
    actions: tuple[int, ...]
    visits: np.ndarray
    positives: tuple[int, ...] = ()
    value_target: float | None = None


def samples_from_tree(
    result: SearchResult,
    *,
    min_visits: int = 8,
    max_nodes: int = 64,
) -> list[PolicySample]:
    """Visit-count policy targets and observed-survival value targets from one tree.

    The value target is the honest one for `p(solve within the remaining budget)`:
    did this subtree actually reach the unknot inside the budget it had.
    """
    if result.root is None:
        return []
    # The node -> action map along the witness, by depth.  Looking the action up by
    # membership instead would pick whichever of the witness's actions happens to
    # be legal here, which is a different site at a different depth.
    on_path: dict[int, int | None] = {}
    if result.solved and result.actions is not None:
        node: Node | None = result.root
        for action in result.actions:
            if node is None or action not in node.actions:
                break
            on_path[id(node)] = action
            index = node.actions.index(action)
            node = node.children[index]
        if node is not None:
            on_path.setdefault(id(node), None)

    samples: list[PolicySample] = []
    stack = [result.root]
    # Witness nodes first: they are the ones carrying `A_plus`, and `max_nodes`
    # must not drop them in favour of an arbitrary branch.
    ordered: list[Node] = []
    while stack:
        node = stack.pop()
        if not node.expanded or len(node.actions) == 0:
            continue
        ordered.append(node)
        for child in node.children:
            if child is not None:
                stack.append(child)
    ordered.sort(key=lambda n: (id(n) not in on_path, -n.count))
    for node in ordered[:max_nodes]:
        if node.count < min_visits and id(node) not in on_path:
            continue
        action = on_path.get(id(node))
        positives = (action,) if action is not None else ()
        samples.append(
            PolicySample(
                instance=result.instance,
                state=node.state,
                actions=node.actions,
                visits=node.visits.copy(),
                positives=positives,
                value_target=1.0 if id(node) in on_path and result.solved else 0.0,
            )
        )
    return samples


def samples_from_witness(
    witness: UnknotWitness,
    *,
    name: str,
    cap_moves: int = 64,
    growth_len: int = 2,
    growth_strands: int = 1,
    include_commutation: bool = True,
) -> list[PolicySample]:
    """Set-valued `A_plus` supervision from any replay-verified witness.

    `rf_knots.aplus.spine_from_witness` is deliberately written to re-index "any
    verified witness, whoever produced it", so `unknotdb`'s replay-verified paths
    and the fast-6 cohort's evidence index feed this on exactly the same footing
    as the generator's own.
    """
    from rf_knots.aplus import commutation_known_good, spine_from_witness

    witness.verify()
    spine = spine_from_witness(witness)
    out: list[PolicySample] = []
    for depth, site in enumerate(spine.sites):
        state = site.state
        instance = Instance(
            name=f"{name}@{depth}",
            word=state.word,
            strands=state.strands,
            # The budget the completion was actually demonstrated under.  Handing
            # this state a larger budget would claim the site is good under a
            # budget nobody exhibited a completion for.
            cap_crossing_changes=int(site.remaining_crossing_changes),
            cap_moves=min(int(site.remaining_moves), cap_moves),
            growth_len=growth_len,
            growth_strands=growth_strands,
            source="witness",
            uid=f"{name}@{depth}",
        )
        spec = instance.spec
        search_state = SearchState(state.word, state.strands, 0, 0)
        candidates = [action for action, _ in successors_of(search_state, instance, spec)]
        # Action ids are `ActionSpec`-relative, and the witness carries its own
        # spec.  Re-encoding through the semantic `(kind, position, generator,
        # sign)` tuple is the only correct way across the two layouts -- and it is
        # exactly the round trip §M1 relies on.
        raw = {site.action}
        if include_commutation:
            try:
                for good in commutation_known_good(spine, depth):
                    raw.add(good.action)
            except Exception:  # pragma: no cover - the closure is an optimisation
                pass
        positives = {spec.encode(*spine.spec.decode(action)) for action in raw}
        positives &= set(candidates)
        if not positives or not candidates:
            continue
        out.append(
            PolicySample(
                instance=instance,
                state=search_state,
                actions=tuple(candidates),
                visits=np.zeros(len(candidates)),
                positives=tuple(sorted(positives)),
                value_target=1.0,
            )
        )
    return out


# ---------------------------------------------------------------- training


@dataclass
class LossReport:
    policy: float
    aplus: float
    value: float
    total: float
    rows: int
    aplus_rows: int


def training_loss(
    net: SitePolicyValueNet,
    samples: Sequence[PolicySample],
    *,
    device: torch.device | str = "cpu",
    value_weight: float = 1.0,
    aplus_weight: float = 1.0,
    policy_weight: float = 1.0,
) -> tuple[torch.Tensor, LossReport]:
    """Visit-count cross-entropy, the set-valued `A_plus` term, and value BCE.

    The `A_plus` term is `site_head.set_valued_loss` verbatim.  Its signature is
    the enforcement of §5.3: the only masks are "legal" and "known good", so no
    caller can express "bad", and a site the witness did not take carries only its
    own share of the normaliser -- it is never pushed down on its own account.
    """
    requests = [(s.instance, s.state, s.actions) for s in samples]
    batch = build_batch(requests, device)
    logits, value, effective = net(
        batch["word"],
        batch["lengths"],
        batch["strands"],
        batch["sites"],
        batch["site_mask"],
        batch["budgets"],
    )
    log_pi = SiteHead.masked_log_policy(logits, effective)

    rows, max_sites = logits.shape
    visit_target = torch.zeros((rows, max_sites), dtype=logits.dtype, device=logits.device)
    visit_rows = torch.zeros(rows, dtype=torch.bool, device=logits.device)
    positive_mask = torch.zeros((rows, max_sites), dtype=torch.bool, device=logits.device)
    value_target = torch.zeros(rows, dtype=logits.dtype, device=logits.device)
    value_rows = torch.zeros(rows, dtype=torch.bool, device=logits.device)
    effective_cpu = effective.detach().cpu().numpy()
    for index, sample in enumerate(samples):
        total = float(sample.visits.sum())
        if total > 0:
            usable = np.zeros(max_sites)
            usable[: len(sample.visits)] = sample.visits
            usable = usable * effective_cpu[index]
            if usable.sum() > 0:
                visit_target[index] = torch.tensor(
                    usable / usable.sum(), dtype=logits.dtype, device=logits.device
                )
                visit_rows[index] = True
        for action in sample.positives:
            try:
                position = sample.actions.index(action)
            except ValueError:
                continue
            if effective_cpu[index, position]:
                positive_mask[index, position] = True
        if sample.value_target is not None:
            value_target[index] = float(sample.value_target)
            value_rows[index] = True

    zero = logits.new_zeros(())
    if bool(visit_rows.any()):
        per_row = -(visit_target * torch.where(log_pi.isfinite(), log_pi, torch.zeros_like(log_pi)))
        policy = per_row.sum(-1)[visit_rows].mean()
    else:
        policy = zero
    if bool(positive_mask.any()):
        aplus = set_valued_loss(logits, effective, positive_mask & effective, reduction="mean")
    else:
        aplus = zero
    if bool(value_rows.any()):
        value_loss = F.binary_cross_entropy_with_logits(
            value[value_rows], value_target[value_rows]
        )
    else:
        value_loss = zero

    total_loss = policy_weight * policy + aplus_weight * aplus + value_weight * value_loss
    report = LossReport(
        policy=float(policy.detach()),
        aplus=float(aplus.detach()),
        value=float(value_loss.detach()),
        total=float(total_loss.detach()),
        rows=len(samples),
        aplus_rows=int(positive_mask.any(-1).sum()),
    )
    return total_loss, report


def train_on(
    net: SitePolicyValueNet,
    samples: Sequence[PolicySample],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int = 24,
    epochs: int = 1,
    device: torch.device | str = "cpu",
    seed: int = 0,
    grad_clip: float = 1.0,
    log: bool = False,
    log_every: float = 15.0,
    label: str = "",
) -> list[LossReport]:
    rng = np.random.default_rng(seed)
    reports: list[LossReport] = []
    net.train()
    last_log = time.monotonic()
    for epoch in range(epochs):
        order = rng.permutation(len(samples))
        # Length-sorted inside each shuffled chunk: 49.9% padding down to 0.4%.
        for start in range(0, len(order), batch_size):
            chunk = sorted(
                order[start : start + batch_size],
                key=lambda i: len(samples[i].state.word),
            )
            if not len(chunk):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, report = training_loss(net, [samples[i] for i in chunk], device=device)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
            reports.append(report)
            now = time.monotonic()
            if log and now - last_log >= log_every:
                print(
                    f"[{label}] epoch {epoch + 1}/{epochs} batch {start // batch_size} "
                    f"loss {report.total:.4f} (policy {report.policy:.4f} "
                    f"aplus {report.aplus:.4f} value {report.value:.4f})",
                    flush=True,
                )
                last_log = now
    net.eval()
    return reports


# ---------------------------------------------------------------- metrics


def witness_rate(results: Sequence[SearchResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.solved) / len(results)


def per_instance_rate(results: Sequence[SearchResult]) -> dict[str, float]:
    grouped: dict[str, list[bool]] = {}
    for result in results:
        grouped.setdefault(result.instance.label, []).append(result.solved)
    return {name: sum(v) / len(v) for name, v in grouped.items()}


def verify_all(results: Sequence[SearchResult]) -> tuple[int, int]:
    """Replay every claimed witness.  Returns `(claimed, verified)`."""
    claimed = 0
    verified = 0
    for result in results:
        if not result.solved or result.witness is None:
            continue
        claimed += 1
        result.witness.verify()
        # A second replay through JSON, which is the round trip M0.3 used.
        UnknotWitness.from_dict(json.loads(json.dumps(result.witness.to_dict()))).verify()
        verified += 1
    return claimed, verified


# ---------------------------------------------------------------- corpora


def generated_witness_instances(
    *,
    crossing_changes: int = 1,
    walk_length: int = 14,
    settle_walk_length: int = 4,
    target_strands: int = 3,
    target_length: int = 10,
    max_len: int = 14,
    max_strands: int = 4,
    strand_band: int = 1,
    seed: int = 7,
    count: int = 256,
    cap_moves: int = 32,
    growth_len: int = 2,
    growth_strands: int = 1,
) -> list[tuple[Instance, UnknotWitness]]:
    """Backward-generated instances with the witness that certifies them.

    Tier 1 of §5.3b and the base case: the walk is built from the answer, so the
    reverse path *is* a verified witness and the action it takes at each state is
    known-good by construction.  Measured here at ~1 ms an instance.
    """
    from rf_knots.interleaved_generator import InterleavedConfig, generate

    config = InterleavedConfig(
        crossing_changes=crossing_changes,
        walk_length=walk_length,
        settle_walk_length=settle_walk_length,
        target_strands=target_strands,
        target_length=target_length,
        max_len=max_len,
        max_strands=max_strands,
        strand_band=strand_band,
    )
    out: list[tuple[Instance, UnknotWitness]] = []
    for index in range(count):
        generated = generate(config, seed=seed, index=index)
        witness = generated.witness
        if witness is None:
            continue
        witness.verify()
        upper = witness.crossing_changes
        instance = Instance(
            name=f"gen{index:04d}",
            word=tuple(generated.word),
            strands=int(generated.strands),
            cap_crossing_changes=int(upper),
            cap_moves=cap_moves,
            growth_len=growth_len,
            growth_strands=growth_strands,
            source="interleaved",
            uid=f"interleaved:{seed}:{index:04d}",
        )
        out.append((instance, witness))
    return out


def evidence_index_witnesses(
    path: str | Path,
    *,
    exclude_names: Sequence[str] = (),
    exclude_keys: Sequence[tuple[Word, int]] = (),
    limit: int | None = None,
) -> list[tuple[str, UnknotWitness]]:
    """The fast-6 cohort's replay-verified witnesses, re-indexed for `A_plus`.

    Note 20's evidence index came from these scientists and
    `aplus.spine_from_witness` is deliberately written to re-index "any verified
    witness, whoever produced it", so they enter supervision on exactly the same
    footing as the generator's own.  Every one is replayed here before it is used;
    the index's own `replay_verified` flag is not taken on trust.

    `exclude_names` and `exclude_keys` are the leakage guard.  The index and the
    curriculum banks come from the same corpus, so an entry naming an evaluation
    rung's knot -- or starting from its exact word -- is dropped.
    """
    payload = json.loads(Path(path).read_text())
    blocked_names = {str(n) for n in exclude_names}
    blocked_keys = {(tuple(word), int(strands)) for word, strands in exclude_keys}
    out: list[tuple[str, UnknotWitness]] = []
    for knot, record in sorted(payload.get("knots", {}).items()):
        if str(knot) in blocked_names:
            continue
        scientists = record.get("scientists")
        if not isinstance(scientists, dict):
            continue
        for scientist, entry in sorted(scientists.items()):
            actions = entry.get("semantic_actions")
            start = entry.get("start")
            spec_row = entry.get("action_spec")
            if not actions or not start or not spec_row:
                continue
            word = tuple(int(x) for x in start["word"])
            strands = int(start["strands"])
            if (word, strands) in blocked_keys:
                continue
            spec = ActionSpec(int(spec_row["max_len"]), int(spec_row["max_strands"]))
            try:
                witness = UnknotWitness.from_actions(word, strands, spec, actions)
            except ValueError:
                continue
            out.append((f"{knot}/{scientist}", witness))
            if limit is not None and len(out) >= limit:
                return out
    return out
