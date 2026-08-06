# 17 — Text-mediated sharing: invented language, invented theory, and a solver that reads

[16](16-scientists-collaboration.md) shares a *solution*: one verified witness for one
representation, translated into the receiver's action space and used as a training target.
The proposal here is different in kind. The scientists should invent a **language**, write a
**theory** in it, and a solver should read the accumulated theory — not a hint about the knot
in front of it — and thereby unknot representations nobody has seen. In the limit, the theory
should be able to contain something like the Jones polynomial.

This note takes that seriously enough to say which parts are testable now, which part is a new
architecture rather than a new config flag, and in what order the cheap falsifications should
be run.

The short verdict is:

* this is **not** a stronger version of witness sharing; it replaces a training-target
  mechanism with an inference-time one, and that is its main advantage;
* the decisive argument in its favour comes from this project's own results: every failure in
  [16](16-scientists-collaboration.md) since 2026-08-03 is a failure of the *update rule*, and
  reading a document requires no update at all;
* the ambition decomposes into four capabilities that must be gated separately — invented
  language, verifiable theory, a solver that reads, and invented invariants — because three of
  them can succeed while the fourth silently carries the result;
* the first experiment must be a **human-written theory read by a reader**. If a good
  hand-written theory does not help, no agent-invented theory will;
* a "theory" is only distinguishable from a hint if it comes with a **refutation procedure**.
  The project has a verifier for solutions; it has no falsifier for statements, and that is the
  missing component;
* knot invariants are **constant along the simplification trajectory** ([10 §2](10-invariants-and-representations.md)),
  so "invent Jones" and "unknot better" are connected only through crossing-change/unknotting-number
  mode — they are not the same experiment;
* under the zero-human-knowledge constraint, agent-written text is a fair channel and human
  text is an oracle arm; both are worth running, only one is the headline; and
* the current K=3 roster cannot read anything. Requirement (4) is a different network class and
  a different compute tier.

## 1. What is actually being asked for

Four capabilities, deliberately separated because they fail independently:

| | capability | the artifact | how it can be falsified |
|---|---|---|---|
| `L1` | invent a language | a vocabulary of named, verified objects (macros, tactics, predicates, features) | compression of the witness archive; reuse across *different* authors; survives compute matching |
| `L2` | invent a theory | statements quantified over a *class* of representations, not one instance | an automated falsifier finds no counterexample in a declared search budget |
| `L3` | read the theory | a solver whose input is `(document, representation)` | paired coverage on identity-disjoint knots, same weights, document present vs. absent |
| `L4` | invent invariants | a computable function of the braid word with a bounded jump across `Σ₁` | the Lipschitz test on self-play distance-1 pairs; agreement with a known invariant on a held-out table slice |

`L3` is the load-bearing one and the only one that delivers the user-visible property —
*unknot a knot nobody has seen, by reading*. `L1` and `L2` without `L3` produce a nice archive
that nothing consumes. `L3` without `L1`/`L2` is still valuable: it is a solver that can be
told things.

The failure mode to design against is that `L1`–`L3` all report success while the gain comes
entirely from the extra search compute spent generating and validating the documents. The
compute-matched re-evaluation of TroVE and LEGO-Prover
([arXiv:2507.22069](https://arxiv.org/pdf/2507.22069)) found exactly this: a growing library of
reusable lemmas showed **no clear evidence of reuse or gains once compute was matched**. Every
gate below is therefore stated against a compute-matched control, not against the initial
checkpoint.

## 2. Why text, given what [16](16-scientists-collaboration.md) has measured

The argument is not aesthetic. It follows from the recorded failures.

As of 2026-08-05 the position is: replay-v3 admitted, the `s-window-128` ordering critic
admitted, split-success-24 admitted as a *consolidation* update that "did not invent a solution
on any zero-positive task", and **bounded-option sharing not admitted** — two exactly translated
witnesses produced no new solved identity and made `s-w11-128` lose `12a_146`. Increasing
preservation weight, reducing dose, sampling one position, training one whole witness, and
lowering the option learning rate are all recorded as rejected repairs.

Read that as a class: *every mechanism for moving knowledge between scientists so far has been a
weight update, and weight updates in this system destroy as much as they add.* The earlier
policy-update diagnostic named the mechanism precisely — destructive imitation of failure-only
search — and the best sharing variant ever measured, `adaptive-sharing-aux-only`, was the one
that shared **information** (solve supervision, one-sided cost bounds) while masking
**imitation** (policy and scalar-value targets).

A document read at inference time is the limit of that direction. It cannot forget `12a_146`,
because it does not touch the weights that solve `12a_146`.

This is also the consensus direction in the current literature, arrived at from the other side:

* [GEPA (ICLR 2026 oral)](https://arxiv.org/abs/2507.19457) reflects on trajectories in natural
  language and writes the lesson into the prompt, beating GRPO by ~10% with up to 35× fewer
  rollouts, on the explicit argument that language is a richer medium than a scalar reward.
* [Artificial Generational Intelligence](https://arxiv.org/pdf/2406.00392) (NeurIPS 2024)
  separates in-context accumulation ("knowledge") from in-weights accumulation ("skill") and
  shows accumulating agents beat single-lifetime agents at equal cumulative experience. `L3` is
  the in-context branch; [16](16-scientists-collaboration.md) has only ever run the in-weights one.
* The 2026 experience-library cluster —
  [continual experience internalization](https://arxiv.org/pdf/2606.04703),
  [experiential reflective learning](https://arxiv.org/html/2603.24639),
  [R²-Mem](https://arxiv.org/pdf/2605.13486),
  [CODESKILL](https://arxiv.org/pdf/2605.25430),
  survey: [From Storage to Experience](https://aclanthology.org/2026.findings-acl.2069.pdf) —
  converges on writing *actionable lessons* rather than trajectories, and repeatedly finds
  component-level evolution succeeding where policy-level updates regress.

Caveat: all of the above run on language models, whose weights already encode a reader. That is
the whole cost of this programme, and §3 is about paying it.

## 3. `L3` — the reader is the hard part

### 3.1 The current roster cannot do this

`s-window-128` sees seven cells and 102,439 parameters. There is no input through which a
theory could enter, and no architectural reason to expect one to help if bolted on. Conditioning
on a document requires attention over a variable-length symbol sequence that is not the braid
word — a second stream, and a mechanism relating the two.

The honest precedent is **RTFM** ([arXiv:1910.08210](https://arxiv.org/abs/1910.08210), ICLR
2020): an RL agent that jointly reasons over a goal, a *document describing the environment
dynamics*, and observations, and generalises to **dynamics never seen in training, via reading**.
Its `txt2π` model is built around three-way goal/document/observation interaction. That is
requirement (4), stated a decade early, and it worked at a scale far below an LLM. It is the
right thing to copy, not GPT.

### 3.2 Learn to read before there is anything worth reading

The bootstrapping trap: an agent-invented theory is worthless until a reader exists, and a
reader cannot be trained on documents that do not exist yet. RTFM's answer is the right one —
**procedurally generate the documents**.

Concretely, and buildable against the existing environment:

1. Define a family of *sound but arbitrary* statements about braid words that the existing
   reference semantics can check — e.g. "when the top generator appears exactly once, destabilise
   first"; "prefer commutation before R2 when the word has more than `k` sign changes".
2. Sample a document = a set of such statements. Sample a task family for which the document's
   statements are *true and useful*, and a matched family for which they are *irrelevant*.
3. Train `(document, representation) → policy`. The measurable target is not solve rate but
   **the gap**: performance with the correct document minus performance with a shuffled or
   adversarial one, on held-out documents.

If that gap is zero on procedurally generated theory, the programme stops here, cheaply, and it
stops for a reason that is about architecture rather than about the scientists.

### 3.3 Two tiers, not one replacement

Do not rebuild the roster. Run the reader as a **separate scientist** in the same population:

| tier | who | reads | writes | update rule |
|---|---|---|---|---|
| 1 | existing K=3 serial scientists | nothing | verified witnesses | split-success-24, replay-v3 (admitted) |
| 2 | one reader-solver | the theory document + representation | candidate statements | its own; isolated from tier 1 |

Tier 1 keeps producing the raw evidence — witnesses, failures, calibrated solve curves — which
is exactly the corpus a theory has to be *about*. Tier 2 consumes it. Nothing in tier 1 changes,
so no admitted result is put at risk, and the compute-matched control for tier 2 is
straightforward: give tier 1 the reader's total network evaluations.

### 3.4 Cost realism

[05](05-compute-budget.md) and the measured batch-1 profile are the constraint: MCTS here is
latency bound at 605 µs per simulation with 77% of it in the forward pass of a 48K network. A
document encoder in the MCTS leaf evaluation multiplies the dominant cost by whatever the
encoder costs, on every simulation. Two mitigations, both required before any reader run:

* **encode the document once per task**, not per leaf — cache the document representation and
  let only the cross-attention query vary; and
* declare the reader's simulation budget in *measured network evaluations and wall time*, as
  [16 §8](16-scientists-collaboration.md) already requires, because "same simulations" will not
  mean the same compute across tiers.

## 4. `L2` — what makes a theory more than a long hint

The user's requirement (3) is the sharp one: use the **whole theory**, not a hint about one
knot. The property that distinguishes them is not length or abstraction. It is this:

> A hint is verified by solving one instance. A theory is a statement over a class, and it is
> only meaningful if there is a procedure that could **refute** it.

`rf-knots` has a verifier for solutions — an exactly replayed trajectory ending at the empty
word with one strand. It has no falsifier for statements. That component is the actual
deliverable of `L2`, and it is ordinary engineering rather than research:

```text
Statement = (schema, parameters, scope, budget)

falsify(statement) -> {refuted(counterexample) | not_refuted(budget, n_tested) }
```

Statement schemas that the current environment can already test, in increasing strength:

| schema | example | falsifier |
|---|---|---|
| move preference | "on words with property `P`, action `a` reduces length within 3 moves" | enumerate/sample words with `P`; replay |
| macro validity | "`M` = this fixed sequence is an isotopy on any word matching pattern `Q`" | apply `M`, compare via reference semantics |
| ordering | "if `P(w)` then `u`-cost of route A ≤ route B" | paired search under matched budget |
| invariance | "`f(w)` is unchanged by every Markov/isotopy move" | random walk in the move graph, compare `f` |
| bound | "`f(w) ≤ u(K_w)`" — i.e. a 1-Lipschitz claim | distance-1 pairs from self-play ([10 §1B](10-invariants-and-representations.md)) |

Two rules keep this honest. **Scope must be explicit and bounded** — a statement that quantifies
over all braid words is untestable, so every statement carries the envelope it is claimed on
(strands, length) and the budget under which it survived. And **`not_refuted` is not `true`**;
the archive stores the tested count, exactly as [16](16-scientists-collaboration.md) stores
`budget_exhausted` separately from failure rather than collapsing them.

The last two rows are where this stops being bookkeeping. An *invariance* statement is a
conjecture that a learned function is a knot invariant; a *bound* statement is a conjecture that
it is an unknotting lower bound. Those are the objects of `L4`.

## 5. `L1` — invented language, and how to tell it is real

The nearest working system is **DreamProver**
([arXiv:2604.26311](https://arxiv.org/abs/2604.26311)): wake phase proves, sleep phase abstracts
recurring proof fragments into named, transferable lemmas, and later problems reuse them —
DreamCoder's compression loop applied to theorem proving. The knot analogue is direct: mine the
witness archive for recurring braid-word rewrite motifs, name them, and let them be referred to.

This solves a problem [16 §6](16-scientists-collaboration.md) has been fighting head-on. A
witness is currently a semantic word-state path that must be *routed* into each receiver's head,
tape, and window actions — and bounded-option distillation of exactly that routing is what
failed. A named macro over braid words has no author-side action space in it at all. It is
defined by what it does to the word, so there is nothing to translate.

What to measure, in this order, all compute-matched:

1. **Compression.** Does the archive get shorter when rewritten in the vocabulary? Description
   length is the cheapest non-circular signal and needs no training run.
2. **Cross-author reuse.** A term invented from `s-window-128` witnesses that appears in
   `s-tape4` and `s-w11-128` witnesses is evidence of a shared language. A term used only by its
   author is private notation.
3. **Transfer.** Coverage on identity-disjoint knots with the vocabulary available as macro
   actions, against a control given the same number of *random* macros — this controls for
   "longer moves help", which they will.

Note the last control carefully. Macro-actions shorten the search horizon regardless of whether
they mean anything, so a raw improvement is not evidence of language.

There is a second, weaker sense of "invent a language" — a learned discrete message channel
between the small nets ([emergent discrete messages](https://arxiv.org/pdf/2102.12550),
[language-grounded MARL](https://arxiv.org/html/2409.17348v1)). It is not recommended here. Each
attempt in this system is single-agent with a verifier at the end, so a learned message has no
bandwidth advantage over handing over the verified artifact, and it would add a learned channel
to networks whose update rule has only just been stabilised.

## 6. `L4` — invented invariants, and the constraint that decides where they matter

Requirement (5) — invent something like the Jones polynomial — is already half-designed in this
repository, and its main obstacle is already recorded.

**The obstacle.** [10 §2](10-invariants-and-representations.md): in the simplification game the
closure is the unknot at every state by construction, so `V(t) ≡ 1`, `σ ≡ 0`, and every knot
invariant is constant along the entire trajectory. An invented invariant carries **exactly zero**
information about progress in the game as built. Therefore:

> `L4` cannot be evaluated by "does it unknot better" in the simplification game. It is
> evaluated in crossing-change/unknotting-number mode, where invariants are lower bounds and
> transposition keys — or not at all.

This is the single most important thing to state before the experiment is designed, because
"the scientists invented Jones and it improved solve rate" is not a result the current game can
produce, and an apparent version of it would be a bug.

**The honest role**, from [10 §1B](10-invariants-and-representations.md): not "predict `u(K)`"
but *search for a computable function whose jump across `Σ₁` is bounded*, then have the jump
bound proved. Self-play supplies the training data for free — every crossing change is a
labelled distance-1 pair.

**A concrete, gradeable Jones milestone.** [12 §3](12-serial-formulation.md) already establishes
that `TL₅` decomposes into irreducibles of dimension 1, 4, and 5, and that the 4-dimensional
block *is* the reduced Burau representation — so a 4×4 carrier reaches Alexander and one of
Jones's three blocks, not Jones. That converts a vague ambition into a measurable target:

```text
milestone L4-a: an accumulator whose transported state, evaluated on a held-out
                slice of the knot table, separates pairs that Alexander/Burau
                confuses and Jones does not
milestone L4-b: that state is a faithful carrier of the missing 5-dimensional block
```

`L4-a` is testable today against `rf_knots.invariants` and `rf_knots.knot_table` on held-out
identities. It requires no theory, no language, and no reader — which makes it the one part of
this programme that can be run in parallel with everything else.

**Zero human knowledge.** Under the standard in the [README](README.md), a *statement written by
an agent and verified in-system* is a fair channel: nothing enters that the system did not
derive. A human-written theory is an oracle arm — run it to bound the upside, quote it as an
oracle, never as the headline. That is the same rule [10](10-invariants-and-representations.md)
already applies to computed invariants as inputs, and §3.2's procedurally generated documents
sit on the oracle side too. They are training scaffolding for the reader, not evidence about
the scientists.

## 7. Relation to the blocked programme

None of this unblocks [16](16-scientists-collaboration.md). CPU-32, the 200-representation
factorial, and the 2,700 run stay closed on their own gates, and the roster comparison must not
absorb a fifth intervention while sharing itself is unadmitted.

But the dependency runs one way only, and that is the useful observation:

* `L4-a` needs nothing from [16](16-scientists-collaboration.md) — it is an offline invariant
  study over the table and the ladder.
* `L1` step 1 (compression of the existing witness archive) is offline analysis of artifacts
  that already exist.
* `L3` is trained on procedurally generated documents, in tier 2, isolated from the tier-1
  weights that carry every admitted result.

So the theory programme can start on the parts that cannot damage the blocked comparison, and
its central claim — that in-context transfer avoids the destructive-update failure — is exactly
the claim the blocked comparison cannot test.

## 8. Gates, cheapest first

Each gate is preregistered with its own kill criterion. None of them authorises the next.

1. **Archive compression gate (`L1`, offline, hours).** Mine recurring rewrite motifs from
   existing witnesses. Report description-length reduction, the number of motifs used by more
   than one author, and how many recorded *failures* a motif library would have covered. Kill if
   no motif crosses authors.
2. **Falsifier gate (`L2`, engineering).** Implement `falsify(statement)` for the five schemas in
   §4, with scope and budget recorded. Regression: seed it with statements known true and known
   false; require it to refute every false one within budget. No research claim.
3. **Reader-comprehension gate (`L3`, decisive, cheap).** Procedurally generated documents,
   RTFM-style. Primary endpoint is the *paired* gap: correct document vs. shuffled document, on
   held-out documents and identity-disjoint knots. Kill the whole programme if the gap is not
   significant across seeds — architecture, not scientists, and no amount of invented theory
   fixes it.
4. **Oracle-theory gate (`L3` + `L2`, cheap).** One hand-written theory of genuine knot-theoretic
   content, read by the gate-3 reader. This bounds the upside of *any* invented theory. If a
   correct human theory does not move held-out coverage, the ceiling is zero.
5. **Invariant-separation gate (`L4-a`, parallel).** Held-out table slice; does the learned
   carrier separate what Burau cannot? Compare against a random-projection control of equal
   dimension.
6. **Closed loop (expensive, last).** Tier-1 scientists produce witnesses → statements are mined
   and falsified → the surviving theory is read by tier 2 on identity-disjoint knots. Compute
   matched against tier 1 given the reader's total evaluations. Only this gate can support the
   claim in the title of this note.

Gates 1, 2, 3 and 5 are all runnable on the existing laptop and none of them touches an admitted
checkpoint.

## 9. What would kill the programme

Stop, or reduce the claim, if any of these occur:

* the reader shows no gap between a correct and a shuffled procedurally generated document
  (gate 3) — then requirement (4) is unreachable with this architecture, and the rest is an
  archive project;
* a correct hand-written theory does not improve held-out coverage (gate 4) — the ceiling is
  zero and invented theory cannot exceed an oracle;
* mined vocabulary is never reused across authors — then `L1` produced private notation, and the
  "shared language" claim goes;
* macro-actions improve coverage no more than an equal number of random macros — the gain was
  horizon shortening, not meaning;
* statements survive falsification only because the falsifier's budget is too small — measure
  refutation power on planted false statements before trusting any `not_refuted`;
* a learned invariant separates the table but fails the Lipschitz test on self-play distance-1
  pairs — it is a feature, not a bound, exactly as `a₂` is in
  [10 §3](10-invariants-and-representations.md); or
* the closed loop wins only without total-compute matching — the TroVE result
  ([arXiv:2507.22069](https://arxiv.org/pdf/2507.22069)) is the precedent, and it is the default
  explanation until excluded.

Any of these is still a useful result. Gate 3 in particular is worth running even if the rest of
the programme is never built: *a knot solver that can be told things* is a contribution
independent of whether the things are invented by other networks.

## 10. Related work, annotated

| work | why it matters here |
|---|---|
| [RTFM, ICLR 2020](https://arxiv.org/abs/1910.08210) | the reader architecture; generalises to unseen dynamics *by reading*, at pre-LLM scale |
| [Boundless Socratic Learning with Language Games](https://arxiv.org/abs/2411.16905) | the position statement for closed-system recursive self-improvement; why the feedback and coverage conditions matter more than the language |
| [GEPA, ICLR 2026](https://arxiv.org/abs/2507.19457) | natural-language reflection as a learning signal, beating GRPO with 35× fewer rollouts |
| [Artificial Generational Intelligence, NeurIPS 2024](https://arxiv.org/pdf/2406.00392) | in-context ("knowledge") vs in-weights ("skill") accumulation — the exact axis this note proposes to switch |
| [DreamProver](https://arxiv.org/abs/2604.26311) | wake/sleep lemma-library evolution; the closest working `L1`+`L2` system |
| [Compute-matched re-evaluation of TroVE](https://arxiv.org/pdf/2507.22069) | the negative control: library reuse gains vanish under matched compute |
| [AlphaEvolve](https://arxiv.org/abs/2506.13131) | island-model archive where the shared artifact is verified code plus its score; the discovery-shaped version of gate 6 |
| [From Storage to Experience (survey)](https://aclanthology.org/2026.findings-acl.2069.pdf) | 2026 map of experience-library / component-level self-evolution vs policy-level updates |
| [Emergent discrete messages](https://arxiv.org/pdf/2102.12550), [language-grounded MARL](https://arxiv.org/html/2409.17348v1) | the `L1`-as-learned-channel alternative, and why it is the weaker option in a verifier-terminated single-agent task |
