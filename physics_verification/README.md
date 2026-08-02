# Technical audit: "can classical gravity produce entanglement?"

A computational audit of an **active, unresolved dispute** in the
quantum-gravity-experiment literature. **Nothing here settles it.** Every result
below is either (i) elementary linear algebra that is true regardless of the
physics, or (ii) a consistency check *conditional on* structural forms taken from
the papers. This is material for expert review, not a resolution.

## The dispute

Two masses, each in a two-branch spatial superposition (L/R), interacting only via
gravity. After time `t` the joint state is `|Psi(t)> = (1/2) sum_ij beta_ij |N>_1i |N>_2j`.
For a 2x2 bipartite pure state the separability test is exact and basis-independent:

> `det[beta_ij] == 0` <=> product (separable); `det != 0` <=> entangled.

- **Aziz & Howl**, *Nature* **646**, 813-817 (2025), [arXiv:2510.19714](https://arxiv.org/abs/2510.19714) —
  claim classical gravity **does** entangle, via a 4th-order "crossing" diagram
  (virtual matter exchanged between the objects). Their footnote 68 asserts no
  other 4th-order-or-lower amplitudes exist to combine with and restore separability.
- **Gundhi, Infantino & Bassi**, [arXiv:2604.19696](https://arxiv.org/abs/2604.19696) (2026) —
  claim that conclusion follows only from computing the **diagonal** transition
  amplitude. Summing over **all** initial branch pairs (required by linearity of QM
  applied to a superposed initial state) restores an exact outer-product structure,
  hence exact separability.

## What was verified computationally

Run: `python -m pytest physics_verification/` (10 tests, all passing).

### 1. The algebraic crux — `gie_dispute_checks.py`

Using a **fully generic** coupling kernel `V_ij` (four independent free symbols; no
assumption that `V` is itself low-rank, and it is generically full-rank):

| Construction | Form | `det` | Verdict |
|---|---|---|---|
| Aziz-Howl, diagonal-only | `beta_ij = Lambda t^2 V_ij^2` | `Lambda^2 t^4 (V_LL^2 V_RR^2 - V_LR^2 V_RL^2)` | **generically nonzero** -> entangled |
| Gundhi, full sum | `beta_ij = Lambda t^2 (sum_k V_ik)(sum_m V_mj)` | **identically `0`** | **always separable** |

The second row is a **theorem, not an approximation**: summing a matrix into a
row-sum vector times a column-sum vector always produces an exact outer product,
which always has rank 1. It holds for *any* `V_ij` whatsoever — any geometry, any
parameter values. Verified symbolically, and separately confirmed at rank level
for the n x n generalisation.

Also verified: the diagonal construction reduces exactly to Aziz-Howl's stated
near-field form `kappa/d_ij^2` (with `kappa = Lambda t^2`) under `V_ij = 1/d_ij`.

**So the two papers are not in arithmetic contradiction.** Both computed their own
expression correctly. The dispute is entirely about *which object is the physically
correct transition amplitude* — a modelling question, not an algebra error.

### 2. The open "projection" question — `projection_analysis.py`

Gundhi et al. prove the *full* state stays factorized at all orders, then flag as
unresolved ("not an easy computation") whether projecting onto the 4-branch BMV
subspace could introduce *apparent* entanglement. Addressing the well-posed part:

- **A local projector `P = P_1 (x) P_2` acting on a product state always yields a
  product state** (rank stays <= 1). Verified for the full BMV projector, for
  partial post-selection on a single branch, and swept over a family of local
  projector pairs. The BMV subspace projector *is* of this local product form.
- **Locality is doing real work**: a *non*-product projector (e.g. onto a Bell
  subspace) applied to the same product state gives rank 2 — apparent entanglement.
  So the argument is not vacuous.

**Conclusion (conditional):** projection onto the BMV subspace cannot manufacture
entanglement — *provided* a genuine tensor factorization `H = H_1 (x) H_2` exists.

## What remains assumption-dependent

1. **The hard part of the projection question is untouched.** Both papers work with
   `N` **identical** bosons in one shared Fock space. There, "object 1" and "object 2"
   are *mode subsets of a single field*, and there is no canonical tensor
   factorization. Entanglement is partition-relative. Whether the papers' mode
   partition induces a genuine tensor product structure — and hence whether the
   projection result above even applies — is precisely the difficulty Gundhi et al.
   leave open. **Not settled here.**
2. **The Wick contraction itself was not independently re-derived.** These checks
   validate the *structural claims* (`beta_ij;mk ∝ V_ik V_mj`), not that `V_ij` and
   `Lambda` are the correct output of contracting `H_I` from first principles. Doing
   that requires the papers' full text — **arxiv.org, nature.com and PMC are all
   blocked by this environment's egress policy** (403 at the proxy gateway), so this
   audit worked from the task brief's stated equations plus literature-search
   summaries, *not* primary-source equation-by-equation comparison.
3. **Necessary != sufficient.** These checks can *falsify* a candidate derivation but
   cannot confirm one. Demonstrated concretely: a degenerate all-zeros candidate
   passes 2 of the 3 checks in the companion evolutionary task while being physically
   empty.

## Cross-check against the rest of the literature

The "no entanglement" camp does **not** argue from one shared mechanism — which is
informative, since independent routes to the same conclusion strengthen it:

- **Gundhi et al.** — combinatorial/perturbative: the discarded off-diagonal
  amplitudes restore the outer-product structure. Plus a non-perturbative
  all-orders argument via `U(t,s) a^dag(f) U(t,s)^dag = a^dag(U_1(t,s) f)`.
- **Diósi**, [arXiv:2511.00852](https://arxiv.org/abs/2511.00852) — different route:
  argues the interaction-picture perturbative result is inconsistent with an *exact
  non-perturbative Heisenberg-picture* solution, which precludes entanglement.
- **Schneider, Huggett & Linnemann**, [arXiv:2511.19242](https://arxiv.org/abs/2511.19242) —
  different again, and not a QFT computation at all: a **Newton-Cartan** argument
  that if entanglement *were* observed, something other than gravity must have
  supplied the mediating virtual force.
- **Marletto, Oppenheim, Vedral & Wilson**, [arXiv:2511.07348](https://arxiv.org/abs/2511.07348) —
  mediation would be via matter, not gravity.

Supporting Aziz-Howl: **Lin & Mondal**, [arXiv:2510.23584](https://arxiv.org/abs/2510.23584);
**Di Biagio**, [arXiv:2511.02683](https://arxiv.org/abs/2511.02683).

The dispute is **still expanding** past the reference list this audit started from —
literature search surfaced further entries ([arXiv:2604.16276](https://arxiv.org/abs/2604.16276),
[arXiv:2607.03429](https://arxiv.org/abs/2607.03429),
[arXiv:2512.13675](https://arxiv.org/abs/2512.13675)), including one whose title alone
("Matter-Field Exchange Generates Entanglement, Not Classical Gravity") suggests a
third position: the effect is real but is not *gravitational*.

## Status

Only Aziz & Howl is peer-reviewed (*Nature*). **Every rebuttal is an unrefereed
preprint.** This audit was produced by an AI system without primary-source access,
and should be checked by a physicist against the actual papers before any part of it
is relied upon.

## Files

- `gie_dispute_checks.py` — separability checks for both constructions
- `projection_analysis.py` — the local-projector result on the open projection question
- `test_*.py` — 10 tests, including discrimination tests proving the checks reject broken candidates
- `../examples/gravity_entanglement_wick_derivation.yaml` — companion evolutionary-search task
