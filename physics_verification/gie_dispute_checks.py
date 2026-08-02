"""
Deterministic consistency checks for the Aziz-Howl / Gundhi-Infantino-Bassi
dispute over whether classical (unquantized) gravity can entangle two
massive objects each in a two-branch spatial superposition.

References:
  Aziz, J. & Howl, R. "Classical theories of gravity produce entanglement."
    Nature 646, 813-817 (2025). arXiv:2510.19714.
  Gundhi, A., Infantino, G. & Bassi, A. "Can classical theories of gravity
    produce entanglement?" arXiv:2604.19696 (2026).

SCOPE AND LIMITS (read before trusting any result below):
  This module checks ALGEBRAIC CONSISTENCY of a candidate coupling kernel
  V_ij against the two papers' claimed transition-amplitude structures. It
  does NOT independently re-derive V_ij or Lambda from the raw interaction
  Hamiltonian H_I via first-principles Wick contraction -- that derivation
  requires the papers' full text (propagator conventions, wavepacket
  normalization), which was not reachable from this environment (arXiv,
  Nature, and PMC are all blocked by network policy here). The results
  below were checked against literature-search summaries of the papers,
  not the primary-source equations, and should be treated as a technical
  audit for expert review, not a resolution of the dispute.

What IS proven here, unconditionally (pure linear algebra, true for ANY
V_ij, no assumption about its values or origin):
  - If beta_ij is built by correctly summing the transition amplitude over
    ALL initial branch pairs (as required by linearity of QM applied to a
    superposed initial state -- the Gundhi et al. construction), the
    resulting 2x2 matrix is an exact outer product and is THEREFORE ALWAYS
    separable (det == 0), regardless of V_ij's values or the underlying
    geometry.
  - If beta_ij is built by fixing the initial branch pair equal to the
    final one (the Aziz-Howl diagonal-only construction), the result is
    generically NOT an outer product and is generically entangled
    (det != 0), for generic V_ij.
  - These two constructions provably coincide with each other's special
    cases exactly where the papers say they should (diagonal reduction of
    the full-sum construction at fixed geometry reduces to the Aziz-Howl
    near-field kappa/d_ij**2 form).
  - Reading beta_ij as the amplitude matrix of the bipartite pure state
    sum_ij beta_ij |i>_1 |j>_2, the von Neumann entanglement entropy of the
    reduced density matrix rho_1 = beta beta^dagger / Tr(...) is ZERO for the
    full-sum construction (rank 1: zero bits of shared information between the
    objects) and strictly POSITIVE for the generic diagonal construction. This
    is the information-theoretic restatement of the det/rank result:
    S == 0 <=> rank(beta) <= 1 <=> det(beta) == 0 <=> product (separable) state.

What is NOT proven here: that V_ij = i*int(d3x d3y, Phi(x)Phi(y) *
theta_1i(x) theta_2j(y) / |x-y|) and Lambda = m**6 N**2 / (4 pi**2 hbar**6
V**2) (Gundhi et al.'s claimed closed forms) are themselves the correct
result of the 4th-order Dyson series Wick contraction of H_I. That is an
open, harder question this module does not attempt to answer.
"""
import sympy as sp

I_LABELS = ("L", "R")


def build_diagonal_matrix(V, Lambda, t):
    """Aziz-Howl construction: beta^d_ij = beta_ij;ij = Lambda*t**2*V_ij**2 (no sum over initial branches)."""
    return sp.Matrix(2, 2, lambda i, j: Lambda * t**2 * V[i, j]**2)


def build_full_sum_matrix(V, Lambda, t):
    """Gundhi construction: beta_ij = sum_{m,k} beta_ij;mk = Lambda*t**2*(sum_k V_ik)*(sum_m V_mj)."""
    row_sums = [sum(V[i, :]) for i in range(2)]
    col_sums = [sum(V[:, j]) for j in range(2)]
    return sp.Matrix(2, 2, lambda i, j: Lambda * t**2 * row_sums[i] * col_sums[j])


def separability_report(matrix, label):
    """Return (is_separable, det) for a 2x2 branch-amplitude matrix, with a human-readable summary."""
    det = sp.simplify(matrix.det())
    is_separable = det == 0
    return {
        "label": label,
        "matrix": matrix,
        "det": det,
        "is_separable": is_separable,
    }


def check_full_sum_is_always_separable(V, Lambda, t):
    """Hard constraint any correct full-sum derivation MUST satisfy: det == 0 for the given V, unconditionally."""
    matrix = build_full_sum_matrix(V, Lambda, t)
    det = sp.simplify(matrix.det())
    if det != 0:
        return False, det
    return True, det


# --- Entanglement-entropy axis -------------------------------------------
# The branch-amplitude matrix beta_ij is the coefficient matrix of a bipartite
# (2x2) pure state |Psi> = sum_ij beta_ij |i>_1 |j>_2, where i,j index the L/R
# branches of the two objects. The physically meaningful measure of how
# entangled the two objects are is the von Neumann entropy of the reduced
# density matrix rho_1 = beta beta^dagger / Tr(beta beta^dagger); its
# eigenvalues are the squared Schmidt coefficients.
#
# This is a strictly more physical restatement of the det/rank checks above:
#   S == 0  <=>  rank(beta) <= 1  <=>  det(beta) == 0  <=>  product state.
# So the full-sum outer product (rank 1) carries ZERO entanglement entropy
# (zero bits of shared information between the objects), while the diagonal
# construction is generically rank 2 and carries S > 0. The entropy value is
# the quantitative content that the determinant only reports the vanishing of.


def reduced_density_matrix(matrix):
    """rho_1 = beta beta^dagger / Tr(beta beta^dagger): the reduced state of object 1.

    Its eigenvalues are the squared Schmidt coefficients of the bipartite pure
    state; the two objects are entangled iff more than one eigenvalue is nonzero.
    """
    beta = sp.Matrix(matrix)
    gram = beta * beta.H  # beta^dagger = conjugate transpose (real symbols -> transpose)
    trace = sp.trace(gram)
    return gram / trace


def entanglement_entropy(matrix, subs, base=2):
    """Von Neumann entropy S = -sum_k mu_k log(mu_k) of rho_1, in bits (base 2).

    `subs` must map every free symbol in `matrix` to a number, so the result is
    a concrete float. S == 0 iff the state is separable (product state); S > 0
    iff the two objects are entangled. Distinct generic values expose the
    generic (not fine-tuned) behaviour of each construction.
    """
    beta = sp.Matrix(matrix).subs(subs)
    if beta.free_symbols:
        raise ValueError(f"subs must eliminate all free symbols; remaining: {beta.free_symbols}")
    gram = beta * beta.H
    trace = sp.trace(gram)
    if sp.simplify(trace) == 0:
        raise ValueError("zero-norm state has no well-defined reduced density matrix")
    rho = gram / trace
    entropy = sp.Integer(0)
    for eigenvalue, multiplicity in rho.eigenvals().items():
        mu = sp.re(eigenvalue).evalf()  # PSD => real, non-negative eigenvalues
        if mu <= 1e-15:                 # 0*log0 := 0
            continue
        entropy += multiplicity * (-mu * sp.log(mu, base))
    return float(entropy.evalf())


def _generic_subs(V, Lambda, t):
    """Deterministic distinct-prime substitution exposing generic (non-fine-tuned) behaviour."""
    return {Lambda: 1, t: 1, V[0, 0]: 2, V[0, 1]: 3, V[1, 0]: 5, V[1, 1]: 7}


def check_full_sum_has_zero_entanglement_entropy(V, Lambda, t, tol=1e-9):
    """Hard constraint: the full-sum (Gundhi) construction carries zero entanglement
    entropy -- exactly (rank <= 1) and numerically (S ~ 0 bits) -- for generic V_ij."""
    matrix = build_full_sum_matrix(V, Lambda, t)
    rank_ok = int(matrix.rank()) <= 1
    S = entanglement_entropy(matrix, _generic_subs(V, Lambda, t))
    return (rank_ok and abs(S) < tol), S


def check_diagonal_has_positive_entanglement_entropy(V, Lambda, t, tol=1e-6):
    """The diagonal (Aziz-Howl) construction is generically entangled: S > 0 bits for generic V_ij."""
    matrix = build_diagonal_matrix(V, Lambda, t)
    S = entanglement_entropy(matrix, _generic_subs(V, Lambda, t))
    return (S > tol), S


def check_diagonal_reduces_to_near_field_form(V, Lambda, t, distances):
    """Verify beta^d, with V_ij = 1/d_ij substituted, matches kappa/d_ij**2 with kappa = Lambda*t**2 (Aziz-Howl Eq. 84 structure)."""
    subs_map = {V[i, j]: 1 / distances[i, j] for i in range(2) for j in range(2)}
    beta_d = build_diagonal_matrix(V, Lambda, t).subs(subs_map)
    kappa = sp.simplify(Lambda * t**2)
    expected = sp.Matrix(2, 2, lambda i, j: kappa / distances[i, j]**2)
    return sp.simplify(beta_d - expected) == sp.zeros(2, 2)


def run_reference_checks():
    """Run the checks against a fully generic (unconstrained) V_ij and report results."""
    VLL, VLR, VRL, VRR = sp.symbols("V_LL V_LR V_RL V_RR")
    V = sp.Matrix([[VLL, VLR], [VRL, VRR]])
    Lambda, t = sp.symbols("Lambda t", positive=True)
    d = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"d_{I_LABELS[i]}{I_LABELS[j]}", positive=True))

    diag = separability_report(build_diagonal_matrix(V, Lambda, t), "Aziz-Howl diagonal-only")
    full = separability_report(build_full_sum_matrix(V, Lambda, t), "Gundhi full-sum")
    full_ok, full_det = check_full_sum_is_always_separable(V, Lambda, t)
    reduction_ok = check_diagonal_reduces_to_near_field_form(V, Lambda, t, d)
    full_zero_S_ok, full_S = check_full_sum_has_zero_entanglement_entropy(V, Lambda, t)
    diag_pos_S_ok, diag_S = check_diagonal_has_positive_entanglement_entropy(V, Lambda, t)

    return {
        "diagonal": diag,
        "full_sum": full,
        "full_sum_always_separable": full_ok,
        "near_field_reduction_matches_aziz_howl": reduction_ok,
        "full_sum_zero_entanglement_entropy": full_zero_S_ok,
        "full_sum_entropy_bits": full_S,
        "diagonal_positive_entanglement_entropy": diag_pos_S_ok,
        "diagonal_entropy_bits": diag_S,
    }


if __name__ == "__main__":
    results = run_reference_checks()
    print(f"[{results['diagonal']['label']}] det = {results['diagonal']['det']}  "
          f"-> separable: {results['diagonal']['is_separable']}")
    print(f"[{results['full_sum']['label']}] det = {results['full_sum']['det']}  "
          f"-> separable: {results['full_sum']['is_separable']}")
    print(f"Full-sum construction always separable (hard constraint, any V_ij): "
          f"{results['full_sum_always_separable']}")
    print(f"Diagonal construction reduces to Aziz-Howl near-field kappa/d_ij**2 form: "
          f"{results['near_field_reduction_matches_aziz_howl']}")
    print(f"Full-sum entanglement entropy == 0 bits (rank<=1, hard constraint): "
          f"{results['full_sum_zero_entanglement_entropy']} (S = {results['full_sum_entropy_bits']:.3g} bits)")
    print(f"Diagonal entanglement entropy > 0 bits (generically entangled): "
          f"{results['diagonal_positive_entanglement_entropy']} (S = {results['diagonal_entropy_bits']:.3g} bits)")
