"""
The open "projection" question in the Gundhi-Infantino-Bassi vs Aziz-Howl dispute.

Gundhi et al. prove (their Sec. III, non-perturbative) that the FULL state stays
exactly factorized at all times. They then explicitly flag as unresolved -- "not
an easy computation" -- whether PROJECTING that state onto the 4-branch BMV
subspace {|N>_1L, |N>_1R} x {|N>_2L, |N>_2R} could introduce APPARENT entanglement.

This module addresses the well-posed part of that question and isolates precisely
where the genuine difficulty lies.

RESULT 1 (proven here, elementary): if the full state is an exact product
|psi_1> (x) |psi_2> with respect to a tensor factorization H = H_1 (x) H_2, and
the projector onto the observed subspace is LOCAL, i.e. factorizes as
P = P_1 (x) P_2, then

    P (|psi_1> (x) |psi_2>) = (P_1|psi_1>) (x) (P_2|psi_2>)

is STILL an exact product state, and normalization does not change that.
=> Projection onto the BMV subspace CANNOT manufacture entanglement, provided
   both conditions hold. The BMV projector IS of this local product form
   (P_1 = sum_i |N>_1i<N|_1i, P_2 = sum_j |N>_2j<N|_2j).

RESULT 2 (proven here by explicit counterexample): the locality of the projector
is doing real work. A NON-product projector acting on a product state generically
DOES yield an entangled (rank-2) output. So the argument is not vacuous -- it
turns entirely on whether the observed subspace's projector factorizes.

WHERE THE REAL DIFFICULTY LIES (not resolved here): both results above presuppose
a genuine tensor factorization H = H_1 (x) H_2 labelled by "object 1" and
"object 2". For N IDENTICAL bosons in a single common Fock space -- which is the
actual setting of both papers (complex Klein-Gordon field) -- there is no
canonical such factorization. "Object 1" and "object 2" are mode subsets of one
field, and entanglement statements are partition-relative. Whether the
mode-partition used by either paper induces a genuine tensor product structure
(and hence whether RESULT 1 applies to their setting) is exactly the hard part
Gundhi et al. leave open. This module does NOT settle it, and no result here
should be read as doing so.

Treat all output as a technical audit for expert review, not a resolution.
"""
import sympy as sp


def local_projector_preserves_product(psi1, psi2, P1, P2):
    """RESULT 1: check that (P1 (x) P2)(psi1 (x) psi2) is still an exact product state.

    Represents the joint state as the outer-product coefficient matrix
    C_ij = psi1_i * psi2_j. Separable <=> rank(C) <= 1.
    Returns (projected_matrix, rank, is_still_product).
    """
    C = psi1 * psi2.T                     # outer product => rank <= 1 by construction
    C_proj = (P1 * C) * P2.T              # (P1 (x) P2) acting on the coefficient matrix
    C_proj = sp.simplify(C_proj)
    rank = C_proj.rank()
    return C_proj, rank, rank <= 1


def nonlocal_projector_can_create_apparent_entanglement(psi1, psi2, P_joint):
    """RESULT 2: a NON-product projector on the joint space generically breaks the product form.

    P_joint acts on vec(C) (the 4-dim joint space), not as P1 (x) P2.
    Returns (projected_matrix, rank, became_entangled).
    """
    C = psi1 * psi2.T
    vec_C = sp.Matrix([C[0, 0], C[0, 1], C[1, 0], C[1, 1]])   # row-major vectorization
    vec_out = sp.simplify(P_joint * vec_C)
    C_out = sp.Matrix([[vec_out[0], vec_out[1]], [vec_out[2], vec_out[3]]])
    C_out = sp.simplify(C_out)
    rank = C_out.rank()
    return C_out, rank, rank > 1


def run_projection_analysis():
    a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2")
    psi1 = sp.Matrix([a1, a2])
    psi2 = sp.Matrix([b1, b2])

    # --- RESULT 1: BMV-style LOCAL projector (product of per-object projectors) ---
    # Full identity on the 2-branch subspace of each object (the BMV subspace):
    P1_full = sp.eye(2)
    P2_full = sp.eye(2)
    C1, rank1, ok1 = local_projector_preserves_product(psi1, psi2, P1_full, P2_full)

    # A strictly smaller LOCAL projector (e.g. post-select object 1 onto branch L only):
    P1_partial = sp.Matrix([[1, 0], [0, 0]])
    C2, rank2, ok2 = local_projector_preserves_product(psi1, psi2, P1_partial, P2_full)

    # --- RESULT 2: a NON-product (entangling) projector on the joint space ---
    # Projector onto the Bell-like subspace spanned by (|LL> + |RR>)/sqrt(2):
    # This does NOT factorize as P1 (x) P2.
    bell = sp.Matrix([1, 0, 0, 1]) / sp.sqrt(2)
    P_bell = sp.simplify(bell * bell.T)
    C3, rank3, entangled3 = nonlocal_projector_can_create_apparent_entanglement(psi1, psi2, P_bell)

    return {
        "local_full_projector": {"matrix": C1, "rank": rank1, "still_product": ok1},
        "local_partial_projector": {"matrix": C2, "rank": rank2, "still_product": ok2},
        "nonlocal_bell_projector": {"matrix": C3, "rank": rank3, "became_entangled": entangled3},
    }


if __name__ == "__main__":
    r = run_projection_analysis()
    print("RESULT 1a -- LOCAL projector (full BMV subspace, P1 (x) P2):")
    print(f"  rank = {r['local_full_projector']['rank']}, "
          f"still an exact product: {r['local_full_projector']['still_product']}")
    print("RESULT 1b -- LOCAL projector (partial: post-select object 1 on branch L):")
    print(f"  rank = {r['local_partial_projector']['rank']}, "
          f"still an exact product: {r['local_partial_projector']['still_product']}")
    print("RESULT 2  -- NON-product projector (Bell subspace, does NOT factorize):")
    print(f"  rank = {r['nonlocal_bell_projector']['rank']}, "
          f"apparent entanglement created: {r['nonlocal_bell_projector']['became_entangled']}")
    print()
    print("=> Projection onto the BMV subspace cannot manufacture entanglement,")
    print("   BECAUSE that projector is local (P1 (x) P2). Locality is essential:")
    print("   a non-product projector generically does create apparent entanglement.")
    print()
    print("   NOT SETTLED HERE: whether a genuine tensor factorization H_1 (x) H_2")
    print("   exists at all for N identical bosons in a shared Fock space. That is")
    print("   the actual open difficulty Gundhi et al. flag. See module docstring.")
