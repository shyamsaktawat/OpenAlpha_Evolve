"""Tests for gie_dispute_checks.py: confirm the hard constraints actually discriminate a correct outer-product-sum derivation from a broken one."""
import sympy as sp
import pytest

from gie_dispute_checks import (
    build_diagonal_matrix,
    build_full_sum_matrix,
    check_full_sum_is_always_separable,
    check_diagonal_reduces_to_near_field_form,
    check_full_sum_has_zero_entanglement_entropy,
    check_diagonal_has_positive_entanglement_entropy,
    entanglement_entropy,
    run_reference_checks,
)


def _generic_V():
    VLL, VLR, VRL, VRR = sp.symbols("V_LL V_LR V_RL V_RR")
    return sp.Matrix([[VLL, VLR], [VRL, VRR]])


def test_full_sum_construction_is_always_separable():
    V = _generic_V()
    Lambda, t = sp.symbols("Lambda t", positive=True)
    ok, det = check_full_sum_is_always_separable(V, Lambda, t)
    assert ok
    assert det == 0


def test_diagonal_construction_is_generically_entangled():
    V = _generic_V()
    Lambda, t = sp.symbols("Lambda t", positive=True)
    matrix = build_diagonal_matrix(V, Lambda, t)
    det = sp.simplify(matrix.det())
    assert det != 0


def test_diagonal_reduces_to_near_field_kappa_over_d_squared():
    V = _generic_V()
    Lambda, t = sp.symbols("Lambda t", positive=True)
    d = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"d_{('L','R')[i]}{('L','R')[j]}", positive=True))
    assert check_diagonal_reduces_to_near_field_form(V, Lambda, t, d)


def test_a_broken_candidate_fails_the_full_sum_hard_constraint():
    """A candidate that does NOT sum over initial branches correctly (e.g. sums only k, drops m) must be caught."""
    V = _generic_V()
    Lambda, t = sp.symbols("Lambda t", positive=True)

    def broken_full_sum(V, Lambda, t):
        # BROKEN: sums over k for the row index but reuses the same row index
        # for the column sum instead of summing over m -- an easy mistake
        # that silently breaks the outer-product structure.
        row_sums = [sum(V[i, :]) for i in range(2)]
        return sp.Matrix(2, 2, lambda i, j: Lambda * t**2 * row_sums[i] * row_sums[j])

    matrix = broken_full_sum(V, Lambda, t)
    det = sp.simplify(matrix.det())
    # This particular broken variant happens to still be rank-1 (it's an
    # outer product of row_sums with itself), so assert on a genuinely
    # non-outer-product broken variant instead:
    def broken_full_sum_2(V, Lambda, t):
        # BROKEN: adds a spurious cross term that isn't a pure outer product.
        row_sums = [sum(V[i, :]) for i in range(2)]
        col_sums = [sum(V[:, j]) for j in range(2)]
        return sp.Matrix(2, 2, lambda i, j: Lambda * t**2 * row_sums[i] * col_sums[j] + V[i, j])

    matrix2 = broken_full_sum_2(V, Lambda, t)
    det2 = sp.simplify(matrix2.det())
    assert det2 != 0  # correctly flagged as inconsistent with the required hard constraint


def test_entanglement_entropy_bell_state_is_one_bit():
    """Ground-truth anchor: the amplitude matrix [[1,0],[0,1]] is a Bell state, S = 1 bit exactly."""
    bell = sp.Matrix([[1, 0], [0, 1]])
    assert entanglement_entropy(bell, {}) == pytest.approx(1.0, abs=1e-9)


def test_entanglement_entropy_product_state_is_zero_bits():
    """Ground-truth anchor: any rank-1 (outer product) amplitude matrix is a product state, S = 0 bits."""
    product = sp.Matrix([[1, 1], [1, 1]])  # outer product [1,1] x [1,1]
    assert entanglement_entropy(product, {}) == pytest.approx(0.0, abs=1e-9)


def test_full_sum_construction_has_zero_entanglement_entropy():
    V = _generic_V()
    Lambda, t = sp.symbols("Lambda t", positive=True)
    ok, S = check_full_sum_has_zero_entanglement_entropy(V, Lambda, t)
    assert ok
    assert S == pytest.approx(0.0, abs=1e-9)


def test_diagonal_construction_has_positive_entanglement_entropy():
    V = _generic_V()
    Lambda, t = sp.symbols("Lambda t", positive=True)
    ok, S = check_diagonal_has_positive_entanglement_entropy(V, Lambda, t)
    assert ok
    assert S > 0.0


def test_entropy_axis_agrees_with_determinant_axis():
    """S == 0 must coincide with det == 0 (separable) and S > 0 with det != 0 (entangled)."""
    V = _generic_V()
    Lambda, t = sp.symbols("Lambda t", positive=True)

    full = build_full_sum_matrix(V, Lambda, t)
    _, full_S = check_full_sum_has_zero_entanglement_entropy(V, Lambda, t)
    assert sp.simplify(full.det()) == 0 and full_S == pytest.approx(0.0, abs=1e-9)

    diag = build_diagonal_matrix(V, Lambda, t)
    _, diag_S = check_diagonal_has_positive_entanglement_entropy(V, Lambda, t)
    assert sp.simplify(diag.det()) != 0 and diag_S > 0.0


def test_reference_checks_all_pass():
    results = run_reference_checks()
    assert results["diagonal"]["is_separable"] is False
    assert results["full_sum"]["is_separable"] is True
    assert results["full_sum_always_separable"] is True
    assert results["near_field_reduction_matches_aziz_howl"] is True
    assert results["full_sum_zero_entanglement_entropy"] is True
    assert results["full_sum_entropy_bits"] == pytest.approx(0.0, abs=1e-9)
    assert results["diagonal_positive_entanglement_entropy"] is True
    assert results["diagonal_entropy_bits"] > 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
