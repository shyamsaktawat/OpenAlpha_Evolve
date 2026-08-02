"""Tests for projection_analysis.py: local projectors preserve product states; non-local ones need not."""
import sympy as sp

from projection_analysis import (
    local_projector_preserves_product,
    nonlocal_projector_can_create_apparent_entanglement,
    run_projection_analysis,
)


def _generic_product_state():
    a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2")
    return sp.Matrix([a1, a2]), sp.Matrix([b1, b2])


def test_full_local_projector_preserves_product():
    psi1, psi2 = _generic_product_state()
    _, rank, ok = local_projector_preserves_product(psi1, psi2, sp.eye(2), sp.eye(2))
    assert ok and rank <= 1


def test_partial_local_projector_preserves_product():
    """Even a strictly smaller local projector (post-selection on one branch) cannot entangle."""
    psi1, psi2 = _generic_product_state()
    P1 = sp.Matrix([[1, 0], [0, 0]])
    _, rank, ok = local_projector_preserves_product(psi1, psi2, P1, sp.eye(2))
    assert ok and rank <= 1


def test_arbitrary_local_projectors_preserve_product():
    """Sweep several local projector pairs; product structure must survive every one."""
    psi1, psi2 = _generic_product_state()
    candidates = [
        sp.eye(2),
        sp.Matrix([[1, 0], [0, 0]]),
        sp.Matrix([[0, 0], [0, 1]]),
        sp.Matrix([[sp.Rational(1, 2), sp.Rational(1, 2)],
                   [sp.Rational(1, 2), sp.Rational(1, 2)]]),  # projector onto (|L>+|R>)/sqrt2
    ]
    for P1 in candidates:
        for P2 in candidates:
            _, rank, ok = local_projector_preserves_product(psi1, psi2, P1, P2)
            assert ok, f"local projector pair unexpectedly entangled: rank={rank}"


def test_nonlocal_projector_can_create_apparent_entanglement():
    """Locality is essential -- a Bell-subspace projector does create apparent entanglement."""
    psi1, psi2 = _generic_product_state()
    bell = sp.Matrix([1, 0, 0, 1]) / sp.sqrt(2)
    P_bell = sp.simplify(bell * bell.T)
    _, rank, entangled = nonlocal_projector_can_create_apparent_entanglement(psi1, psi2, P_bell)
    assert entangled and rank > 1


def test_run_projection_analysis_summary():
    r = run_projection_analysis()
    assert r["local_full_projector"]["still_product"] is True
    assert r["local_partial_projector"]["still_product"] is True
    assert r["nonlocal_bell_projector"]["became_entangled"] is True
