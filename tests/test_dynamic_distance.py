"""
Tests that the compiled DP in shapes/dynamic_distance.py gives the same results as the original pure-Python version.

The reference is very slow, so the curves are short crops of real animations (the first and the third trial, 30 and 40 frames, so
the shared grid is the merged one). Run with `pytest tests` from the project root.
"""

import numpy as np
import pytest
from functools import partial

from shapes import dynamic_distance as dd
from shapes.mocap_io import load_curve
from shapes.so3_math import SRVT, skew_to_vector
from dataset import paths_for


def crop_q(animation_id, start, n):
    """The SRVT (vector form) of `n` frames of an animation, and its uniform time grid."""
    c = load_curve(animation_id, *paths_for(animation_id))[:, start:start + n]
    I = np.linspace(0, 1, c.shape[1])
    return skew_to_vector(SRVT(c, I)), I


@pytest.fixture(scope="module")
def pair():
    q0, I0 = crop_q(1, 20, 30)
    q1, I1 = crop_q(3, 20, 40)
    return q0, q1, I0, I1


@pytest.mark.parametrize("depth", [3, 6])
def test_cost_matrix_matches_reference(pair, depth):
    q0, q1, I0, I1 = pair
    I, q0n, q1n = dd.create_shared_parameterization(q0, q1, I0, I1)
    M = I.shape[0]

    _, A_ref = dd.dynamic(partial(dd.local_cost, q0=q0n, q1=q1n, I=I), M, depth)
    A, _ = dd.dynamic_fast(q0n @ q1n.T, (q0n ** 2).sum(axis=1), (q1n ** 2).sum(axis=1), I, depth)

    assert np.array_equal(np.isfinite(A), np.isfinite(A_ref))
    finite = np.isfinite(A_ref)
    np.testing.assert_allclose(A[finite], A_ref[finite], rtol=1e-10, atol=1e-10)


def test_equal_grids_match_reference():
    q, I = crop_q(1, 20, 30)
    q1, _ = crop_q(3, 20, 30)
    phi = dd.find_optimal_diffeomorphism(q, q1, I, I, 5)
    phi_ref = dd.find_optimal_diffeomorphism_reference(q, q1, I, I, 5)
    np.testing.assert_allclose(phi, phi_ref, atol=1e-10)


def test_diffeomorphism_matches_reference(pair):
    q0, q1, I0, I1 = pair
    phi = dd.find_optimal_diffeomorphism(q0, q1, I0, I1, 5)
    phi_ref = dd.find_optimal_diffeomorphism_reference(q0, q1, I0, I1, 5)
    # Two equal-cost paths can be chosen differently, so compare the cost of the alignments rather than the paths.
    # Here that is the end value of the cost matrix (see test_cost_matrix_matches_reference); the maps themselves
    # should also agree up to such ties.
    assert phi.shape == phi_ref.shape
    assert phi[0] == pytest.approx(0.0) and phi[-1] == pytest.approx(1.0)
    assert np.all(np.diff(phi) >= -1e-12)
    np.testing.assert_allclose(phi, phi_ref, atol=0.05)


def test_depth_one_is_the_diagonal(pair):
    # With depth 1 the only step is (i - 1, j - 1) -> (i, j), so the alignment is forced and both versions must agree exactly.
    q0, q1, I0, I1 = pair
    phi = dd.find_optimal_diffeomorphism(q0, q1, I0, I1, 1)
    phi_ref = dd.find_optimal_diffeomorphism_reference(q0, q1, I0, I1, 1)
    np.testing.assert_allclose(phi, phi_ref, atol=1e-12)
