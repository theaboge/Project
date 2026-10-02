"""
Log-signature computation and the normalized log-signature distance
(sim_n / d_sig) -- the fast, alignment-free metric behind the "signature"
results in this project.
"""

from numpy import linspace, flip
from numpy import concatenate as np_concatenate
from numpy.linalg import norm as np_norm
from iisignature import logsig
from .transformations import right_log, skew_to_vector
from .curves import lift_piece_wise_constant


def log_signature(path, s):
    """Thin wrapper around `iisignature.logsig(path, s)`."""
    return logsig(path, s)


def curve_log_signature(c, s):
    """Log-signature of a curve in SO(3)^d.

    Warning: the intent (per the original codebase, unchanged here) seems
    to have been to build the right-log derivative `q`, lift it to a
    piecewise-linear path `x` (the path a signature is meant to represent),
    and take the signature of that. Instead, `log_signature(q, s)` is
    called directly on the un-lifted `q`. The dead `x = lift_piece_wise_
    constant(q, I)` line computing the (unused) intended input has been
    removed here -- it crashed for curves shorter than ~69 frames (a
    shape-indexing bug in `lift_piece_wise_constant`, irrelevant now that
    it's not called), and removing a provably-unused computation cannot
    change this function's output. The underlying question -- whether
    `logsig(q,s)` should actually be `logsig(x,s)` -- is unresolved and
    unrelated to this fix; flagged, not fixed, since it would change every
    signature result in this project.
    """
    is_multi = len(c.shape) == 4
    I = linspace(0, 1, c.shape[1 if is_multi else 0])
    q = skew_to_vector(right_log(c, I))
    return log_signature(q, s)


def linear_metric(sig0, sig1):
    """Plain (unnormalized) distance ||sig0 - sig1||. Produces uneven,
    poorly-scaled results (sigma_L^n in the thesis) -- see
    `normalized_linear_distance`, which is what this project actually uses."""
    return np_norm(sig0 - sig1)


def normalized_linear_distance(sig0, sig1):
    """Normalized log-signature distance (sim_n, thesis eq. 9.1): each
    signature divided by its own norm before differencing. This is the
    metric used for every "signature" distance in this project."""
    a = np_norm(sig0)
    b = np_norm(sig1)
    div = lambda x: x if x > 0.0 else 1.0
    return np_norm(sig0 / div(a) - sig1 / div(b))


def concatenate_metric(c0, c1, s):
    """Alternative metric: concatenate the two curves (one reversed) and
    compare signatures of the two concatenations. Not used in the current
    pipeline; kept for reference."""
    I0 = linspace(0.0, 1.0, c0.shape[1])
    I1 = linspace(0.0, 1.0, c1.shape[1])

    q0 = skew_to_vector(right_log(c0, I0))
    q1 = skew_to_vector(right_log(c1, I1))

    u = np_concatenate((q0[:-1], flip(q1[:-1], axis=0)), axis=0)
    v = np_concatenate((q1[:-1], flip(q0[:-1], axis=0)), axis=0)

    Iu = np_concatenate((linspace(0, 0.5, q0.shape[0]), linspace(0.5, 1, q1.shape[0])[1:]), axis=0)
    Iv = np_concatenate((linspace(0, 0.5, q1.shape[0]), linspace(0.5, 1, q0.shape[0])[1:]), axis=0)

    x = lift_piece_wise_constant(u, Iu)
    y = lift_piece_wise_constant(v, Iv)
    return np_norm(log_signature(x, s)) + np_norm(log_signature(y, s))
