"""
Log-signature computation and the normalized log-signature distance (sim_n / d_sig).
"""

from numpy import linspace
from numpy.linalg import norm as np_norm
from iisignature import logsig
from .so3_math import right_log, skew_to_vector


def log_signature(path, s):
    """
    Thin wrapper around `iisignature.logsig`.

    Parameters
    ----------
    path : ndarray, shape (n, dim)
        A piecewise-linear path (n points in `dim`-dimensional space).
    s : iisignature "prepare" object
        A signature-preparation object for the given dimension and
        truncation level, as returned by `iisignature.prepare(dim, level)`.

    Returns
    -------
    ndarray
        The log-signature of `path`, truncated at the level `s` was
        prepared for.
    """
    return logsig(path, s)


def curve_log_signature(c, s):
    """
    Compute the log-signature of a curve in SO(3)^d.

    Parameters
    ----------
    c : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        A curve (or multi-joint curve) of rotation matrices.
    s : iisignature "prepare" object
        A signature-preparation object (see `log_signature`).

    Returns
    -------
    ndarray
        The log-signature of the curve.

    Note
    ----
    This computes the right-logarithmic derivative `q` of `c` (the same
    quantity `SRVT` normalizes) and calls `log_signature(q, s)` directly on
    that piecewise-constant `q`, rather than on a piecewise-linear lift of
    `q`. Whether the signature should technically be taken of a lifted
    path instead is an open question -- flagged here since it would alter
    every signature-based result in this project.
    """
    is_multi = len(c.shape) == 4
    I = linspace(0, 1, c.shape[1 if is_multi else 0])
    q = skew_to_vector(right_log(c, I))
    return log_signature(q, s)


def normalized_linear_distance(sig0, sig1):
    """
    Normalized log-signature distance: sim_n(x, y) from the thesis
    (Sec. 9.3, eq. 9.1) / d_sig throughout this project.

    Each signature is normalized by its own norm before differencing --
    the raw, unnormalized distance (sigma_L^n in the thesis / Sec. 9.2) is
    known to produce uneven, poorly-scaled results across different
    animations, which is why this normalized version exists and is what
    this project actually uses.

    Parameters
    ----------
    sig0, sig1 : ndarray
        Two (log-)signature vectors of the same length.

    Returns
    -------
    float
        || sig0/||sig0|| - sig1/||sig1|| ||. If either signature has zero
        norm, that signature is left un-normalized (divided by 1 instead
        of 0) rather than raising an error.
    """
    a = np_norm(sig0)
    b = np_norm(sig1)
    div = lambda x: x if x > 0.0 else 1.0
    return np_norm(sig0 / div(a) - sig1 / div(b))
