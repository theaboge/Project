"""
Small, shared utilities used throughout the package: a norm for so(3)
elements and a shape-sniffing helper.
"""

from numpy import sqrt, trace, dot
from numpy.linalg import norm as np_norm

#: Tolerance below which an angle/norm is treated as exactly zero, to avoid
#: division-by-zero in the log/exp maps (see `so3_math.py`).
TOL = 0.000000000001


def norm(r):
    """
    Norm of an element of the Lie algebra so(3).

    Parameters
    ----------
    r : ndarray, shape (3, 3) or (3,)
        An element of so(3), either as a skew-symmetric matrix or as its
        flat vector representation.

    Returns
    -------
    float
        The norm of `r`.
    """
    if r.shape is (3, 3):
        return sqrt(0.5 * trace(dot(r, r.T)))
    return np_norm(r, ord=2)


def is_3x3_matrix(q):
    """
    Check whether the last two axes of `q` hold 3x3 matrices (so(3)
    elements in matrix form) rather than already-flattened 3-vectors.

    Parameters
    ----------
    q : ndarray
        An array whose last two dimensions are to be checked.

    Returns
    -------
    bool
        True if `q.shape[-2:] == (3, 3)`.
    """
    return q.shape[-2:] == (3, 3)
