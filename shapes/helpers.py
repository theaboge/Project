"""
Small, shared utilities used throughout the package: a multi-dimensional
version of numpy's `interp`, a norm for so(3) elements, a shape-sniffing
helper, and curve cropping.
"""

from numpy import stack, interp as np_interp, array, sqrt, trace, dot
from numpy.linalg import norm as np_norm

#: Tolerance below which an angle/norm is treated as exactly zero, to avoid
#: division-by-zero in the log/exp maps (see `transformations.py`).
TOL = 0.000000000001


def interp(x, xp, fp):
    """Linearly interpolate a vector-valued function, point by point.

    numpy's own `interp` only works for scalar-valued `fp`; this applies it
    independently to each of `fp`'s columns and stacks the results back
    together.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated function.
    xp : array_like, shape (n,)
        The x-coordinates of the data points, must be increasing.
    fp : array_like, shape (n, k)
        The function values at `xp`; each row is a k-dimensional vector.

    Returns
    -------
    ndarray, shape (len(x), k)
        The interpolated vector-valued function, evaluated at `x`.
    """
    return stack((np_interp(x, xp, array([f[i] for f in fp])) for i in range(fp.shape[1])), -1)


def norm(r):
    """Norm of an element of the Lie algebra so(3).

    Intended to use the Frobenius-equivalent formula `sqrt(0.5*trace(r @ r.T))`
    when `r` is given in matrix ("hatted") form, and numpy's ordinary vector
    2-norm when `r` is given in flat vector form.

    Parameters
    ----------
    r : ndarray, shape (3, 3) or (3,)
        An element of so(3), either as a skew-symmetric matrix or as its
        flat vector representation.

    Returns
    -------
    float
        The norm of `r`.

    Notes
    -----
    The matrix-form branch is guarded by `r.shape is (3,3)`, which compares
    object identity rather than equality and in practice is always False
    -- so every call actually falls through to `np_norm(r, ord=2)`. This
    turns out to be harmless here: for a skew-symmetric 3x3 matrix coming
    from `hat(v)`, numpy's `ord=2` (largest singular value) and the
    intended Frobenius-equivalent formula both equal `|v|`, so the two
    branches agree numerically even though only one of them ever runs.
    Left as-is rather than "fixed", since changing the comparison changes
    which code path executes for inputs this codebase never actually
    passes in.
    """
    if r.shape is (3, 3):
        return sqrt(0.5 * trace(dot(r, r.T)))
    return np_norm(r, ord=2)


def is_3x3_matrix(q):
    """Check whether the last two axes of `q` hold 3x3 matrices (so(3)
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


def crop_curve(c, start=0, step=1, stop=None):
    """Crop a curve (or multi-joint curve) to the frame range [start:stop:step].

    Used to trim standstill lead-in/lead-out frames from a recorded
    animation before it enters the distance pipeline.

    Parameters
    ----------
    c : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        A curve, or multi-joint curve, to crop along its frame axis.
    start : int, optional
        First frame to keep (default 0, i.e. no cropping at the start).
    step : int, optional
        Stride between kept frames (default 1, i.e. keep every frame).
    stop : int or None, optional
        First frame to drop (default None, i.e. no cropping at the end).

    Returns
    -------
    ndarray
        The cropped curve. If `start == 0` and `stop is None`, `c` is
        returned unchanged (no copy is made).
    """
    if stop or start != 0:
        if len(c.shape) == 4:
            return array([e[start:stop:step] for e in c])
        else:
            return c[start:stop:step]
    return c
