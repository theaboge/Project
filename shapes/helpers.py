"""Small shared utilities: multi-dim interp, an so(3) norm, a shape check,
and curve cropping."""

from numpy import stack, interp as np_interp, array, sqrt, trace, dot
from numpy.linalg import norm as np_norm

#: Tolerance below which an angle/norm counts as zero (avoids div-by-zero
#: in the log/exp maps).
TOL = 0.000000000001


def interp(x, xp, fp):
    """Like numpy's `interp`, but for vector-valued `fp` (n, k): interpolates
    each of the k columns independently and stacks the result back together."""
    return stack((np_interp(x, xp, array([f[i] for f in fp])) for i in range(fp.shape[1])), -1)


def norm(r):
    """Norm of an so(3) element: Frobenius-equivalent for matrix form (3,3),
    ordinary 2-norm for vector form.

    Note: the matrix-form check (`r.shape is (3,3)`) compares identity, not
    equality, so it's always False in practice -- every call falls through
    to the vector-norm branch. Harmless here: for a skew-symmetric matrix
    from `hat(v)`, that branch also returns `|v|`, same as intended. Left
    as-is rather than changed, since fixing it changes which code path runs.
    """
    if r.shape is (3, 3):
        return sqrt(0.5 * trace(dot(r, r.T)))
    return np_norm(r, ord=2)


def is_3x3_matrix(q):
    """True if the last two axes of `q` are (3, 3), i.e. matrix-form so(3)
    rather than flat vector form."""
    return q.shape[-2:] == (3, 3)


def crop_curve(c, start=0, step=1, stop=None):
    """Crop a curve (or multi-joint curve) to frames [start:stop:step].
    Used to trim standstill lead-in/lead-out before the distance pipeline."""
    if stop or start != 0:
        if len(c.shape) == 4:
            return array([e[start:stop:step] for e in c])
        else:
            return c[start:stop:step]
    return c
