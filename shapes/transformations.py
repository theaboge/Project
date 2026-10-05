"""
Core Lie-group toolbox for SO(3): the hat/hat-inverse isomorphism between
R^3 and the Lie algebra so(3), the log/exp maps between SO(3) and so(3),
geodesic interpolation, the right-logarithmic derivative, and the Square
Root Velocity Transform (SRVT).

Shapes used throughout this file:
    c in Imm(I, SO(3))     : n x 3 x 3      (a curve of rotation matrices)
    c in Imm(I, SO(3)^d)   : d x n x 3 x 3  (d joints, each a curve in SO(3))
    q in C(I, so(3))       : n x 3 x 3      (matrix / "hatted" form)
    q in C(I, so(3)^d)     : d x n x 3 x 3
    q in C(I, so(3))       : n x 3          (vector form, after hatinv)
    q in C(I, so(3)^d)     : n x 3*d
"""

from numpy import array, isnan, zeros, eye, trace, dot
from numpy import sin, cos, arccos, sqrt
from .helpers import TOL, norm, is_3x3_matrix


# ---------------------------------------------------------------------------
# SO(3) basis rotations
# ---------------------------------------------------------------------------

def Rx(a):
    """
    Rotation matrix for a rotation of angle `a` about the x-axis.

    Parameters
    ----------
    a : float
        Rotation angle in radians.

    Returns
    -------
    (3, 3) ndarray
        The corresponding element of SO(3).
    """
    s, c = sin(a), cos(a)
    return array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    """
    Rotation matrix for a rotation of angle `a` about the y-axis.

    Parameters
    ----------
    a : float
        Rotation angle in radians.

    Returns
    -------
    (3, 3) ndarray
        The corresponding element of SO(3).
    """
    s, c = sin(a), cos(a)
    return array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    """Rotation matrix for a rotation of angle `a` about the z-axis.

    Parameters
    ----------
    a : float
        Rotation angle in radians.

    Returns
    -------
    (3, 3) ndarray
        The corresponding element of SO(3).
    """
    s, c = sin(a), cos(a)
    return array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------------------
# The hat isomorphism R^3 <-> so(3)
# ---------------------------------------------------------------------------

def hat(v):
    """
    Map a vector in R^3 to its skew-symmetric matrix in so(3).

    This is the standard isomorphism R^3 <-> so(3): for v = (v1, v2, v3),
    hat(v) is the matrix such that hat(v) @ x == cross(v, x) for all x.

    Recurses automatically over whole curves (n x 3) or multi-joint curves
    (d x n x 3), applying the isomorphism element-wise.

    Parameters
    ----------
    v : ndarray, shape (3,), (n, 3), or (d, n, 3)
        A single vector, a curve of vectors, or a multi-joint curve.

    Returns
    -------
    ndarray, shape (3, 3), (n, 3, 3), or (d, n, 3, 3)
        The corresponding skew-symmetric matrix/matrices.
    """
    if len(v.shape) == 4:
        return array([hat(curve) for curve in v])
    if len(v.shape) == 3:
        return array([hat(vector) for vector in v])
    return array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def hatinv(M):
    """Inverse of `hat`: recover the vector in R^3 from a skew-symmetric
    matrix in so(3).

    Recurses automatically over whole curves, the same way `hat` does.

    Parameters
    ----------
    M : ndarray, shape (3, 3), (n, 3, 3), or (d, n, 3, 3)
        A single skew-symmetric matrix, a curve of them, or a multi-joint
        curve of them.

    Returns
    -------
    ndarray, shape (3,), (n, 3), or (d, n, 3)
        The corresponding vector(s) in R^3.
    """
    if len(M.shape) == 4:
        return array([hatinv(curve) for curve in M])
    if len(M.shape) == 3:
        return array([hatinv(matrix) for matrix in M])
    return array([-M[1, 2], M[0, 2], -M[0, 1]])


# ---------------------------------------------------------------------------
# log / exp maps between SO(3) and so(3)
# ---------------------------------------------------------------------------

def rot_log(R):
    """
    Matrix logarithm SO(3) -> so(3) (Rodrigues' formula, inverted).

    Assumes R is close to, but not equal to, the identity -- this is the
    regime the rest of the pipeline always calls it in (consecutive frames
    of a motion-capture recording rotate only a small amount).

    Parameters
    ----------
    R : (3, 3) ndarray
        A rotation matrix in SO(3).

    Returns
    -------
    (3, 3) ndarray
        The corresponding element of so(3) (a skew-symmetric matrix).

    Notes
    -----
    When the rotation angle theta is zero or comes out as NaN, any element
    of so(3) is a valid choice, so `0.5 * (R - R.T)` (the antisymmetric
    part of R, which is the correct formula's limit as theta -> 0) is
    returned directly. NaNs here are a known floating-point artifact: the
    angle is computed from `arccos(0.5*(trace(R)-1))`, and rounding error
    can push `trace(R)` fractionally above 3 (e.g. 3.000000000000002),
    putting the arccos argument just outside its valid domain.
    """
    theta = arccos(0.5 * (trace(R) - 1))
    if abs(theta) < TOL or isnan(theta):
        return 0.5 * (R - R.T)
    return (theta / sin(theta)) * 0.5 * (R - R.T)


def rot_exp(r):
    """
    Matrix exponential so(3) -> SO(3) (Rodrigues' formula).

    Parameters
    ----------
    r : (3, 3) ndarray
        An element of so(3) (a skew-symmetric matrix).

    Returns
    -------
    (3, 3) ndarray
        The corresponding rotation matrix in SO(3).

    Notes
    -----
    `theta = norm(r)` is the rotation angle. When it is zero or NaN, the
    identity matrix is returned directly, since the closed-form expression
    below has a removable singularity at theta = 0.
    """
    theta = norm(r)
    if abs(theta) < TOL or isnan(theta):
        return eye(3)
    return eye(3) + (sin(theta) / theta) * r + ((1 - cos(theta)) / (theta ** 2)) * dot(r, r)


def Ad(A, X):
    """
    Adjoint action of A on X: A^T X A.

    Parameters
    ----------
    A : (3, 3) ndarray
        A rotation matrix in SO(3).
    X : (3, 3) ndarray
        An element of so(3) (or any 3x3 matrix) to transport.

    Returns
    -------
    (3, 3) ndarray
        A^T @ X @ A.
    """
    return dot(dot(A.T, X), A)


def interpolate(R0, R1, s=1):
    """
    Geodesic interpolation between two close rotations in SO(3).

    The Lie-group analogue of linear interpolation A + s(B-A) in R^n: walk
    along the one-parameter subgroup from R0 towards R1, a fraction `s` of
    the way (`s=0` gives R0, `s=1` gives R1).

    Parameters
    ----------
    R0 : (3, 3) ndarray
        Starting rotation.
    R1 : (3, 3) ndarray
        Target rotation (must be close to R0 for `rot_log` to be valid).
    s : float, optional
        Interpolation fraction along the geodesic from R0 to R1 (default 1,
        i.e. land exactly on R1).

    Returns
    -------
    (3, 3) ndarray
        The interpolated rotation.
    """
    return dot(rot_exp(s * rot_log(dot(R1, R0.T))), R0)


# ---------------------------------------------------------------------------
# Right-logarithmic derivative and the Square Root Velocity Transform
# ---------------------------------------------------------------------------

def right_log(c, I):
    """
    Discrete right-logarithmic derivative delta^r(c) = log(dc . c^-1).

    Motion-capture data is a list of rotation matrices (one per frame), not
    a continuous curve, so there is no literal time-derivative to take.
    Treating the data as piecewise-constant in the Lie algebra (interpolate
    consecutive frames via `rot_exp`/`rot_log`) lets this be computed
    exactly per frame: `rot_log(c[i+1] @ c[i].T) / dt` is the (constant)
    right-logarithmic derivative on that segment -- the angular velocity of
    the joint, expressed in a single fixed frame (so(3)) instead of each
    frame's own, different tangent space.

    Recurses automatically over multi-joint curves (shape d x n x 3 x 3).

    Parameters
    ----------
    c : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        A curve (or multi-joint curve) of rotation matrices, one per frame.
    I : ndarray, shape (n,)
        The sample times (parameterization) corresponding to each frame.

    Returns
    -------
    ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        The right-logarithmic derivative at each frame (so(3)-valued). The
        last frame is left as zero, since there is no "next frame" to
        difference against.
    """
    if len(c.shape) == 4:
        return array([right_log(curve, I) for curve in c])

    n_frames = c.shape[0]
    q = zeros((n_frames, 3, 3))
    for i in range(n_frames - 1):
        q[i] = rot_log(dot(c[i + 1], c[i].T)) / (I[i + 1] - I[i])
    return q


def SRVT(c, I):
    """
    Square Root Velocity Transform: Imm(I, SO(3)) -> C(I, so(3) \\ 0).

    Defined as R(c) = delta^r(c) / sqrt(||delta^r(c)||), i.e. the right-log
    derivative from `right_log`, additionally normalized by the square root
    of its own norm. This particular scaling is what makes R equivariant
    under reparameterization (R(c o phi) transforms in a simple, predictable
    way), which is exactly what makes the pulled-back L2 metric on SRVT
    representations reparameterization-invariant -- the whole reason this
    transform is used instead of comparing curves directly.

    Recurses automatically over multi-joint curves (shape d x n x 3 x 3).

    Parameters
    ----------
    c : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        A curve (or multi-joint curve) of rotation matrices, one per frame.
    I : ndarray, shape (n,)
        The sample times (parameterization) corresponding to each frame.

    Returns
    -------
    ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        The SRVT representation at each frame. Where the right-log
        derivative's norm is below `TOL` (near-zero angular velocity,
        e.g. a standstill), the normalization falls back to dividing by 1
        instead of 0.
    """
    if len(c.shape) == 4:
        return array([SRVT(curve, I) for curve in c])

    n_frames = c.shape[0]
    q = zeros((n_frames, 3, 3))
    for i in range(n_frames - 1):
        v = rot_log(dot(c[i + 1], c[i].T)) / (I[i + 1] - I[i])
        n = sqrt(norm(v))
        if n < TOL:
            n = 1
        q[i] = v / n
    return q


def inverse_SRVT(q, I):
    """
    Reconstruct a curve in SO(3) from its SRVT representation.

    This is the inverse of `SRVT`: starting from the identity, it walks
    forward frame by frame, applying `rot_exp` of the (rescaled) SRVT value
    at each step. Reconstructed curves always start at the identity
    (c[0] = I), since the SRVT itself is translation-invariant and
    discards the curve's starting point.

    Recurses automatically over multi-joint curves (shape d x n x 3 x 3).

    Parameters
    ----------
    q : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        An SRVT-valued curve (or multi-joint curve), as produced by `SRVT`.
    I : ndarray, shape (n,)
        The sample times (parameterization) corresponding to each frame.

    Returns
    -------
    ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        The reconstructed curve of rotation matrices, starting at the
        identity.
    """
    if len(q.shape) == 4:
        return array([inverse_SRVT(curve, I) for curve in q])

    n_frames = q.shape[0]
    c = zeros((n_frames, 3, 3))
    c[0] = eye(3)
    for i in range(n_frames - 1):
        v = (I[i + 1] - I[i]) * norm(q[i]) * q[i]
        c[i + 1] = dot(rot_exp(v), c[i])
    return c


# ---------------------------------------------------------------------------
# Matrix form <-> vector form
# ---------------------------------------------------------------------------

def skew_to_vector(q):
    """
    Convert an so(3)-valued curve from matrix (hatted) form to flat
    vector form, via `hatinv`.

    d x n x 3 x 3  --(hatinv)-->  d x n x 3  --(swap axes)-->  n x d x 3
                   --(reshape)-->  n x 3*d

    Parameters
    ----------
    q : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        An so(3)-valued curve in matrix form (single joint or multi-joint).

    Returns
    -------
    ndarray, shape (n, 3) or (n, 3*d)
        The same curve in flat vector form, one row per frame.

    Raises
    ------
    Exception
        If `q` does not look like a matrix-form so(3) curve (e.g. it has
        already been converted to vector form).
    """
    if len(q.shape) == 4 and is_3x3_matrix(q):
        d, n, _, _ = q.shape
        return hatinv(q).swapaxes(0, 1).reshape(n, 3 * d)
    if len(q.shape) == 3 and is_3x3_matrix(q):
        return hatinv(q)

    raise Exception("Already vectorized in skew_to_vector")
