"""
Core Lie-group toolbox for SO(3): hat/hat-inverse (R^3 <-> so(3)), log/exp
maps, geodesic interpolation, the right-logarithmic derivative, and SRVT.

Shapes:
    c in Imm(I, SO(3))   : n x 3 x 3      q in C(I, so(3))   : n x 3 x 3
    c in Imm(I, SO(3)^d) : d x n x 3 x 3  q in C(I, so(3)^d) : d x n x 3 x 3
    vector form           q in C(I, so(3))   : n x 3, q in C(I, so(3)^d) : n x 3*d
"""

from numpy import array, isnan, zeros, eye, trace, dot
from numpy import sin, cos, arccos, sqrt
from .helpers import TOL, norm, is_3x3_matrix


def Rx(a):
    """Rotation matrix for angle `a` (rad) about the x-axis."""
    s, c = sin(a), cos(a)
    return array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    """Rotation matrix for angle `a` (rad) about the y-axis."""
    s, c = sin(a), cos(a)
    return array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    """Rotation matrix for angle `a` (rad) about the z-axis."""
    s, c = sin(a), cos(a)
    return array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def hat(v):
    """R^3 -> so(3): vector to skew-symmetric matrix (hat(v) @ x == v cross x).
    Recurses over curves (n,3) and multi-joint curves (d,n,3)."""
    if len(v.shape) == 4:
        return array([hat(curve) for curve in v])
    if len(v.shape) == 3:
        return array([hat(vector) for vector in v])
    return array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def hatinv(M):
    """so(3) -> R^3: inverse of `hat`. Recurses the same way `hat` does."""
    if len(M.shape) == 4:
        return array([hatinv(curve) for curve in M])
    if len(M.shape) == 3:
        return array([hatinv(matrix) for matrix in M])
    return array([-M[1, 2], M[0, 2], -M[0, 1]])


def rot_log(R):
    """Matrix log SO(3) -> so(3) (Rodrigues' formula, inverted). Assumes R
    is close to the identity. Falls back to the antisymmetric part of R
    when the angle is ~0 or NaN (a known rounding-error case)."""
    theta = arccos(0.5 * (trace(R) - 1))
    if abs(theta) < TOL or isnan(theta):
        return 0.5 * (R - R.T)
    return (theta / sin(theta)) * 0.5 * (R - R.T)


def rot_exp(r):
    """Matrix exp so(3) -> SO(3) (Rodrigues' formula). Returns the identity
    when the rotation angle is ~0 or NaN."""
    theta = norm(r)
    if abs(theta) < TOL or isnan(theta):
        return eye(3)
    return eye(3) + (sin(theta) / theta) * r + ((1 - cos(theta)) / (theta ** 2)) * dot(r, r)


def Ad(A, X):
    """Adjoint action of A on X: A^T X A."""
    return dot(dot(A.T, X), A)


def interpolate(R0, R1, s=1):
    """Geodesic interpolation in SO(3) from R0 towards R1 (s=0 -> R0, s=1 -> R1)."""
    return dot(rot_exp(s * rot_log(dot(R1, R0.T))), R0)


def right_log(c, I):
    """Discrete right-logarithmic derivative: log(c[i+1] @ c[i].T) / dt per
    frame -- the angular velocity, expressed in a single fixed frame
    instead of each frame's own tangent space. Recurses over multi-joint
    curves. Last frame is left as zero (no next frame to difference)."""
    if len(c.shape) == 4:
        return array([right_log(curve, I) for curve in c])

    n_frames = c.shape[0]
    q = zeros((n_frames, 3, 3))
    for i in range(n_frames - 1):
        q[i] = rot_log(dot(c[i + 1], c[i].T)) / (I[i + 1] - I[i])
    return q


def SRVT(c, I):
    """Square Root Velocity Transform: R(c) = right_log(c) / sqrt(||right_log(c)||).
    This scaling makes R equivariant under reparameterization, which is why
    the pulled-back L2 metric on SRVT curves is reparameterization-invariant.
    Recurses over multi-joint curves."""
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
    """Reconstruct a curve in SO(3) from its SRVT representation, starting
    at the identity. Recurses over multi-joint curves."""
    if len(q.shape) == 4:
        return array([inverse_SRVT(curve, I) for curve in q])

    n_frames = q.shape[0]
    c = zeros((n_frames, 3, 3))
    c[0] = eye(3)
    for i in range(n_frames - 1):
        v = (I[i + 1] - I[i]) * norm(q[i]) * q[i]
        c[i + 1] = dot(rot_exp(v), c[i])
    return c


def skew_to_vector(q):
    """Matrix form -> flat vector form, via `hatinv`.
    d x n x 3 x 3 -> n x 3*d  (or  n x 3 x 3 -> n x 3)."""
    if len(q.shape) == 4 and is_3x3_matrix(q):
        d, n, _, _ = q.shape
        return hatinv(q).swapaxes(0, 1).reshape(n, 3 * d)
    if len(q.shape) == 3 and is_3x3_matrix(q):
        return hatinv(q)

    raise Exception("Already vectorized in skew_to_vector")


def gradient_close_curve(q, I):
    """Gradient step for deforming an open curve towards a closed one.
    Not used in the current distance pipeline. Recurses over multi-joint curves."""
    if len(q.shape) == 4:
        return array([gradient_close_curve(e, I) for e in q])

    c = inverse_SRVT(q, I)
    grad = zeros(q.shape)
    res = rot_log(c[-1])

    for i in range(q.shape[0]):
        tmp = dot(c[i], dot(res, c[i].T))
        n = norm(q[i])
        if n < TOL or isnan(n):
            continue
        grad[i] = n * tmp + trace(dot(tmp.T, q[i] / n)) * q[i]

    return grad
