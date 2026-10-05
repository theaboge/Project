"""
Operations on whole curves in SO(3)^d: the SRVT-only distance (no time
alignment), the DP-optimized distance (with time alignment), geodesic
interpolation, reparameterization, and the piecewise-linear lift used to
prepare paths for signature computation.
"""

from numpy import linspace, array, dot, zeros
from .dynamic_distance import find_optimal_diffeomorphism, L2_metric
from .transformations import skew_to_vector, SRVT, inverse_SRVT
from .transformations import interpolate as tf_interpolate


def distance(c0, c1, I0=None, I1=None):
    """SRVT-only distance d_P*(c0, c1): no time-alignment search, just the
    L2 distance between the two curves' SRVT representations under their
    given (or default, evenly-spaced) parameterizations."""
    is_multi = len(c0.shape) == 4
    if I0 is None:
        I0 = linspace(0, 1, c0.shape[1 if is_multi else 0])
    if I1 is None:
        I1 = linspace(0, 1, c1.shape[1 if is_multi else 0])

    q0 = skew_to_vector(SRVT(c0, I0))
    q1 = skew_to_vector(SRVT(c1, I1))

    return L2_metric(q0, q1, I0, I1)


def dynamic_distance(c0, c1, depth=5):
    """SRVT + dynamic-programming distance d_S*(c0, c1): find the
    reparameterization of c1 minimizing the SRVT distance to c0 (via
    `dynamic_distance.find_optimal_diffeomorphism`), reparameterize, then
    measure the remaining distance. Equation (4): dS := inf_phi dP(c0, c1 o phi).

    `depth` is the DP search-window size -- the main cost/accuracy knob,
    roughly O(depth^3) cost.
    """
    is_multi = len(c0.shape) == 4
    I0 = linspace(0, 1, c0.shape[1 if is_multi else 0])
    I1 = linspace(0, 1, c1.shape[1 if is_multi else 0])

    q0 = skew_to_vector(SRVT(c0, I0))
    q1 = skew_to_vector(SRVT(c1, I1))

    I1_new = find_optimal_diffeomorphism(q0, q1, I0, I1, depth)
    c1_new = reparameterize(I1_new, I1, c1)
    return distance(c0, c1_new, I0=I0, I1=I1)


def move_origin_to_zero(c):
    """Right-translate a curve so it starts at the identity: c(0) = I.
    Required for the P* restriction the SRVT relies on. Recurses over
    multi-joint curves."""
    if len(c.shape) == 4:
        return array([move_origin_to_zero(e) for e in c])

    return array([dot(matrix, c[0].T) for matrix in c])


def interpolate(c0, c1, s):
    """Linearly interpolate between two curves via SRVT / inverse SRVT
    (the shape-space analogue of straight-line interpolation)."""
    if len(c0.shape) == 4:
        d, N, _, _ = c0.shape
        I = linspace(0, 1, N)
        q = array([(1 - s) * SRVT(c0[i], I) + s * SRVT(c1[i], I) for i in range(d)])
        return array([inverse_SRVT(q[i], I) for i in range(d)])
    else:
        I = linspace(0, 1, c0.shape[0])
        q = (1 - s) * SRVT(c0, I) + s * SRVT(c1, I)
        return inverse_SRVT(q, I)


def reparameterize(I_new, I, c):
    """Reparameterize curve c at new sample times I_new: for each new time,
    find its subinterval in the original grid I and geodesically
    interpolate c between those two endpoints. Recurses over multi-joint
    curves."""
    if len(c.shape) == 4:
        return array([reparameterize(I_new, I, e) for e in c])

    c_new = zeros(c.shape)
    N = c.shape[0]

    j = 0
    for i in range(N):
        phi = I_new[i]
        while not I[j % N] <= phi <= I[(j + 1) % N]:
            j += 1
            if j > 5 * N:
                raise Exception("Could not place t_0 <= phi < t_1 in reparameterize")

        s = (phi - I[j % N]) / (I[(j + 1) % N] - I[j % N])
        c_new[i] = tf_interpolate(c[j % N], c[(j + 1) % N], s)

    return c_new


# L2_metric used to be defined here too (identical to dynamic_distance.py's
# version) -- now imported from there instead, see the import block above.
