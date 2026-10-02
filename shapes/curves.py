"""
Operations on whole curves in SO(3)^d: the SRVT-only distance (no time
alignment), the DP-optimized distance (with time alignment), geodesic
interpolation, reparameterization, and the piecewise-linear lift used to
prepare paths for signature computation.
"""

from numpy import linspace, array, dot, zeros, array_equal, searchsorted
from numpy import sqrt, unique, concatenate as np_concatenate
from .dynamic_distance import find_optimal_diffeomorphism
from .transformations import skew_to_vector, SRVT, inverse_SRVT
from .transformations import interpolate as tf_interpolate, gradient_close_curve


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


def reparameterize_to_optimal(c0, c1, depth=5):
    """Find the optimal time-alignment of c1 onto c0 and return c1
    reparameterized accordingly (distance not computed)."""
    is_multi = len(c0.shape) == 4
    I0 = linspace(0, 1, c0.shape[1 if is_multi else 0])
    I1 = linspace(0, 1, c1.shape[1 if is_multi else 0])

    q0 = skew_to_vector(SRVT(c0, I0))
    q1 = skew_to_vector(SRVT(c1, I1))

    I1_new = find_optimal_diffeomorphism(q0, q1, I0, I1, depth)
    return reparameterize(I1_new, I1, c1)


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


def close(c, iterations=25, alpha=0.025, move_origin=False):
    """Deform an open curve towards a closed one via gradient descent on
    its SRVT representation. Not used in the current distance pipeline."""
    if move_origin:
        c = move_origin_to_zero(c)

    is_multi = len(c.shape) == 4
    I = linspace(0, 1, c.shape[1 if is_multi else 0])

    q = SRVT(c, I)
    for k in range(iterations):
        q -= alpha * gradient_close_curve(q, I)

    return inverse_SRVT(q, I)


def L2_metric(q0, q1, I0, I1):
    """L2 metric between two piecewise-constant curves: sqrt(integral(||q0-q1||^2)).
    Assumes vector form and I0[0]==I1[0], I0[-1]==I1[-1]. Builds a shared
    grid when I0 != I1 and carries each curve's value forward onto it.

    Vectorized via `searchsorted`, same as the identical function in
    `dynamic_distance.py` -- see that file's docstring for the reasoning.
    This copy used to crash on I0 == I1 (a plain-Python-loop version called
    numpy's `sum` on a generator, which recent numpy rejects); the
    vectorized form doesn't have that problem.
    """
    if array_equal(I0, I1):
        diffs = I0[1:] - I0[:-1]
        sq_norms = ((q0[:-1] - q1[:-1]) ** 2).sum(axis=1)
        return sqrt((diffs * sq_norms).sum())

    I = unique(np_concatenate((I0, I1)))

    idx0 = searchsorted(I0, I[:-1], side='right') - 1
    idx1 = searchsorted(I1, I[:-1], side='right') - 1
    diffs = I[1:] - I[:-1]
    sq_norms = ((q0[idx0] - q1[idx1]) ** 2).sum(axis=1)

    return sqrt((diffs * sq_norms).sum())


def lift_piece_wise_constant(q, I):
    """Lift a piecewise-constant curve to the piecewise-linear path format
    `iisignature` expects, by accumulating q[i]*(I[i+1]-I[i]) from x(0)=0."""
    x = zeros(q.shape)

    for i in range(q.shape[1] - 1):
        x[i + 1] = q[i] * (I[i + 1] - I[i]) + x[i]

    return x
