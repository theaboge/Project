"""
Operations on whole curves in SO(3)^d: the SRVT-only distance (no time alignment), the DP-optimized distance (with time alignment), 
moving a curve's origin to the identity, and reparameterization.
"""

from numpy import linspace, array, dot, zeros
from .dynamic_distance import find_optimal_diffeomorphism, L2_metric
from .so3_math import skew_to_vector, SRVT
from .so3_math import interpolate as tf_interpolate


def distance(c0, c1, I0=None, I1=None):
    """
    SRVT distance d_P(c0, c1) between two curves in SO(3)^d, with optional sample-time grids I0 and I1. 
    Recurses over multi-joint curves.

    Parameters
    ----------
    c0 : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        A curve (or multi-joint curve) of rotation matrices.
    c1 : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        Another curve (or multi-joint curve) of rotation matrices.
    I0 : ndarray, shape (n,), optional
        Sample times for `c0`. If not provided, assumed to be uniform in [0, 1].
    I1 : ndarray, shape (n,), optional      
        Sample times for `c1`. If not provided, assumed to be uniform in [0, 1].
    
    Returns 
    -------
    float
        The SRVT distance between the two curves.
    """

    is_multi = len(c0.shape) == 4
    if I0 is None:
        I0 = linspace(0, 1, c0.shape[1 if is_multi else 0])
    if I1 is None:
        I1 = linspace(0, 1, c1.shape[1 if is_multi else 0])

    q0 = skew_to_vector(SRVT(c0, I0))
    q1 = skew_to_vector(SRVT(c1, I1))

    return L2_metric(q0, q1, I0, I1)


def dynamic_distance(c0, c1, depth=5):
    """
    Dynamic programming distance between two curves in SO(3)^d, with time alignment. Recurses over multi-joint curves.

    Parameters
    ----------
    c0 : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        A curve (or multi-joint curve) of rotation matrices.
    c1 : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        Another curve (or multi-joint curve) of rotation matrices.
    depth : int, optional   
        The depth of the dynamic programming search. Higher values are more accurate but slower. Default is 5.

    Returns
    -------
    float
        The dynamic programming distance between the two curves.
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
    """
    Move the origin of a curve in SO(3)^d to the identity. Recurses over multi-joint curves.
    
    Parameters
    ----------
    c : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        A curve (or multi-joint curve) of rotation matrices.

    Returns
    -------
    ndarray
        The curve with its origin moved to the identity.
    """
    if len(c.shape) == 4:
        return array([move_origin_to_zero(e) for e in c])

    return array([dot(matrix, c[0].T) for matrix in c])


def reparameterize(I_new, I, c):
    """
    Reparameterize a curve in SO(3)^d to a new sample-time grid. Recurses over multi-joint curves.
    
    Parameters
    ----------      
    I_new : ndarray, shape (n,)
        The new sample-time grid.
    I : ndarray, shape (n,)
        The original sample-time grid.
    c : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        A curve (or multi-joint curve) of rotation matrices.

    Returns
    -------
    ndarray
        The reparameterized curve.
    """
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
