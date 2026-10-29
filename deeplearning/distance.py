"""
Deep learning distance between two curves in SO(3)^d: the DL counterpart of `shapes.curves.dynamic_distance`.
"""

from numpy import linspace

from shapes.curves import distance, reparameterize
from shapes.so3_math import skew_to_vector, SRVT
from .optimal_phi import find_optimal_diffeomorphism_dl


def dl_distance(c0, c1, layers=10, functions=10, k=1024, max_iter=500,
                optimizer="lbfgs", lr=1e-2, steps=3000, schedule="constant", smoothing=(1,)):
    """
    Deep learning distance between two curves in SO(3)^d, with time alignment found by training a reparameterization network.
    Follows the same four steps as `dynamic_distance`; only the search for the reparameterization differs.

    Parameters
    ----------
    c0 : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        A curve (or multi-joint curve) of rotation matrices.
    c1 : ndarray, shape (n, 3, 3) or (d, n, 3, 3)
        Another curve (or multi-joint curve) of rotation matrices.
    layers : int, optional
        Number of layers L of the reparameterization network. Default is 10.
    functions : int, optional
        Number of basis functions M in each layer. Default is 10.
    k : int, optional
        Number of points in [0, 1] where the loss is evaluated. Default is 1024.
    max_iter : int, optional
        Maximum number of L-BFGS iterations. Default is 500.
    optimizer, lr, steps, schedule, smoothing :
        The optimization method and its settings, see `find_optimal_diffeomorphism_dl`. The defaults are the original L-BFGS setup.

    Returns
    -------
    float
        The deep learning distance between the two curves.
    """
    is_multi = len(c0.shape) == 4
    I0 = linspace(0, 1, c0.shape[1 if is_multi else 0])
    I1 = linspace(0, 1, c1.shape[1 if is_multi else 0])

    q0 = skew_to_vector(SRVT(c0, I0))
    q1 = skew_to_vector(SRVT(c1, I1))

    I1_new = find_optimal_diffeomorphism_dl(q0, q1, I0, I1, layers=layers, functions=functions, k=k, max_iter=max_iter,
                                            optimizer=optimizer, lr=lr, steps=steps, schedule=schedule, smoothing=smoothing)
    c1_new = reparameterize(I1_new, I1, c1)
    return distance(c0, c1_new, I0=I0, I1=I1)
