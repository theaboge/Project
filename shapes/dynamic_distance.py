"""
Dynamic programming for time-alignment of curves in SO(3)^d.
The main entry point is `dynamic_distance`, which computes the DP distance between two curves, 
and `find_optimal_diffeomorphism`, which returns the optimal reparameterization of one curve onto another.
"""

from numpy import zeros, inf, array, interp, sqrt, concatenate, linspace, array_equal, unique, searchsorted
from functools import partial


def create_shared_parameterization(q0, q1, I0, I1):
    """
    Create a shared parameterization for two curves in flat vector form, 
    by merging their sample times and re-expressing the curves on that shared grid.

    This is a prerequisite for the dynamic programming distance computation, 
    which requires both curves to be defined on the same time grid.

    Parameters
    ----------
    q0, q1 : ndarray, shape (n0, k) and (n1, k)
        The two curves in flat vector form (k-dimensional, piecewise constant between consecutive sample times).
    I0, I1 : ndarray, shape (n0,) and (n1,)
        The corresponding sample times for `q0`/`q1`.   
    
    Returns
    -------
    I : ndarray, shape (m,)
        The merged sample times, sorted and unique.
    q0_new, q1_new : ndarray, shape (m, k)  
        The two curves re-expressed on the shared grid `I`, with piecewise-constant interpolation.
    """
    if array_equal(I0, I1):
        return I0, q0, q1

    I = unique(concatenate((I0, I1)))
    q0_new = zeros((I.shape[0], q0.shape[1]))
    q1_new = zeros((I.shape[0], q0.shape[1]))

    idx0 = searchsorted(I0, I[:-1], side='right') - 1
    idx1 = searchsorted(I1, I[:-1], side='right') - 1
    q0_new[:-1] = q0[idx0]
    q1_new[:-1] = q1[idx1]

    return I, q0_new, q1_new


def dynamic(local_cost, M, depth):
    """
    Fill the dynamic programming cost matrix for two curves in flat vector form, using a given local cost function and search depth.
    
    Parameters
    ----------
    local_cost : callable       
        A function that computes the local cost of a DP step from (k, l) to (i, j).
    M : int
        The size of the DP grid (the number of sample points in the shared parameterization).
    depth : int
        The search-window size for the DP algorithm, limiting how far back along each axis to consider predecessors.

    Returns
    -------
    pointers : dict
        Maps each grid point (i, j) to its optimal predecessor (k, l).
    A : ndarray, shape (M, M)
        The filled DP cost matrix, where A[i, j] is the minimum cost to reach (i, j) from (0, 0).
        (0, 0) is the only point with a built-in zero cost; every other point on the first
        row/column starts at infinity and only gets a finite cost once a real path reaches
        it, so the DP can't treat "never computed" as "free" and skip a stretch of either
        curve for nothing.
    """
    A = zeros((M, M))
    A[:, :] = inf
    A[0, 0] = 0.0
    pointers = dict()

    for i in range(1, M):
        for j in range(1, M):
            min_cost = inf
            best_pred = None
            for pred in predecessors(i, j, depth):
                k, l = pred
                cost = local_cost(k, l, i, j) + A[k, l]
                if min_cost > cost:
                    min_cost = cost
                    best_pred = pred
            A[i, j] = min_cost
            pointers[(i, j)] = best_pred

    return pointers, A


def local_cost(k, l, i, j, q0, q1, I):
    """
    Compute the local cost of a DP step from (k, l) to (i, j) for two curves in flat vector form, 
    using the L2 metric between the corresponding segments of the curves.

    Expensive to call, so this is the hottest function in the whole DP run. 
    It is called O(M^2 * depth^2) times, where M is the number of sample points in the shared parameterization and depth is the search-window size.

    Parameters
    ----------
    k, l : int
        The predecessor grid point (k, l).
    i, j : int
        The current grid point (i, j).
    q0, q1 : ndarray, shape (n0, k) and (n1, k)
        The two curves in flat vector form (k-dimensional, piecewise constant between consecutive sample times).
    I : ndarray, shape (m,)
        The shared sample times for `q0`/`q1`, as returned by `create_shared_parameterization`.

    Returns
    -------
    float
        The local cost of the DP step from (k, l) to (i, j), computed as the L2 metric between the segments of `q0` and `q1` 
        corresponding to the intervals [I[k], I[i]] and [I[l], I[j]].
    """
    return L2_metric(
        q0[k:i + 1],
        sqrt((I[j] - I[l]) / (I[i] - I[k])) * q1[l:j + 1],
        I[k:i + 1],
        linspace(I[k], I[i], (j - l + 1))
    ) ** 2


def L2_metric(q0, q1, I0, I1):
    """
    Compute the L2 metric between two curves in flat vector form, with optional sample-time grids I0 and I1. 
    This is the integral of the squared norm of the difference between the two curves, weighted by the sample-time intervals.
    
    Parameters
    ----------
    q0 : ndarray, shape (n0, k)
        The first curve in flat vector form (k-dimensional, piecewise constant between consecutive sample times).
    q1 : ndarray, shape (n1, k)
        The second curve in flat vector form (k-dimensional, piecewise constant between consecutive sample times).
    I0 : ndarray, shape (n0,), optional
        Sample times for `q0`. If not provided, assumed to be uniform in [0, 1].
    I1 : ndarray, shape (n1,), optional      
        Sample times for `q1`. If not provided, assumed to be uniform in [0, 1].
        
    Returns
    -------
    float
        The L2 metric between the two curves, computed as the integral of the squared norm 
        of the difference between the two curves, weighted by the sample-time intervals.
    """
    if array_equal(I0, I1):
        diffs = I0[1:] - I0[:-1]
        sq_norms = ((q0[:-1] - q1[:-1]) ** 2).sum(axis=1)
        return sqrt((diffs * sq_norms).sum())

    I = unique(concatenate((I0, I1)))

    idx0 = searchsorted(I0, I[:-1], side='right') - 1
    idx1 = searchsorted(I1, I[:-1], side='right') - 1
    diffs = I[1:] - I[:-1]
    sq_norms = ((q0[idx0] - q1[idx1]) ** 2).sum(axis=1)

    return sqrt((diffs * sq_norms).sum())


def reconstruct(pointers, M, N):
    """
    Backtrack through the DP `pointers` to recover the optimal path from (0, 0) to (M, N).

    Parameters
    ----------
    pointers : dict
        Maps each grid point to its optimal predecessor, as returned by
        `dynamic`.
    M, N : int
        The grid point to backtrack from (typically the last point of the
        grid, i.e. `M == N == grid size - 1`).

    Returns
    -------
    list of (int, int)
        The optimal path through the grid, from (0, 0) to (M, N), in
        forward order.

    Raises
    ------
    ValueError
        If no finite-cost path reaches some point on the way back to
        (0, 0) -- `depth` was too small for how much the two curves need
        to be stretched relative to each other near that point.
    """
    path = [(M, N)]
    while path[-1] in pointers:
        pred = pointers[path[-1]]
        if pred is None:
            raise ValueError(
                f"No finite-cost alignment reaches grid point {path[-1]} -- "
                "depth is too small for how much these two curves need to "
                "be stretched here. Try a larger depth."
            )
        path.append(pred)

    path.reverse()
    return path


def predecessors(i, j, depth):
    """
    Generate all candidate predecessor grid points (k, l) for a given grid point (i, j), within a specified search depth.

    Parameters
    ----------
    i, j : int
        The current grid point.
    depth : int
        How far back along each axis to search for predecessors.

    Yields
    ------
    (int, int)
        Each candidate predecessor grid point (k, l).
    """
    for k in range(max(0, i - depth), i):
        for l in range(max(0, j - depth), j):
            yield (k, l)


def find_optimal_diffeomorphism(q0, q1, I0, I1, depth):
    """
    Find the optimal reparameterization of one curve onto another, using
    dynamic programming to minimize the L2 metric between the two curves in flat vector form.

    Parameters
    ----------
    q0, q1 : ndarray, shape (n0, k) and (n1, k)
        The two curves in flat vector form to align. `q0` is held fixed;
        the returned reparameterization applies to `q1`.
    I0, I1 : ndarray, shape (n0,) and (n1,)
        The corresponding sample times for `q0`/`q1`.
    depth : int
        Search-window size passed to `dynamic`/`predecessors`.

    Returns
    -------
    ndarray, shape (n1,)
        `I1_new`: the new sample times phi(I1), i.e. the optimal
        reparameterization of `q1`'s original timestamps.
    """
    I, q0_new, q1_new = create_shared_parameterization(q0, q1, I0, I1)
    M = I.shape[0]
    local_cost_partial = partial(local_cost, q0=q0_new, q1=q1_new, I=I)

    pointers, A = dynamic(local_cost_partial, M, depth)
    path = reconstruct(pointers, M - 1, M - 1)

    # Construct reparametrization
    x = array([p[0] for p in path]) / float(M - 1)
    y = array([p[1] for p in path]) / float(M - 1)

    return interp(I1, x, y)
