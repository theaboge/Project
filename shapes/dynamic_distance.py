"""
Dynamic programming for time-alignment of curves in SO(3)^d.
The main entry point is `dynamic_distance`, which computes the DP distance between two curves, 
and `find_optimal_diffeomorphism`, which returns the optimal reparameterization of one curve onto another.

`find_optimal_diffeomorphism` uses a compiled (numba) version of the DP, `dynamic_fast`. The original pure-Python version
(`dynamic`, `local_cost`, `find_optimal_diffeomorphism_reference`) is kept as the reference it is tested against
(see tests/test_dynamic_distance.py). Both compute the same costs; they can only differ in which of two equal-cost paths is chosen.
"""

from numpy import zeros, inf, array, interp, sqrt, concatenate, linspace, array_equal, unique, searchsorted, full, int32
from functools import partial
from numba import njit


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


def find_optimal_diffeomorphism_reference(q0, q1, I0, I1, depth):
    """
    The original pure-Python version of `find_optimal_diffeomorphism`. Very slow (minutes to an hour per pair at
    depth 9); kept as the reference that `find_optimal_diffeomorphism` is tested against.

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


@njit(cache=True)
def segment_cost(G, n0, n1, I, k, l, i, j):
    """
    Compiled equivalent of `local_cost`: the same value, computed without any array allocation.

    `local_cost` takes the squared L2 distance between q0 on [I[k], I[i]] and q1[l:j+1] rescaled onto that interval 
    (on a uniform grid of j - l + 1 points). Walking the merged breakpoints of the two grids, each piece contributes
    its length times |q0[p] - s q1[r]|^2 = |q0[p]|^2 + s^2 |q1[r]|^2 - 2 s q0[p].q1[r], and the three terms are looked up in
    precomputed tables instead of summed over the k dimensions every time.

    Parameters
    ----------
    G : ndarray, shape (M, M)
        G[p, r] = q0[p] . q1[r].
    n0, n1 : ndarray, shape (M,)
        Squared norms of the rows of q0 and q1.
    I : ndarray, shape (M,)
        The shared sample times.
    k, l, i, j : int
        The DP step from (k, l) to (i, j).

    Returns
    -------
    float
        The local cost of the step, equal to `local_cost(k, l, i, j, ...)`.
    """
    s2 = (I[j] - I[l]) / (I[i] - I[k])
    s = sqrt(s2)
    m0 = i - k + 1
    m1 = j - l + 1
    start = I[k]
    step = (I[i] - I[k]) / (m1 - 1) if m1 > 1 else 0.0

    p = 0  # position in the q0 grid I[k..i], relative to k
    r = 0  # position in the uniform q1 grid, relative to l
    t = start
    total = 0.0
    while True:
        next0 = I[k + p + 1] if p + 1 < m0 else inf
        next1 = start + (r + 1) * step if r + 1 < m1 else inf
        nxt = min(next0, next1)
        if nxt == inf:
            break
        if nxt > t:
            total += (nxt - t) * (n0[k + p] + s2 * n1[l + r] - 2.0 * s * G[k + p, l + r])
            t = nxt
        if next0 <= nxt:
            p += 1
        if next1 <= nxt:
            r += 1
    return total


@njit(cache=True)
def dynamic_fast(G, n0, n1, I, depth):
    """
    Compiled equivalent of `dynamic`: fills the DP cost matrix and the predecessor of every grid point.

    Parameters
    ----------
    G, n0, n1, I :
        As in `segment_cost`.
    depth : int
        Search-window size, as in `dynamic`.

    Returns
    -------
    A : ndarray, shape (M, M)
        The DP cost matrix, as in `dynamic`.
    pointers : ndarray of int, shape (M, M, 2)
        pointers[i, j] = (k, l), the optimal predecessor of (i, j); (-1, -1) where no finite-cost path reaches (i, j).
    """
    M = I.shape[0]
    A = full((M, M), inf)
    A[0, 0] = 0.0
    pointers = full((M, M, 2), -1, dtype=int32)

    for i in range(1, M):
        for j in range(1, M):
            min_cost = inf
            best_k = -1
            best_l = -1
            for k in range(max(0, i - depth), i):
                for l in range(max(0, j - depth), j):
                    prev = A[k, l]
                    if prev == inf:
                        continue
                    cost = segment_cost(G, n0, n1, I, k, l, i, j) + prev
                    if min_cost > cost:
                        min_cost = cost
                        best_k = k
                        best_l = l
            A[i, j] = min_cost
            pointers[i, j, 0] = best_k
            pointers[i, j, 1] = best_l

    return A, pointers


def find_optimal_diffeomorphism(q0, q1, I0, I1, depth):
    """
    Find the optimal reparameterization of one curve onto another, using
    dynamic programming to minimize the L2 metric between the two curves in flat vector form.

    Same result as `find_optimal_diffeomorphism_reference`, but with the dot products between the two curves
    computed once (`G = q0 q1^T`) and the DP loop compiled, which is several hundred times faster.

    Parameters
    ----------
    q0, q1 : ndarray, shape (n0, k) and (n1, k)
        The two curves in flat vector form to align. `q0` is held fixed;
        the returned reparameterization applies to `q1`.
    I0, I1 : ndarray, shape (n0,) and (n1,)
        The corresponding sample times for `q0`/`q1`.
    depth : int
        Search-window size passed to `dynamic_fast`.

    Returns
    -------
    ndarray, shape (n1,)
        `I1_new`: the new sample times phi(I1), i.e. the optimal
        reparameterization of `q1`'s original timestamps.

    Raises
    ------
    ValueError
        If no finite-cost path reaches some point on the way back to (0, 0), see `reconstruct`.
    """
    I, q0_new, q1_new = create_shared_parameterization(q0, q1, I0, I1)
    M = I.shape[0]

    G = q0_new @ q1_new.T
    n0 = (q0_new ** 2).sum(axis=1)
    n1 = (q1_new ** 2).sum(axis=1)
    A, pointers = dynamic_fast(G, n0, n1, I, depth)

    # Backtrack, stopping at the first row/column like `reconstruct` (the only finite-cost point there is (0, 0)).
    path = [(M - 1, M - 1)]
    while path[-1][0] > 0 and path[-1][1] > 0:
        i, j = path[-1]
        k, l = pointers[i, j]
        if k < 0:
            raise ValueError(
                f"No finite-cost alignment reaches grid point {path[-1]} -- "
                "depth is too small for how much these two curves need to "
                "be stretched here. Try a larger depth."
            )
        path.append((int(k), int(l)))
    path.reverse()

    x = array([p[0] for p in path]) / float(M - 1)
    y = array([p[1] for p in path]) / float(M - 1)

    return interp(I1, x, y)
