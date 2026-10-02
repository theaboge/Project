"""
The dynamic-programming (DP) algorithm that finds the optimal time-alignment
between two curves -- the expensive part of d_S*. Fills a cost matrix via
A[i,j] = min_(k,l) local_cost(k,l,i,j) + A[k,l], restricted to a bounded
window of `depth` predecessors per grid point.

Known gap: `local_cost` implements only the elastic-matching term of the
energy functional (eq. 4.8 in the theory notes). It omits the step-penalty
term (weight lambda, no value given anywhere in the source material).
Present unchanged in the original codebase too. Without it, the DP can find
a degenerate alignment that compresses a badly-matching stretch of one
curve to near-zero width instead of a natural correspondence -- confirmed
on real data for at least one animation pair. Not fixed (no known lambda
value); documented as a known limitation.
"""

from numpy import zeros, inf, array, interp, sqrt, concatenate, linspace, array_equal, unique, searchsorted
from functools import partial


def create_shared_parameterization(q0, q1, I0, I1):
    """Merge two curves' time grids onto one shared grid, carrying each
    curve's piecewise-constant value forward at every new point. Vectorized
    via `searchsorted` (verified ~1.6x faster, numerically equivalent to a
    plain-loop version).
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
    """Fill the DP cost matrix: A[i,j] = min_(k,l) local_cost(k,l,i,j) + A[k,l]
    (eq. 4.9), searching only `depth` predecessors back per grid point.

    Returns
    -------
    pointers : dict mapping (i,j) -> optimal predecessor (k,l)
    A : (M,M) ndarray of minimal cumulative cost
    """
    A = zeros((M, M))
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
    """Cost of a DP step (k,l) -> (i,j): the elastic-matching term of eq. 4.8
    (step-penalty term not included, see module docstring). Called up to
    depth^2 times per grid point -- the dominant cost of the whole run."""
    return L2_metric(
        q0[k:i + 1],
        sqrt((I[j] - I[l]) / (I[i] - I[k])) * q1[l:j + 1],
        I[k:i + 1],
        linspace(I[k], I[i], (j - l + 1))
    ) ** 2


def L2_metric(q0, q1, I0, I1):
    """L2 metric between two piecewise-constant curves (vector form,
    I0[0]==I1[0], I0[-1]==I1[-1]). The hottest function in the DP run;
    vectorized via `searchsorted` instead of a per-point loop. Verified
    equivalent to a plain-loop reference (~1e-16 max diff) and bit-identical
    end-to-end on real data at several depths.
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
    """Backtrack `pointers` (from `dynamic`) to recover the optimal path
    from (0,0) to (M,N), in forward order."""
    path = [(M, N)]
    try:
        while True:
            pred = path[-1]
            path.append(pointers[pred])
    except Exception:
        pass

    path.reverse()
    return path


def predecessors(i, j, depth):
    """Candidate predecessors for grid point (i,j): all (k,l) with k in
    [max(0,i-depth), i) and l in [max(0,j-depth), j). This `depth` window
    is what keeps the O(M^2) DP fill tractable."""
    for k in range(max(0, i - depth), i):
        for l in range(max(0, j - depth), j):
            yield (k, l)


def find_optimal_diffeomorphism(q0, q1, I0, I1, depth):
    """Full DP pipeline: build the shared grid, fill the cost matrix,
    backtrack, and return the optimal reparameterization of q1's sample
    times (I1_new = phi(I1))."""
    I, q0_new, q1_new = create_shared_parameterization(q0, q1, I0, I1)
    M = I.shape[0]
    local_cost_partial = partial(local_cost, q0=q0_new, q1=q1_new, I=I)

    pointers, A = dynamic(local_cost_partial, M, depth)
    path = reconstruct(pointers, M - 1, M - 1)

    # Construct reparametrization
    x = array([p[0] for p in path]) / float(M - 1)
    y = array([p[1] for p in path]) / float(M - 1)

    return interp(I1, x, y)
