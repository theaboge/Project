"""
The dynamic-programming (DP) algorithm that searches for the optimal
time-alignment between two curves -- the expensive part of the
reparameterization-invariant distance d_S*.

Implements the discretization from the theory notes: build a grid over the
two curves' shared time domain, define a local cost for stepping between
grid points, and fill a cost matrix A via the DP recursion
A[i,j] = min_(k,l) local_cost(k,l,i,j) + A[k,l], restricted to a bounded
window of `depth` predecessors per grid point to keep the cost in check.

Known gap: `local_cost` below implements only the elastic-matching integral
term of the energy functional (eq. 4.8 in the theory notes / thesis). It
omits the second term, a step penalty weighted by lambda > 0, summed over
every discretization point in the segment. No value for lambda is given in
the source material, and the thesis itself describes this term only in its
definition -- its own implementation section references the full energy
functional without flagging that only half of it is actually computed, in
the original code this is ported from as well as here. Without the
penalty, the DP can (and in practice does, for at least one pair in the
original dataset) find a degenerate alignment that compresses a
badly-matching stretch of one curve down to near-zero width to avoid
paying for the mismatch, rather than a natural one-to-one correspondence.
Left as a known, documented limitation rather than "fixed" with an
arbitrary lambda value.
"""

from numpy import zeros, inf, array, interp, sqrt, concatenate, linspace, array_equal, unique, searchsorted
from functools import partial


def create_shared_parameterization(q0, q1, I0, I1):
    """Merge two curves' time grids onto one shared grid, carrying each
    curve's piecewise-constant value forward at every new shared point.

    Since `I0` and `I1` are both increasing, and every point of `I0` (and
    `I1`) already appears as its own point in the merged grid `I` by
    construction, the index of the piece of `I0` active at a given point
    `I[k]` is exactly `searchsorted(I0, I[k], side='right') - 1`, and
    likewise for `I1`. This is computed for every point of `I` at once
    with a vectorized `searchsorted` call, rather than a Python-level loop
    that advances an index one step at a time -- see the module-level
    performance note in `dynamic_distance.py`'s history for why this
    matters (this function sits in the hot path, called once per curve
    pair).

    Parameters
    ----------
    q0, q1 : ndarray, shape (n0, k) and (n1, k)
        The two curves in flat vector form.
    I0, I1 : ndarray, shape (n0,) and (n1,)
        The corresponding sample times for `q0`/`q1`.

    Returns
    -------
    I : ndarray, shape (M,)
        The shared, merged, deduplicated grid (M = number of distinct
        points in `I0` union `I1`).
    q0_new, q1_new : ndarray, shape (M, k)
        `q0`/`q1`, re-expressed on the shared grid `I`. The last row of
        each is left as zero and never assigned -- matching the original,
        unvectorized implementation's exact behaviour rather than "fixing"
        it as a side effect of vectorizing.
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
    """Fill the dynamic-programming cost matrix via the recursion
    A[i,j] = min_(k,l) local_cost(k,l,i,j) + A[k,l].

    Equation (4.9) in the theory notes, restricted to a bounded window of
    predecessors per grid point (see `predecessors`) to keep the O(M^2) x
    O(depth^2) search tractable.

    Parameters
    ----------
    local_cost : callable
        A function `local_cost(k, l, i, j) -> float` giving the cost of
        stepping from grid point (k, l) to (i, j). Typically a `functools.
        partial` of the module-level `local_cost`, with `q0`, `q1`, `I`
        already bound.
    M : int
        The size of the (square) grid, i.e. the number of points in the
        shared parameterization.
    depth : int
        How far back (in grid points, along both axes) to search for a
        predecessor at each grid point. This is the main cost/accuracy
        knob: larger `depth` considers more candidate alignments per step,
        at roughly cubic extra cost (see `local_cost`'s docstring).

    Returns
    -------
    pointers : dict
        Maps each grid point `(i, j)` to its optimal predecessor `(k, l)`,
        for backtracking the optimal path (see `reconstruct`).
    A : ndarray, shape (M, M)
        The minimal cumulative energy to reach each grid point from (0, 0).
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
    """Cost of a single DP step from grid point (k, l) to (i, j): the
    elastic-matching term of the energy functional E(k,l;i,j) (eq. 4.8).

    This is called up to `depth^2` times for every one of the `M^2` grid
    points in the DP matrix, making it the single most expensive function
    in the whole pipeline -- effectively O(M^2 * depth^2) calls per curve
    pair, each itself doing O(depth) work inside `L2_metric`, for an
    overall O(M^2 * depth^3) cost. See the module docstring for the
    (documented, not fixed) gap between this and the full eq. 4.8, which
    also includes a step-penalty term this function does not compute.

    Parameters
    ----------
    k, l : int
        The predecessor grid point's indices, into `q0` and `q1`
        respectively.
    i, j : int
        The current grid point's indices, into `q0` and `q1` respectively.
    q0, q1 : ndarray, shape (M, dim)
        The two curves on the shared grid (as produced by
        `create_shared_parameterization`).
    I : ndarray, shape (M,)
        The shared grid's sample times.

    Returns
    -------
    float
        The squared L2 cost of the piecewise-linear step from (k, l) to
        (i, j).
    """
    return L2_metric(
        q0[k:i + 1],
        sqrt((I[j] - I[l]) / (I[i] - I[k])) * q1[l:j + 1],
        I[k:i + 1],
        linspace(I[k], I[i], (j - l + 1))
    ) ** 2


def L2_metric(q0, q1, I0, I1):
    """L2 metric between two piecewise-constant, vector-valued curves:
    sqrt(integral(||q0 - q1||^2)).

    Assumes `q0`, `q1` are in flat vector form, and `I0`, `I1` are
    increasing with matching endpoints: `I0[0] == I1[0]`, `I0[-1] == I1[-1]`.

    This is the hottest function in the whole DP run (see `local_cost`'s
    docstring for the call-count arithmetic). Both branches below use a
    vectorized `searchsorted` lookup instead of a per-point Python loop to
    find which piece of `q0`/`q1` is active at each point of the shared
    grid -- the sum over intervals of (width * squared-norm-of-difference)
    itself is unchanged from the straightforward formula. Verified
    numerically equivalent to a plain-loop reference implementation (max
    absolute difference ~1e-16, i.e. floating-point summation-order noise,
    over 300+ random test cases) and end-to-end bit-identical on real
    animation data at several search depths; see `tests/` if present.

    Parameters
    ----------
    q0, q1 : ndarray, shape (n0, k) and (n1, k)
        The two curves in flat vector form (k-dimensional, piecewise
        constant between consecutive sample times).
    I0, I1 : ndarray, shape (n0,) and (n1,)
        The corresponding sample times for `q0`/`q1`.

    Returns
    -------
    float
        The L2 distance between the two piecewise-constant curves.
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
    """Backtrack through the DP `pointers` to recover the optimal path
    from (0, 0) to (M, N).

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
    """
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
    """Yield the candidate predecessor grid points considered when filling
    the DP matrix at (i, j): all (k, l) with `k` in `[max(0, i-depth), i)`
    and `l` in `[max(0, j-depth), j)`.

    Strongly restricted (to `depth` steps back along each axis, rather
    than every earlier grid point) to keep the O(M^2) DP fill
    computationally tractable; this is the `depth` / search-window
    parameter referred to throughout this package.

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
    """Find the optimal time-reparameterization of `q1` onto `q0`: the
    full DP pipeline from two curves to a new set of sample timestamps.

    Builds the shared parameterization, fills the DP cost matrix, backtracks
    to the optimal path, and converts that path into a reparameterization
    function evaluated at the original sample times `I1`.

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
