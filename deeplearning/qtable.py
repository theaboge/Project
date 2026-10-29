import torch


class QTable:
    """
    A piecewise linear function of time, built from a table of values q_i at times t_i.

    Makes the SRVT of a recorded motion, which is only known at the frames, usable as
    q or r in `CurveDistance`: it can be evaluated at any time in [0, 1], and is
    differentiable in the time argument, so gradients can flow through phi(x).

    Parameters
    ----------
    q : ndarray, shape (n, d)
        The SRVT values, one row per frame, in vector form (d = 3 * number of joints).
    t : ndarray, shape (n,)
        Increasing sample times in [0, 1], one per row of `q`.
    """

    def __init__(self, q, t):
        self.t = torch.as_tensor(t, dtype=torch.get_default_dtype())
        self.q = torch.as_tensor(q, dtype=torch.get_default_dtype())

    def __call__(self, X, h=None):
        """
        Interpolate q at the times in X.

        Parameters
        ----------
        X : tensor, shape (K, 1)
            Times in [0, 1]. Times outside [t[0], t[-1]] get the value at the nearest end.
        h : ignored
            Only there to match the call signature `CurveDistance` uses for analytic curves.

        Returns
        -------
        tensor, shape (K, d)
            The linearly interpolated values of q.
        """
        x = X.reshape(-1)

        # Index of the interval [t[i], t[i+1]] containing each time. The index is an integer, so it
        # carries no gradient; the gradient comes from the weight w below.
        i = torch.searchsorted(self.t, x.detach(), right=True) - 1
        i = i.clamp(0, len(self.t) - 2)

        # Position inside the interval, 0 at t[i] and 1 at t[i+1].
        w = ((x - self.t[i]) / (self.t[i + 1] - self.t[i])).clamp(0, 1).unsqueeze(1)

        return (1 - w) * self.q[i] + w * self.q[i + 1]