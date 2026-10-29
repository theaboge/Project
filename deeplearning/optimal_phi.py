"""
Deep learning search for the optimal reparameterization of one curve onto another.
The DL counterpart of `shapes.dynamic_distance.find_optimal_diffeomorphism`.
"""

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from .layers import SineSeries
from .networks import CurveReparametrizer
from .loss import CurveDistance
from .qtable import QTable
from .reparametrize import reparametrize
from .logger import Silent


def find_optimal_diffeomorphism_dl(q0, q1, I0, I1, layers=10, functions=10, k=1024, max_iter=500, return_loss=False,
                                   optimizer="lbfgs", lr=1e-2, steps=3000, schedule="constant", smoothing=(1,),
                                   return_history=False):
    """
    Find the optimal reparameterization of one curve onto another, by training a
    residual network phi = phi_L o ... o phi_1 to minimize the discrete
    version of the L2 distance between the SRVTs of the curves.

    The defaults are the setup of the original method: L-BFGS, no smoothing.

    Parameters
    ----------
    q0, q1 : ndarray, shape (n0, d) and (n1, d)
        The two curves in flat vector form. `q0` is held fixed; the returned
        reparameterization applies to `q1`.
    I0, I1 : ndarray, shape (n0,) and (n1,)
        The corresponding sample times for `q0`/`q1`.
    layers : int, optional
        Number of layers L of the network.
    functions : int, optional
        Number of basis functions M in each layer.
    k : int, optional
        Number of evenly spaced points in [0, 1] where the loss is evaluated.
    max_iter : int, optional
        Maximum number of L-BFGS iterations (per smoothing stage).
    return_loss : bool, optional
        If True, also return the value of the loss at the end of the optimization.
    optimizer : {"lbfgs", "adam"}, optional
        Optimization method. Default is "lbfgs".
    lr : float, optional
        Learning rate, the size of the steps (Adam only).
    steps : int, optional
        Total number of steps, shared equally between the smoothing stages (Adam only).
    schedule : {"constant", "cosine"}, optional
        How the learning rate changes during the steps (Adam only). "cosine" decreases it smoothly from `lr` to 1 % of `lr`.
    smoothing : tuple of int, optional
        Widths (in frames) of a moving average applied to `q0` and `q1` during the search, one stage per width, from coarse to fine,
        keeping the network between the stages. (1,) means no smoothing. The final loss and distance always use the unsmoothed curves.
    return_history : bool, optional
        If True, also return the loss at every iteration or step (on the smoothed problem of its stage).

    Returns
    -------
    ndarray, shape (n1,)
        `I1_new`: the new sample times phi(I1), the same output as the DP version.
    float, optional
        The final value of the loss on the unsmoothed curves (only if `return_loss` is True).
    list of float, optional
        The loss history (only if `return_history` is True).
    """
    if optimizer not in ("lbfgs", "adam"):
        raise ValueError(f"optimizer must be 'lbfgs' or 'adam', got {optimizer!r}")
    if schedule not in ("constant", "cosine"):
        raise ValueError(f"schedule must be 'constant' or 'cosine', got {schedule!r}")

    # The tolerances below assume double precision, so it is set here and restored afterwards.
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        network = CurveReparametrizer([SineSeries(functions) for _ in range(layers)])
        history = []

        for width in smoothing:
            # q0 is the fixed target; q1 is evaluated at phi(x), so it must be differentiable in its argument.
            a = uniform_filter1d(q0, size=width, axis=0, mode="nearest") if width > 1 else q0
            b = uniform_filter1d(q1, size=width, axis=0, mode="nearest") if width > 1 else q1
            loss = CurveDistance(QTable(a, I0), QTable(b, I1), k=k, h=None)

            if optimizer == "lbfgs":
                opt = torch.optim.LBFGS(
                    network.parameters(),
                    max_iter=max_iter,
                    max_eval=3 * max_iter,
                    history_size=max_iter,
                    line_search_fn="strong_wolfe",
                    tolerance_grad=1e-10,
                    tolerance_change=1e-10,
                )
                history.extend(float(e) for e in reparametrize(network, loss, opt, None, Silent()))
            else:
                n_steps = max(1, steps // len(smoothing))
                opt = torch.optim.Adam(network.parameters(), lr=lr)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=0.01 * lr) if schedule == "cosine" else None
                for _ in range(n_steps):
                    opt.zero_grad()
                    loss(network).backward()
                    opt.step()
                    if sched is not None:
                        sched.step()
                    network.project()      # keep every layer a diffeomorphism after the step
                    history.append(loss.get_last())

        # The loss of the final network on the real (unsmoothed) problem.
        network.detach()
        final = CurveDistance(QTable(q0, I0), QTable(q1, I1), k=k, h=None)
        with torch.no_grad():
            final_loss = float(final(network))

        # Evaluate phi at the sample times of c_1.
        with torch.no_grad():
            phi = network(torch.as_tensor(I1)[:, None]).squeeze(1).numpy()
    finally:
        torch.set_default_dtype(old_dtype)

    # Rounding can push the end points marginally outside [0, 1], which `reparameterize` rejects.
    phi = np.clip(phi, 0.0, 1.0)

    out = [phi]
    if return_loss:
        out.append(final_loss)
    if return_history:
        out.append(history)
    return out[0] if len(out) == 1 else tuple(out)
