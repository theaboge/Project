"""
Tools for comparing settings of the DL method on a set of pairs, used by the tuning section of notebooks/explore.ipynb.

A setting (a "config") is a dict with some of the keys in DEFAULTS; the rest are filled in from DEFAULTS. The result of every
(config, pair) is appended to a CSV, so a run can be interrupted and resumed, and the configs can be compared afterwards.
The distances of DP and of the original DL setup, which are already computed for all pairs, serve as the reference.
"""

import csv, time
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

from shapes import curves
from shapes.mocap_io import load_curve
from shapes.so3_math import skew_to_vector, SRVT
from dataset import ANIMATIONS, paths_for
from .optimal_phi import find_optimal_diffeomorphism_dl

DEFAULTS = dict(optimizer="lbfgs", lr=0.01, steps=3000, max_iter=500, schedule="constant", smoothing=(1,),
                layers=10, functions=10, k=1024)

COLUMNS = ["config", "optimizer", "lr", "steps", "schedule", "smoothing", "layers", "functions", "k",
           "id1", "id2", "dl_distance", "loss", "n_iter", "seconds"]


def full_config(cfg):
    """Fill in the keys that `cfg` does not set from DEFAULTS."""
    c = dict(DEFAULTS)
    c.update(cfg)
    c["smoothing"] = tuple(c["smoothing"])
    return c


def config_name(cfg):
    """A short, unique name for a config, used as its label in the CSV and in the plots."""
    c = full_config(cfg)
    if c["optimizer"] == "adam":
        name = f"adam lr={c['lr']:g} {c['schedule']} steps={c['steps']}"
    else:
        name = f"lbfgs iter={c['max_iter']}"
    if c["smoothing"] != (1,):
        name += " smooth=" + "-".join(str(w) for w in c["smoothing"])
    return f"{name} L={c['layers']} M={c['functions']} K={c['k']}"


def _init_worker():
    """Limit PyTorch to one thread per worker process, so that the processes do not slow each other down."""
    torch.set_num_threads(1)


def _solve(cfg, id1, id2):
    """
    Run the DL method with the settings `cfg` on one pair, and return the distance on the rotation curves (and the loss and the history).
    """
    c = full_config(cfg)
    c0 = load_curve(id1, *paths_for(id1))
    c1 = load_curve(id2, *paths_for(id2))

    t0 = time.perf_counter()
    I0 = np.linspace(0, 1, c0.shape[1])
    I1 = np.linspace(0, 1, c1.shape[1])
    q0 = skew_to_vector(SRVT(c0, I0))
    q1 = skew_to_vector(SRVT(c1, I1))
    phi, loss, history = find_optimal_diffeomorphism_dl(
        q0, q1, I0, I1, layers=c["layers"], functions=c["functions"], k=c["k"], max_iter=c["max_iter"],
        optimizer=c["optimizer"], lr=c["lr"], steps=c["steps"], schedule=c["schedule"], smoothing=c["smoothing"],
        return_loss=True, return_history=True)
    distance = curves.distance(c0, curves.reparameterize(phi, I1, c1), I0=I0, I1=I1)
    return distance, loss, history, time.perf_counter() - t0


def run_job(job):
    """
    One unit of work for the worker processes: one config on one pair.

    Parameters
    ----------
    job : (dict, int, int)
        (config, id1, id2).

    Returns
    -------
    dict
        One row of the results CSV, or {"config", "id1", "id2", "error"} if the run failed.
    """
    cfg, id1, id2 = job
    c = full_config(cfg)
    try:
        distance, loss, history, seconds = _solve(cfg, id1, id2)
    except Exception as e:
        return {"config": config_name(cfg), "id1": id1, "id2": id2, "error": repr(e)}
    return {"config": config_name(cfg), "optimizer": c["optimizer"], "lr": c["lr"], "steps": c["steps"], "schedule": c["schedule"],
            "smoothing": "-".join(str(w) for w in c["smoothing"]), "layers": c["layers"], "functions": c["functions"], "k": c["k"],
            "id1": id1, "id2": id2, "dl_distance": distance, "loss": loss, "n_iter": len(history), "seconds": seconds}


def run_jobs(jobs, out_csv, processes=8):
    """
    Run a list of (config, id1, id2) jobs in parallel and append the results to a CSV.

    Resumable: jobs whose (config, pair) is already in the file are skipped, and every result is written as soon as it is done.
    A job that fails is reported and not written, so that a rerun tries it again.

    Parameters
    ----------
    jobs : list of (dict, int, int)
        The configs and pairs to run.
    out_csv : str or Path
        The results file.
    processes : int, optional
        Number of worker processes. Default is 8.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.config, prev.id1, prev.id2))
    pending = [j for j in jobs if (config_name(j[0]), j[1], j[2]) not in done]
    print(f"{len(jobs) - len(pending)} of {len(jobs)} runs are already in {out_csv.name}; {len(pending)} to run")
    if not pending:
        return

    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        with mp.Pool(processes, initializer=_init_worker) as pool:
            for n_done, row in enumerate(pool.imap_unordered(run_job, pending), start=1):
                if "error" in row:
                    print(f"FAILED  {row['config']}  {row['id1']},{row['id2']}: {row['error']}")
                    continue
                writer.writerow({c: row[c] for c in COLUMNS})
                f.flush()
                if n_done % 10 == 0 or n_done == len(pending):
                    print(f"{n_done}/{len(pending)} done")


def load_results(out_csv):
    """Read the results CSV (empty table if it does not exist yet)."""
    if not Path(out_csv).exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(out_csv)


def load_reference(dp_csv, dl_csv):
    """
    The distances that every config is compared with: no alignment, DP, and the original DL setup, for every pair of different animations.

    Parameters
    ----------
    dp_csv, dl_csv : str or Path
        The results files of `compute_dp.py` and `compute_dl.py`.

    Returns
    -------
    pandas.DataFrame
        One row per pair, with `srvt`, `dp`, `dl_baseline`, `same_class`, `dp_gain` (how much DP lowers the distance), `gap` (how much
        higher the original DL is than DP) and `share` (the share of DP's improvement that the original DL reaches).
    """
    dp = pd.read_csv(dp_csv)[["id1", "id2", "distance", "dp_distance"]].rename(columns={"distance": "srvt", "dp_distance": "dp"})
    dl = pd.read_csv(dl_csv)[["id1", "id2", "dl_distance"]].rename(columns={"dl_distance": "dl_baseline"})
    ref = dp.merge(dl, on=["id1", "id2"])
    ref = ref[ref.id1 != ref.id2].copy()
    ref["same_class"] = [ANIMATIONS[a]["label"] == ANIMATIONS[b]["label"] for a, b in zip(ref.id1, ref.id2)]
    ref["dp_gain"] = ref.srvt - ref.dp
    ref["gap"] = ref.dl_baseline - ref.dp
    ref["share"] = (ref.srvt - ref.dl_baseline) / ref.dp_gain
    return ref


def choose_pairs(ref, n_hard=8, n_cross=4, n_easy=4):
    """
    Choose a small set of pairs for the tuning that covers the cases that matter.

    - hard pairs in the same class: where the original DL is furthest above DP (the problem we want to fix)
    - hard pairs in different classes: the same, for pairs from different classes
    - easy pairs: where DP gains a lot and the original DL already reaches at least 95 % of it (a setting must not make these worse)

    Parameters
    ----------
    ref : pandas.DataFrame
        The output of `load_reference`.
    n_hard, n_cross, n_easy : int, optional
        How many pairs of each kind.

    Returns
    -------
    pandas.DataFrame
        The chosen pairs, with their kind, file names and reference distances.
    """
    hard = ref[ref.same_class].nlargest(n_hard, "gap").assign(kind="hard, same class")
    cross = ref[~ref.same_class].nlargest(n_cross, "gap").assign(kind="hard, different classes")
    easy = ref[(ref.share >= 0.95) & (ref.dp_gain > 0.1 * ref.srvt)].nlargest(n_easy, "dp_gain").assign(kind="easy")
    pairs = pd.concat([hard, cross, easy]).drop_duplicates(["id1", "id2"]).reset_index(drop=True)
    pairs["file1"] = [ANIMATIONS[a]["file"] for a in pairs.id1]
    pairs["file2"] = [ANIMATIONS[a]["file"] for a in pairs.id2]
    return pairs


def merge_with_reference(results, ref):
    """Add the reference distances and the derived `gap` and `share` to the results of the configs, one row per (config, pair)."""
    m = results.merge(ref[["id1", "id2", "srvt", "dp", "same_class"]], on=["id1", "id2"])
    m["gap"] = m.dl_distance - m.dp
    m["share"] = (m.srvt - m.dl_distance) / (m.srvt - m.dp)
    return m


def summarise(results, ref):
    """
    One row per config: how close it gets to DP on the chosen pairs, and what it costs.

    Parameters
    ----------
    results : pandas.DataFrame
        The results of the configs to compare (from `load_results`).
    ref : pandas.DataFrame
        The output of `load_reference`.

    Returns
    -------
    pandas.DataFrame
        `mean share` and `worst share` are the share of DP's improvement that the config reaches (1 means as good as DP, 0 no better than no alignment),
        over the pairs where DP improves the distance by more than 5 %. `mean gap` is the mean distance above DP. Sorted by `mean share`.
    """
    m = merge_with_reference(results, ref)
    rows = []
    for name, g in m.groupby("config"):
        meaningful = g[(g.srvt - g.dp) > 0.05 * g.srvt]
        rows.append({"config": name, "pairs": len(g), "mean share": meaningful.share.mean(), "median share": meaningful.share.median(),
                     "worst share": meaningful.share.min(), "mean gap": g.gap.mean(), "pairs within 0.5 of DP": (g.gap < 0.5).mean(),
                     "mean seconds": g.seconds.mean()})
    return pd.DataFrame(rows).set_index("config").sort_values("mean share", ascending=False)


def history_for(cfg, id1, id2):
    """Run one config on one pair and return its loss history and final distance, for plotting loss curves."""
    distance, loss, history, seconds = _solve(cfg, id1, id2)
    return history, distance
