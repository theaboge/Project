"""
Computes the SRVT+DL pairwise distances for the dataset (see dataset.py) and
appends the results to a CSV, one row per pair.

Resumable: if `--out` already has rows for some pairs (e.g. a previous run
was interrupted), those pairs are skipped on the next run instead of being
recomputed. The settings are stored in every row, and a run refuses to append
to a file that was made with different settings. (A file from before this was added has no
`settings` column and is accepted as it is.)

The defaults are the original setup (L-BFGS, 500 iterations, L = M = 10, K = 1024).
The tuning section of notebooks/explore.ipynb compares other settings, for example Adam:

    python compute_dl.py --optimizer adam --lr 0.03 --schedule cosine --steps 3000 --out data/results/dl_adam.csv
"""

import csv, time, argparse, sys
from pathlib import Path
from itertools import combinations_with_replacement
import multiprocessing as mp
import torch
from shapes.mocap_io import load_curve
from deeplearning.distance import dl_distance
from deeplearning.tune import config_name
from dataset import ANIMATIONS, paths_for


def already_done(out_path, settings):
    """
    Read which (id1, id2) pairs already have a row in `out_path`, so a
    rerun can skip them instead of recomputing from scratch.

    Parameters
    ----------
    out_path : str or Path
        The CSV file this run is appending to.
    settings : str
        The settings string of this run. If the file was made with other settings, the program stops.

    Returns
    -------
    set of (int, int)
        Pairs already present in the file (empty set if it doesn't exist yet).
    """
    done = set()
    if Path(out_path).exists():
        with open(out_path) as f:
            for row in csv.DictReader(f):
                if "settings" in row and row["settings"] != settings:
                    sys.exit(f"{out_path} was made with different settings ({row['settings']}); use another --out.")
                done.add((int(row["id1"]), int(row["id2"])))
    return done


def init_worker():
    """
    Limit PyTorch to one thread per worker process. Without this, every worker
    uses all cores, and the processes slow each other down.
    """
    torch.set_num_threads(1)


def worker(args):
    """Compute the DL distance for one pair of animations.

    Loads both curves (cached after the first time, see `mocap_io.load_curve`), then computes the SRVT+DL distance (`deeplearning.distance.dl_distance`).
    Diagonal pairs (id1 == id2) are skipped and reported as distance 0, since a curve's distance to itself is trivially zero and not worth computing.
    If the computation fails, the error is returned instead of raised, so one bad pair does not stop the whole run.

    Parameters
    ----------
    args : (int, int, dict)
        (id1, id2, settings), where `settings` are the keyword arguments of `dl_distance` -- bundled into one tuple since this is the
        unit of work handed to each multiprocessing worker.

    Returns
    -------
    (int, int, float or None, float, str or None)
        (id1, id2, dl_distance, seconds_elapsed, error). `dl_distance` is None and `error` is a message if the pair failed.
    """
    id1, id2, settings = args
    asf1, amc1, crop1 = paths_for(id1)
    asf2, amc2, crop2 = paths_for(id2)
    a = load_curve(id1, asf1, amc1, crop1)
    b = load_curve(id2, asf2, amc2, crop2)

    t0 = time.perf_counter()
    try:
        dl = 0.0 if id1 == id2 else dl_distance(a, b, **settings)
    except Exception as e:
        return id1, id2, None, time.perf_counter() - t0, repr(e)
    return id1, id2, dl, time.perf_counter() - t0, None


if __name__ == "__main__":
    # Guarded by __name__ == "__main__" because this script uses
    # multiprocessing.Pool -- without this guard, macOS's "spawn" start
    # method re-imports this file in every worker process, which would
    # re-trigger pool creation recursively and crash.
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs", help="optimization method")
    ap.add_argument("--max-iter", type=int, default=500, help="maximum number of L-BFGS iterations")
    ap.add_argument("--lr", type=float, default=0.01, help="learning rate (Adam)")
    ap.add_argument("--steps", type=int, default=3000, help="number of steps (Adam)")
    ap.add_argument("--schedule", choices=["constant", "cosine"], default="constant", help="how the learning rate changes (Adam)")
    ap.add_argument("--smoothing", type=int, nargs="+", default=[1], help="widths of the moving average on q during the search, coarse to fine (1 = none)")
    ap.add_argument("--layers", type=int, default=10, help="number of layers L in the reparameterization network")
    ap.add_argument("--functions", type=int, default=10, help="number of basis functions M per layer")
    ap.add_argument("--k", type=int, default=1024, help="number of points where the loss is evaluated")
    ap.add_argument("--processes", type=int, default=8, help="number of parallel worker processes")
    ap.add_argument("--out", default="data/results/dl.csv", help="CSV file to append results to")
    args = ap.parse_args()

    settings = dict(layers=args.layers, functions=args.functions, k=args.k, max_iter=args.max_iter, optimizer=args.optimizer,
                    lr=args.lr, steps=args.steps, schedule=args.schedule, smoothing=tuple(args.smoothing))
    tag = config_name(settings)

    ids = sorted(ANIMATIONS)
    done = already_done(args.out, tag)
    # Every unordered pair (i, j) with i <= j, including self-pairs
    # (i == i), minus whatever's already in the output file. The DL distance is not exactly symmetric
    # (the optimizer ends in a slightly different place in the two directions), so only one direction is computed,
    # and the matrix is mirrored when it is read.
    pairs = [(i, j, settings) for i, j in combinations_with_replacement(ids, 2) if (i, j) not in done]

    write_header = not Path(args.out).exists()
    # A file from before the settings column was added keeps its four columns.
    legacy_file = Path(args.out).exists() and "settings" not in Path(args.out).open().readline()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id1", "id2", "dl_distance", "seconds", "settings"])
        with mp.Pool(args.processes, initializer=init_worker) as pool:
            # imap_unordered: results are written as soon as each pair finishes, in whatever order workers complete them in (not
            # submission order) -- and flushed immediately so a crash or interrupt only loses the pair currently in flight, not
            # everything computed so far.
            for id1, id2, dl, secs, error in pool.imap_unordered(worker, pairs):
                if error is not None:
                    print(f"{id1},{id2}  FAILED: {error}")   # not written to the file, so a rerun tries it again
                    continue
                writer.writerow([id1, id2, dl, secs] + ([] if legacy_file else [tag]))
                f.flush()
                print(f"{id1},{id2}  {secs:.1f}s")
