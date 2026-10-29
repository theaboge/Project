"""
Computes the SRVT-only and SRVT+DP pairwise distances for the dataset
(see dataset.py) and appends the results to a CSV, one row per pair.

Resumable: if `--out` already has rows for some pairs (e.g. a previous run
was interrupted), those pairs are skipped on the next run instead of being
recomputed.
"""

import csv, time, argparse
from pathlib import Path
from itertools import combinations_with_replacement
import multiprocessing as mp
from shapes import curves
from shapes.mocap_io import load_curve
from dataset import ANIMATIONS, paths_for


def already_done(out_path):
    """
    Read which (id1, id2) pairs already have a row in `out_path`, so a
    rerun can skip them instead of recomputing from scratch.

    Parameters
    ----------
    out_path : str or Path
        The CSV file this run is appending to.

    Returns
    -------
    set of (int, int)
        Pairs already present in the file (empty set if it doesn't exist yet).
    """
    done = set()
    if Path(out_path).exists():
        with open(out_path) as f:
            for row in csv.DictReader(f):
                done.add((int(row["id1"]), int(row["id2"])))
    return done


def worker(args):
    """Compute both distances for one pair of animations.

    Loads both curves (cached after the first time, see `mocap_io. load_curve`), then computes the SRVT-only distance (`curves.distance`, cheap) 
    and the SRVT+DP distance (`curves.dynamic_distance`, expensive -- this is the part `depth` controls the cost of). 
    Diagonal pairs (id1 == id2) are skipped and reported as distance 0, since a curve's distance to itself is trivially zero and not worth computing.

    Parameters
    ----------
    args : (int, int, int)
        (id1, id2, depth) -- bundled into one tuple since this is the
        unit of work handed to each multiprocessing worker.

    Returns
    -------
    (int, int, float, float, float)
        (id1, id2, srvt_distance, dp_distance, seconds_elapsed).
    """
    id1, id2, depth = args
    asf1, amc1, crop1 = paths_for(id1)
    asf2, amc2, crop2 = paths_for(id2)
    a = load_curve(id1, asf1, amc1, crop1)
    b = load_curve(id2, asf2, amc2, crop2)

    t0 = time.perf_counter()
    dp = 0.0 if id1 == id2 else curves.dynamic_distance(a, b, depth=depth)
    srvt = 0.0 if id1 == id2 else curves.distance(a, b)
    return id1, id2, srvt, dp, time.perf_counter() - t0


if __name__ == "__main__":
    # Guarded by __name__ == "__main__" because this script uses
    # multiprocessing.Pool -- without this guard, macOS's "spawn" start
    # method re-imports this file in every worker process, which would
    # re-trigger pool creation recursively and crash.
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=9, help="DP search-window size (cost/accuracy knob)")
    ap.add_argument("--processes", type=int, default=8, help="number of parallel worker processes")
    ap.add_argument("--out", default="data/results/dp_depth9.csv", help="CSV file to append results to")
    args = ap.parse_args()

    ids = sorted(ANIMATIONS)
    done = already_done(args.out)
    # Every unordered pair (i, j) with i <= j, including self-pairs
    # (i == i), minus whatever's already in the output file.
    pairs = [(i, j, args.depth) for i, j in combinations_with_replacement(ids, 2) if (i, j) not in done]

    write_header = not Path(args.out).exists()
    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id1", "id2", "distance", "dp_distance", "seconds"])
        with mp.Pool(args.processes) as pool:
            # imap_unordered: results are written as soon as each pair finishes, in whatever order workers complete them in (not
            # submission order) -- and flushed immediately so a crash or interrupt only loses the pair currently in flight, not
            # everything computed so far.
            for id1, id2, srvt, dp, secs in pool.imap_unordered(worker, pairs):
                writer.writerow([id1, id2, srvt, dp, secs])
                f.flush()
                print(f"{id1},{id2}  {secs:.1f}s")
