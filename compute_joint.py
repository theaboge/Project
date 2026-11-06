"""
Computes the SRVT-only and SRVT+DP pairwise distances for single joints: the same as compute_dp.py, but on the curve of one joint
(a curve in SO(3)) instead of all 23 joints together (a curve in SO(3)^23).

One CSV per joint is written, `<outdir>/joint_<name>_dp<depth>.csv`, with the same columns as compute_dp.py (one row per pair).
It is resumable in the same way: pairs already in a joint's file are skipped.

    python compute_joint.py --joints root lfemur head
    python compute_joint.py --joints all
"""

import csv, time, argparse
from pathlib import Path
from itertools import combinations_with_replacement
import multiprocessing as mp
from shapes import curves
from shapes.mocap_io import load_curve
from dataset import ANIMATIONS, JOINTS, STATIC_JOINTS, paths_for

DEFAULT_JOINTS = ["root", "lfemur", "ltibia", "lowerback", "lhumerus", "head"]


def already_done(out_path):
    """
    Read which (id1, id2) pairs already have a row in `out_path`, so a rerun can skip them.

    Parameters
    ----------
    out_path : str or Path
        The CSV file of one joint.

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
    """Compute both distances for one pair of animations, on one joint.

    Diagonal pairs (id1 == id2) are skipped and reported as distance 0, as in compute_dp.py.

    Parameters
    ----------
    args : (str, int, int, int)
        (joint, id1, id2, depth) -- bundled into one tuple since this is the unit of work handed to each multiprocessing worker.

    Returns
    -------
    (str, int, int, float, float, float)
        (joint, id1, id2, srvt_distance, dp_distance, seconds_elapsed).
    """
    joint, id1, id2, depth = args
    j = JOINTS.index(joint)
    a = load_curve(id1, *paths_for(id1))[j]
    b = load_curve(id2, *paths_for(id2))[j]

    t0 = time.perf_counter()
    dp = 0.0 if id1 == id2 else curves.dynamic_distance(a, b, depth=depth)
    srvt = 0.0 if id1 == id2 else curves.distance(a, b)
    return joint, id1, id2, srvt, dp, time.perf_counter() - t0


if __name__ == "__main__":
    # Guarded by __name__ == "__main__" for the same reason as in compute_dp.py (macOS "spawn" start method).
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", nargs="+", default=DEFAULT_JOINTS, help="joint names, or 'all' for every joint that moves")
    ap.add_argument("--depth", type=int, default=9, help="DP search-window size (cost/accuracy knob)")
    ap.add_argument("--processes", type=int, default=8, help="number of parallel worker processes")
    ap.add_argument("--outdir", default="data/results", help="folder for the CSV files")
    args = ap.parse_args()

    joints = [j for j in JOINTS if j not in STATIC_JOINTS] if args.joints == ["all"] else args.joints
    unknown = [j for j in joints if j not in JOINTS]
    if unknown:
        raise SystemExit(f"Unknown joint(s) {unknown}; choose from {JOINTS}")

    ids = sorted(ANIMATIONS)
    outs = {j: Path(args.outdir) / f"joint_{j}_dp{args.depth}.csv" for j in joints}
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    tasks = []
    for j in joints:
        done = already_done(outs[j])
        tasks += [(j, a, b, args.depth) for a, b in combinations_with_replacement(ids, 2) if (a, b) not in done]

    files, writers = {}, {}
    for j in joints:
        new = not outs[j].exists()
        files[j] = open(outs[j], "a", newline="")
        writers[j] = csv.writer(files[j])
        if new:
            writers[j].writerow(["id1", "id2", "distance", "dp_distance", "seconds"])
    try:
        with mp.Pool(args.processes) as pool:
            # imap_unordered, flushed immediately, so an interrupt only loses the pairs in flight (see compute_dp.py).
            for joint, id1, id2, srvt, dp, secs in pool.imap_unordered(worker, tasks):
                writers[joint].writerow([id1, id2, srvt, dp, secs])
                files[joint].flush()
                print(f"{joint} {id1},{id2}  {secs:.1f}s")
    finally:
        for f in files.values():
            f.close()
