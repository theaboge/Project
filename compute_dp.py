"""
Computes the SRVT-only and SRVT+DP pairwise distances for the 27-animation
dataset and appends the results to a CSV, one row per pair.

Run as a script (not from a notebook -- uses multiprocessing, which doesn't
play well with notebook kernels):

    python compute_pairwise.py --depth 6 --out data/results/dp_depth6.csv

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

# Where the raw CMU .asf/.amc files live.
SUBJECT_DIR = "/Users/theaboge/Documents/Prosjekt/Signatures in Shape Analysis/Signatures-in-Shape-Analysis/animation/db/subjects"

"""
The 27 animations used throughout this project (thesis Fig. 6.2: subjects 16, 13, 35, 02). `crop` is (start, stop) in frames, trimming standstill
lead-in/lead-out for forward-jump trials and one walk trial; `None` means use the animation as recorded. See the crop derivation notes for how
these specific frame numbers were chosen (root-joint height signal, thresholded at 8% of peak deviation, 10-frame margin).
"""

ANIMATIONS = {
    1:  {"subject": "35", "file": "35_11", "label": "walk",         "crop": None},
    2:  {"subject": "35", "file": "35_12", "label": "walk",         "crop": None},
    3:  {"subject": "35", "file": "35_18", "label": "run/jog",      "crop": None},
    4:  {"subject": "35", "file": "35_26", "label": "run/jog",      "crop": None},
    5:  {"subject": "35", "file": "35_32", "label": "walk",         "crop": None},
    6:  {"subject": "35", "file": "35_22", "label": "run/jog",      "crop": None},
    7:  {"subject": "02", "file": "02_03", "label": "run/jog",      "crop": None},
    8:  {"subject": "16", "file": "16_46", "label": "run/jog",      "crop": None},
    9:  {"subject": "16", "file": "16_47", "label": "walk",         "crop": None},
    10: {"subject": "16", "file": "16_45", "label": "run/jog",      "crop": None},
    11: {"subject": "16", "file": "16_56", "label": "run/jog",      "crop": None},
    12: {"subject": "16", "file": "16_31", "label": "walk",         "crop": (55, None)},
    14: {"subject": "16", "file": "16_36", "label": "run/jog",      "crop": None},
    15: {"subject": "16", "file": "16_22", "label": "walk",         "crop": None},
    16: {"subject": "16", "file": "16_21", "label": "walk",         "crop": None},
    17: {"subject": "16", "file": "16_35", "label": "run/jog",      "crop": None},
    18: {"subject": "16", "file": "16_09", "label": "forward jump", "crop": (182, 448)},
    20: {"subject": "16", "file": "16_05", "label": "forward jump", "crop": (59, 241)},
    21: {"subject": "16", "file": "16_07", "label": "forward jump", "crop": (117, 355)},
    22: {"subject": "16", "file": "16_06", "label": "forward jump", "crop": (160, 346)},
    23: {"subject": "16", "file": "16_16", "label": "walk",         "crop": None},
    24: {"subject": "16", "file": "16_15", "label": "walk",         "crop": None},
    25: {"subject": "16", "file": "16_58", "label": "walk",         "crop": None},
    26: {"subject": "13", "file": "13_13", "label": "forward jump", "crop": (73, 348)},
    27: {"subject": "13", "file": "13_11", "label": "forward jump", "crop": (115, 389)},
    28: {"subject": "13", "file": "13_32", "label": "forward jump", "crop": (57, 302)},
    29: {"subject": "13", "file": "13_19", "label": "forward jump", "crop": (101, 387)},
}


def paths_for(animation_id):
    """Look up the .asf/.amc paths and crop window for an animation id.

    Parameters
    ----------
    animation_id : int
        Key into `ANIMATIONS`.

    Returns
    -------
    asf_path, amc_path : str
        Paths to the subject's skeleton file and this trial's motion file.
    crop : (int, int or None) or None
        The crop window to pass to `mocap_io.load_curve`.
    """
    info = ANIMATIONS[animation_id]
    subj = info["subject"]
    asf_path = f"{SUBJECT_DIR}/{subj}/{subj}.asf"
    amc_path = f"{SUBJECT_DIR}/{subj}/{info['file']}.amc"
    return asf_path, amc_path, info["crop"]


def already_done(out_path):
    """Read which (id1, id2) pairs already have a row in `out_path`, so a
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

    Loads both curves (cached after the first time, see `mocap_io.
    load_curve`), then computes the SRVT-only distance (`curves.distance`,
    cheap) and the SRVT+DP distance (`curves.dynamic_distance`, expensive --
    this is the part `depth` controls the cost of). Diagonal pairs
    (id1 == id2) are skipped and reported as distance 0, since a curve's
    distance to itself is trivially zero and not worth computing.

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
    """
    Guarded by __name__ == "__main__" because this script uses multiprocessing.Pool -- without this guard, macOS's "spawn" start
    method re-imports this file in every worker process, which would re-trigger pool creation recursively and crash.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=6, help="DP search-window size (cost/accuracy knob)")
    ap.add_argument("--processes", type=int, default=8, help="number of parallel worker processes")
    ap.add_argument("--out", default="data/results/dp_depth6.csv", help="CSV file to append results to")
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
