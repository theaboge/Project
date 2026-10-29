"""
Computes the normalized log-signature distance for all pairs in the
29-animation dataset and writes them to a CSV.

Unlike compute_dp.py, this doesn't need multiprocessing -- signature
computation is cheap and per-animation, not per-pair, so the whole dataset
runs in well under a minute.

The CSV has the same layout as the other methods' files (id1, id2, the distance, seconds). The time is measured so the
cost can be compared with the other methods: since a signature is computed once per animation and not once per pair,
the `seconds` of a pair is the time of its distance plus its share of the two signatures (each animation is in n - 1
pairs, so each signature is split evenly over those). Summing the `seconds` column therefore gives the total time of the run,
as it does for DP and DL. Loading the curves is not counted, in the same way as in compute_dp.py and compute_dl.py.

    python compute_signatures.py --out data/results/signatures.csv
"""

import csv, argparse, time
from pathlib import Path
from itertools import combinations_with_replacement
import iisignature
from shapes import log_signature_distance as ls
from shapes.mocap_io import load_curve
from dataset import ANIMATIONS, paths_for


def compute_all_signatures(k=3):
    """
    Compute and cache the log-signature of every animation once.

    Parameters
    ----------
    k : int, optional
        Signature truncation level (default 3).

    Returns
    -------
    signatures : dict
        Maps animation id -> its log-signature.
    seconds : dict
        Maps animation id -> the time in seconds to compute its signature (loading the curve is not counted).
    setup_seconds : float
        The time in seconds to prepare `iisignature` for this truncation level (done once).
    """
    d = 23 * 3  # joints * 3 (vector form dimension)
    t0 = time.perf_counter()
    s = iisignature.prepare(d, k)
    setup_seconds = time.perf_counter() - t0

    signatures = {}
    seconds = {}
    for aid in sorted(ANIMATIONS):
        asf_path, amc_path, crop = paths_for(aid)
        curve = load_curve(aid, asf_path, amc_path, crop)

        t0 = time.perf_counter()
        signatures[aid] = ls.curve_log_signature(curve, s)
        seconds[aid] = time.perf_counter() - t0
        print(f"id {aid} ({ANIMATIONS[aid]['file']}) signature computed in {seconds[aid]:.3f}s")
    return signatures, seconds, setup_seconds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="signature truncation level")
    ap.add_argument("--out", default="data/results/signatures.csv")
    args = ap.parse_args()

    signatures, signature_seconds, setup_seconds = compute_all_signatures(k=args.k)

    ids = sorted(ANIMATIONS)
    n_animations = len(ids)
    n_pairs = n_animations * (n_animations - 1) // 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    distance_seconds = 0.0
    written_seconds = 0.0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id1", "id2", "signature_distance", "seconds"])
        for id1, id2 in combinations_with_replacement(ids, 2):
            t0 = time.perf_counter()
            dist = 0.0 if id1 == id2 else ls.normalized_linear_distance(signatures[id1], signatures[id2])
            secs = time.perf_counter() - t0
            distance_seconds += secs
            if id1 != id2:
                # Add this pair's share of the signatures (and of the one-time setup), so that the column sums to the total time.
                secs += (signature_seconds[id1] + signature_seconds[id2]) / (n_animations - 1) + setup_seconds / n_pairs
            written_seconds += secs
            writer.writerow([id1, id2, dist, secs])
            print(f"{id1},{id2}  {dist:.4f}")

    total = setup_seconds + sum(signature_seconds.values()) + distance_seconds
    print(f"\nTime (loading the curves not counted):")
    print(f"  setup of iisignature          {setup_seconds:8.3f} s")
    print(f"  {n_animations} signatures            {sum(signature_seconds.values()):8.3f} s  (mean {sum(signature_seconds.values()) / n_animations:.3f} s each)")
    print(f"  {n_pairs} pair distances        {distance_seconds:8.3f} s")
    print(f"  total                         {total:8.3f} s")
    print(f"  sum of the `seconds` column   {written_seconds:8.3f} s  (equals the total)")