"""
Computes the normalized log-signature distance for all pairs in the
27-animation dataset and writes them to a CSV.

Unlike compute_dp.py, this doesn't need multiprocessing -- signature
computation is cheap and per-animation, not per-pair, so the whole dataset
runs in well under a minute.

    python compute_signatures.py --out data/results/signatures.csv
"""

import csv, argparse
from pathlib import Path
from itertools import combinations_with_replacement
import iisignature
from shapes import log_signature as ls
from shapes.mocap_io import load_curve
from compute_dp import ANIMATIONS, paths_for


def compute_all_signatures(k=3):
    """Compute and cache the log-signature of every animation once.

    Parameters
    ----------
    k : int, optional
        Signature truncation level (default 3).

    Returns
    -------
    dict
        Maps animation id -> its log-signature.
    """
    d = 23 * 3  # joints * 3 (vector form dimension)
    s = iisignature.prepare(d, k)
    signatures = {}
    for aid in sorted(ANIMATIONS):
        asf_path, amc_path, crop = paths_for(aid)
        curve = load_curve(aid, asf_path, amc_path, crop)
        signatures[aid] = ls.curve_log_signature(curve, s)
        print(f"id {aid} ({ANIMATIONS[aid]['file']}) signature computed")
    return signatures


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="signature truncation level")
    ap.add_argument("--out", default="data/results/signatures.csv")
    args = ap.parse_args()

    signatures = compute_all_signatures(k=args.k)

    ids = sorted(ANIMATIONS)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id1", "id2", "signature_distance"])
        for id1, id2 in combinations_with_replacement(ids, 2):
            dist = 0.0 if id1 == id2 else ls.normalized_linear_distance(signatures[id1], signatures[id2])
            writer.writerow([id1, id2, dist])
            print(f"{id1},{id2}  {dist:.4f}")