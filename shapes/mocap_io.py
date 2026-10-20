"""
Loads a parsed animation as a curve in SO(3)^d, with caching to .npy.
"""

import numpy as np
from pathlib import Path
from ._file_parser import parse_asf, parse_amc
from .animation_to_so3 import animation_to_SO3
from .curves import move_origin_to_zero

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "parsed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_curve(animation_id, asf_path, amc_path, crop=None):
    """
    Load one animation as a curve in SO(3)^d.

    Parses and converts the animation on first use, then caches the result to `.npy` under `CACHE_DIR` keyed by `animation_id`; 
    later calls load straight from the cache. Clear the cache if the parsing or conversion code changes, since nothing here checks for that.

    Parameters
    ----------
    animation_id : int or str
        Cache key for this animation.
    asf_path, amc_path : str
        Paths to the subject's skeleton file and this trial's motion file.
    crop : (int, int or None) or None, optional
        Frame range `[start:stop]` to keep, applied after loading/caching.

    Returns
    -------
    ndarray, shape (num_joints, num_frames, 3, 3)
        The curve, cropped if requested and translated to start at the
        identity.
    """
    cache_file = CACHE_DIR / f"{animation_id}.npy"
    if cache_file.exists():
        curve = np.load(cache_file)
    else:
        animation = parse_amc(amc_path)
        skeleton = parse_asf(asf_path)
        curve = animation_to_SO3(skeleton, animation)
        np.save(cache_file, curve)
    if crop:
        start, stop = crop
        curve = curve[:, start:stop]
    return move_origin_to_zero(curve)
