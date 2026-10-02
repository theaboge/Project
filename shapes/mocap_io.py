import numpy as np
from pathlib import Path
from ._dataparse import parse_asf, parse_amc
from .convert import animation_to_SO3
from .curves import move_origin_to_zero

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "parsed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_curve(animation_id, asf_path, amc_path, crop=None):
    """Last en animasjon som kurve i SO(3)^d, med cache til .npy."""
    cache_file = CACHE_DIR / f"{animation_id}.npy"
    if cache_file.exists():
        curve = np.load(cache_file)
    else:
        animation = parse_amc(amc_path)
        animation.move_root_to_origin()
        skeleton = parse_asf(asf_path)
        skeleton.precompute_local_matrices()
        curve = animation_to_SO3(skeleton, animation)
        np.save(cache_file, curve)
    if crop:
        start, stop = crop
        curve = curve[:, start:stop]
    return move_origin_to_zero(curve)