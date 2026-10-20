"""
The 29-animation dataset used throughout this project (subjects 16, 13,
35, 02): where the raw files live, which crop window each trial needs,
and how to resolve an animation id to its file paths.
"""

# Where the raw CMU .asf/.amc files live.
SUBJECT_DIR = "/Users/theaboge/Documents/Prosjekt/motion-shape-analysis/data/subjects"

# `crop` is (start, stop) in frames, trimming standstill lead-in/lead-out for forward-jump trials and some walk trials; 
# `None` means use the
# animation as recorded. Crop windows are derived from the root-joint
# height signal (for jump trials) or forward-velocity (for walk trials with
# a static lead-in), thresholded at 8% of peak deviation with a 10-frame
# margin.
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
    13: {"subject": "16", "file": "16_32", "label": "walk",         "crop": (129, None)},
    14: {"subject": "16", "file": "16_36", "label": "run/jog",      "crop": None},
    15: {"subject": "16", "file": "16_22", "label": "walk",         "crop": None},
    16: {"subject": "16", "file": "16_21", "label": "walk",         "crop": None},
    17: {"subject": "16", "file": "16_35", "label": "run/jog",      "crop": None},
    18: {"subject": "16", "file": "16_09", "label": "forward jump", "crop": (182, 448)},
    19: {"subject": "16", "file": "16_10", "label": "forward jump", "crop": (195, 453)},
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
