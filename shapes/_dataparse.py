"""
Parses raw CMU `.amc` (motion) and `.asf` (skeleton) files into `Animation`
and `Skeleton` objects. The mocap file-format reader.
"""

from ._animation import Animation
from ._skeleton import Skeleton, Bone

import pylab as pl


def parse_array(lst):
    """Convert a list of string tokens to a float ndarray."""
    return pl.array([float(x) for x in lst])


def parse_amc(filename, description="", remove_noisy_channels=True):
    """Parse a `.amc` motion file into an `Animation`.

    Each frame becomes a dict mapping channel name -> float array of
    values. By default, low-DOF extremity channels (toes/thumbs/fingers/
    hands) are dropped as noisy.

    Parameters
    ----------
    filename : str
        Path to the `.amc` file.
    description : str, optional
        Stored as the animation's description.
    remove_noisy_channels : bool, optional
        If True (default), drop toe/thumb/finger/hand channels.

    Returns
    -------
    Animation
        The parsed animation.
    """
    data = []
    with open(filename, 'r') as asf:
        data = asf.readlines()

    data = [line.strip() for line in data]  # Remove leading and trailing whitespace to simplify matters

    frames = []
    frame = None
    noisy_channels = {'rtoes', 'ltoes', 'rthumb', 'lthumb', 'rfingers',
                       'lfingers', 'rhand', 'lhand'}

    for i, line in enumerate(data):
        if line[0] in {':', '#'}:
            continue  # Skip to frames (header/comment lines)

        if line.isdigit():
            # Found a new frame
            if frame:
                frames.append(frame)
            frame = dict()
            continue

        tokens = line.split()

        if remove_noisy_channels and tokens[0] in noisy_channels:
            continue
        frame[tokens[0]] = parse_array(tokens[1:])

    return Animation(frames, description)


def parse_asf(filename, description="", remove_noisy_channels=True):
    """Parse a `.asf` skeleton file into a `Skeleton`.

    Reads the `:bonedata` section into `Bone` objects (name, direction,
    length, axis, degrees of freedom and their limits), then the
    `:hierarchy` section into `skeleton.hierarchy` (parent -> children).
    By default, the same noisy extremity bones `parse_amc` drops are
    excluded here too, for consistency.

    Parameters
    ----------
    filename : str
        Path to the `.asf` file.
    description : str, optional
        Stored as the skeleton's description.
    remove_noisy_channels : bool, optional
        If True (default), drop toe/thumb/finger/hand bones.

    Returns
    -------
    Skeleton or None
        The parsed skeleton, or None (with a printed message) if the file
        is missing a `:bonedata` or `:hierarchy` section.
    """
    noisy_channels = {'rtoes', 'ltoes', 'rthumb', 'lthumb', 'rfingers',
                       'lfingers', 'rhand', 'lhand'}
    data = []
    with open(filename, 'r') as asf:
        data = asf.readlines()

    data = [line.strip() for line in data]  # Remove leading and trailing whitespace to simplify matters

    # Skip to bonedata
    bonedata = -1
    for i, line in enumerate(data):
        if line == ':bonedata':
            bonedata = i
            break

    if bonedata < 0:
        print('No bonedata found!')
        return

    bones = dict()
    bone = None
    end_bones = -1
    in_limits = False

    for i, line in enumerate(data[bonedata + 1:]):
        tokens = line.split()

        if tokens[0][0] == ':':
            end_bones = i  # End of bonedata structure
            break

        if tokens[0] == 'begin':
            bone = Bone()

        if tokens[0] == 'end':
            if remove_noisy_channels and bone.name in noisy_channels:
                continue
            bones[bone.name] = bone

        if tokens[0] == 'name':
            bone.name = tokens[1]

        if tokens[0] == 'direction':
            bone.direction = parse_array(tokens[1:4])

        if tokens[0] == 'length':
            bone.length = float(tokens[1]) / 0.45 * Skeleton.SCALE

        if tokens[0] == 'axis':
            bone.axis = pl.deg2rad(parse_array(tokens[1:4]))

        if tokens[0] == 'dof':
            bone.dof = tokens[1:]

        # Axis limits -- assumes "dof" was already set.
        if tokens[0] == 'limits':
            in_limits = True
            bone.limits.append((float(tokens[1][1:]), float(tokens[2][:-1])))
        elif in_limits:
            bone.limits.append((float(tokens[0][1:]), float(tokens[1][:-1])))
        if len(bone.limits) == len(bone.dof):
            in_limits = False

    # Skip to hierarchy
    hierarchy_index = -1
    for i, line in enumerate(data[bonedata + end_bones:]):
        if line == ':hierarchy':
            hierarchy_index = i
            break

    if hierarchy_index < 0:
        print('No hierarchy found!')
        return

    hierarchy_index += bonedata + end_bones

    skeleton = Skeleton(description)
    skeleton.bones = bones
    for line in data[hierarchy_index + 1:]:
        tokens = line.split()

        if tokens[0] == 'begin':
            continue
        if tokens[0] == 'end':
            break

        skeleton.hierarchy[tokens[0]] = [c for c in tokens[1:] if c not in noisy_channels]

    return skeleton
