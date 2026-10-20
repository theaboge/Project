"""
Converts a parsed skeleton + animation into a curve in SO(3)^d -- one rotation matrix per joint per frame.
"""

import numpy as np


def Rx(a):
    """Homogeneous (4x4) rotation about x by angle `a`, in radians.

    Parameters
    ----------
    a : float

    Returns
    -------
    ndarray of shape (4, 4)
    """
    s, c = np.sin(a), np.cos(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def Ry(a):
    """Homogeneous (4x4) rotation about y by angle `a`, in radians. See `Rx`."""
    s, c = np.sin(a), np.cos(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def Rz(a):
    """Homogeneous (4x4) rotation about z by angle `a`, in radians. See `Rx`."""
    s, c = np.sin(a), np.cos(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def convert_to_SO3(skeleton, frame):
    """
    Convert one mocap frame into {joint name: 4x4 rotation matrix}.

    Root's rotation comes directly from its three Euler angles. Every
    other bone composes one Rx/Ry/Rz per degree of freedom the `.asf`
    skeleton defines for it, in that bone's own `dof` order; a bone with
    no degrees of freedom stays the identity.

    Parameters
    ----------
    skeleton : Skeleton
        The parsed skeleton corresponding to `frame`.
    frame : dict
        One frame's channel data: joint name -> angle values (degrees).

    Returns
    -------
    dict
        Joint name -> 4x4 rotation matrix.
    """
    def convert(node):
        if node == 'root':
            transf = [np.deg2rad(x) for x in frame['root']]
            return np.dot(Rz(transf[5]), np.dot(Ry(transf[4]), Rx(transf[3])))

        cbone = skeleton.bones[node]
        M = np.eye(4)
        try:
            for dof, val in zip(cbone.dof, frame[node]):
                val = np.deg2rad(val)
                R = np.eye(4)
                if dof == 'rx':
                    R = Rx(val)
                elif dof == 'ry':
                    R = Ry(val)
                elif dof == 'rz':
                    R = Rz(val)

                M = np.dot(R, M)
        except Exception:
            pass  # No DOF data for this bone -- leave it as the identity.
        return M

    return {key: convert(key) for key in ['root'] + list(skeleton.bones.keys())}


def animation_to_SO3(skeleton, animation):
    """
    Convert a full animation into a curve in SO(3)^d.

    Parameters
    ----------
    skeleton : Skeleton
    animation : Animation

    Returns
    -------
    ndarray of shape (num_joints, num_frames, 3, 3)
        One rotation matrix per joint per frame, joints ordered as
        `['root'] + skeleton.bones.keys()`.
    """
    bone_names = ['root'] + list(skeleton.bones.keys())

    frames = []
    for frame in animation.get_frames():
        frames.append(convert_to_SO3(skeleton, frame))

    channels = np.zeros((len(bone_names), len(frames), 3, 3))

    for key in animation.channel_order:
        if key not in bone_names:
            continue
        i = bone_names.index(key)
        for j, frame in enumerate(frames):
            channels[i, j, :, :] = frame[key][:3, :3]

    # Fill any never-written (all-zero) joint slot with the identity, so it behaves as a valid element of SO(3) instead of a singular matrix.
    for i in range(channels.shape[0]):
        for j in range(channels.shape[1]):
            if np.array_equal(channels[i, j, :, :], np.zeros((3, 3))):
                channels[i, j, :, :] = np.eye(3)

    return channels
