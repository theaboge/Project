"""
Converts a parsed skeleton + animation into a curve in SO(3)^d -- one
rotation matrix per joint per frame. The entry point from raw mocap data
into the math: the discretized version of c in C^inf([0,1], G), G=SO(3)^d
(equation 2 in the theory notes).
"""

import pylab as pl


def Rx(a):
    """Homogeneous (4x4) rotation about x by angle `a`. Internal use only
    (composes with the bone-offset transform `T`); see `transformations.Rx`
    for the plain 3x3 version used elsewhere."""
    s, c = pl.sin(a), pl.cos(a)
    return pl.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def Ry(a):
    """Homogeneous (4x4) rotation about y by angle `a`. See `Rx`."""
    s, c = pl.sin(a), pl.cos(a)
    return pl.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def Rz(a):
    """Homogeneous (4x4) rotation about z by angle `a`. See `Rx`."""
    s, c = pl.sin(a), pl.cos(a)
    return pl.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def T(t):
    """Homogeneous (4x4) translation matrix for vector `t`."""
    M = pl.eye(4)
    M[:3, 3] = t
    return M


def convert_to_SO3(skeleton, frame, filter_noisy_channels=False):
    """Convert one mocap frame into {joint name: 4x4 rotation matrix}.
    Root uses its three Euler angles directly; every other bone composes
    one Rx/Ry/Rz per DOF the .asf skeleton defines (DOF-less bones stay
    identity).

    Note: `node is 'root'` below uses identity rather than equality
    comparison. Harmless in practice -- CPython interns short string
    literals like this, so it behaves like `==` here -- but not guaranteed
    by the language itself."""
    def convert(node):
        if node is 'root':
            transf = [pl.deg2rad(x) for x in frame['root']]
            return pl.dot(Rz(transf[5]), pl.dot(Ry(transf[4]), Rx(transf[3])))

        cbone = skeleton.bones[node]
        M = pl.eye(4)
        try:
            for dof, val in zip(cbone.dof, frame[node]):
                val = pl.deg2rad(val)
                R = pl.eye(4)
                if dof == 'rx':
                    R = Rx(val)
                elif dof == 'ry':
                    R = Ry(val)
                elif dof == 'rz':
                    R = Rz(val)

                M = pl.dot(R, M)
        except Exception:
            pass  # No DOF data for this bone -- leave it as the identity.
        return M

    return {key: convert(key) for key in ['root'] + list(skeleton.bones.keys())}


def animation_to_SO3(skeleton, animation):
    """Convert a full animation into a curve in SO(3)^d, shape
    (d, n_frames, 3, 3), joints ordered as ['root'] + skeleton.bones.keys().

    Bug (present unchanged in the original codebase too): the loop below
    increments its index `i` *before* using it, so every joint's data is
    written one slot too late. Confirmed on real data: channels[0] (meant
    to be root) and the last joint in the list never get written and stay
    at the identity-filled default -- i.e. **root orientation is discarded
    from every curve this function produces.** Not fixed here, since it
    would change every distance and plot in this project; flagged instead
    of silently changed.
    """
    bone_names = ['root'] + list(skeleton.bones.keys())

    frames = []
    for frame in animation.get_frames():
        frames.append(convert_to_SO3(skeleton, frame, True))

    channels = pl.zeros((len(bone_names), len(frames), 3, 3))

    i = 0
    for key in animation.channel_order:
        if not key in bone_names:
            continue
        i += 1
        for j, frame in enumerate(frames):
            channels[i, j, :, :] = frame[key][:3, :3]

    # Fill any never-written (all-zero) joint slot with the identity, so it
    # behaves as a valid element of SO(3) instead of a singular matrix.
    for i in range(channels.shape[0]):
        for j in range(channels.shape[1]):
            if pl.array_equal(channels[i, j, :, :], pl.zeros((3, 3))):
                channels[i, j, :, :] = pl.eye(3)

    return channels
