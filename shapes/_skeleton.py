"""
The Skeleton and Bone classes: bone hierarchy, bone lengths/axes, and the
local transform matrices used to turn joint angles into rotations
(`precompute_local_matrices`, used by `convert.animation_to_SO3`) or into
3D coordinates for a given frame (`get_coords_for_frame`,
`get_lines_for_frame` -- used only for 3D visualization, not by the
distance/signature pipeline).
"""

import pylab as pl
from collections import defaultdict
from ._animation import *


def Rx(a):
    """Homogeneous (4x4) rotation about x by angle `a`."""
    s, c = pl.sin(a), pl.cos(a)
    return pl.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def Ry(a):
    """Homogeneous (4x4) rotation about y by angle `a`."""
    s, c = pl.sin(a), pl.cos(a)
    return pl.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def Rz(a):
    """Homogeneous (4x4) rotation about z by angle `a`."""
    s, c = pl.sin(a), pl.cos(a)
    return pl.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def T(t):
    """Homogeneous (4x4) translation matrix for vector `t`."""
    M = eye(4)
    M[:3, 3] = t
    return M


class Bone(object):
    """One bone from a parsed `.asf` skeleton: name, rest direction,
    length, rotation axis, and its degrees of freedom (`dof`, e.g.
    `['rx','ry']`) with their limits."""

    def __init__(self):
        self.name = ''
        self.direction = zeros(3)
        self.length = 0.
        self.axis = zeros(3)
        self.dof = []
        self.limits = []


class Skeleton(object):
    """A parsed `.asf` skeleton: a hierarchy of `Bone`s plus the local
    transform matrices needed to pose them."""

    SCALE = 0.05

    def __init__(self, description=""):
        self._description = description
        self.bones = dict()
        self.__local_matrices = None
        self.hierarchy = defaultdict(set)
        self.root = 'root'
        self._lines = None
        self._coords = None

    def precompute_local_matrices(self):
        """Precompute each bone's local axis-rotation matrices (C, its
        inverse) and offset transform (B), used by `convert.
        animation_to_SO3` to turn per-frame joint angles into SO(3)
        rotations. Must be called once before converting an animation."""
        self.__local_matrices = dict()

        for key, bone in self.bones.items():
            C = dot(Rz(bone.axis[2]), dot(Ry(bone.axis[1]), Rx(bone.axis[0])))
            Cinv = C.transpose()
            B = T(bone.direction * bone.length)

            self.__local_matrices[key + '__C'] = C
            self.__local_matrices[key + '__Cinv'] = Cinv
            self.__local_matrices[key + '__B'] = B

    # -- Visualization only below: 3D coordinates/lines for drawing the
    # -- skeleton. Not used by the distance/signature pipeline.

    def setup_skeleton_neutral_coords(self):
        """Compute each bone's 3D coordinate in the skeleton's neutral
        (rest) pose. Visualization only."""
        coords = [zeros(3)]

        def get_child_coords(bone, pos):
            for child in self.hierarchy[bone]:
                cbone = self.bones[child]
                cpos = pos + cbone.direction * cbone.length
                coords.append(cpos)
                get_child_coords(child, cpos)

        get_child_coords('root', zeros(3))
        self._coords = coords

    def get_skeleton_neutral_coords(self):
        """Neutral-pose 3D coordinates, computing them on first call.
        Visualization only."""
        if self._coords is None:
            self.setup_skeleton_neutral_coords()

        return pl.vstack(self._coords).T

    def setup_skeleton_neutral_lines(self):
        """Compute the bone-to-bone line segments (parent/child index
        pairs) for drawing the skeleton. Visualization only."""
        lines = []

        def traverse_hierarchy(bone, bcount):
            ncount = bcount
            for child in self.hierarchy[bone]:
                lines.append([bcount, ncount + 1])
                ncount = traverse_hierarchy(child, ncount + 1)

            return ncount

        traverse_hierarchy('root', 0)
        self._lines = lines

    def get_skeleton_neutral_lines(self):
        """Bone line segments, computing them on first call.
        Visualization only."""
        if self._lines is None:
            self.setup_skeleton_neutral_lines()

        return self._lines

    def get_coords_for_frame(self, frame, root_offset=None):
        """3D coordinates of every joint for one posed frame.
        Visualization only -- not used by the distance pipeline, which
        works with rotation matrices directly (see `convert.animation_to_SO3`)."""
        coords = []

        def draw_line_to_children(bone, ppos, P, rpos, ppindex):
            for child in self.hierarchy[bone]:
                cbone = self.bones[child]
                C = self.__local_matrices[child + '__C']
                Cinv = self.__local_matrices[child + '__Cinv']
                B = self.__local_matrices[child + '__B']

                M = eye(4)
                try:
                    for dof, val in zip(cbone.dof, frame[child]):
                        val = pl.deg2rad(val)
                        R = eye(4)
                        if dof == 'rx':
                            R = Rx(val)
                        elif dof == 'ry':
                            R = Ry(val)
                        elif dof == 'rz':
                            R = Rz(val)

                        M = dot(R, M)
                except Exception:
                    pass  # No DOF data for this bone.

                L = C.dot(M).dot(Cinv).dot(B)
                A = dot(P, L)
                cpos = dot(A, [0, 0, 0, 1]) + rpos

                coords.append(cpos[:3])
                draw_line_to_children(child, cpos, A, rpos, len(coords) - 1)

        transf = frame['root'].copy()
        if root_offset is not None:
            transf[0:3] += root_offset

        R = dot(Rz(pl.deg2rad(transf[5])), dot(Ry(pl.deg2rad(transf[4])), Rx(pl.deg2rad(transf[3]))))
        B = T(transf[0:3])
        rpos = dot(B, [0, 0, 0, 1]) / 0.45 * Skeleton.SCALE
        coords.append(rpos[:3])

        draw_line_to_children('root', rpos, R, rpos, 0)

        return pl.vstack(coords).T

    def get_lines_for_frame(self, frame, root_offset=None):
        """Posed 3D line segments for one frame, for drawing. Skips toe
        bones. Visualization only."""
        lines = []

        def draw_line_to_children(bone, ppos, P, rpos):
            for child in self.hierarchy[bone]:
                cbone = self.bones[child]

                C = self.__local_matrices[child + '__C']
                Cinv = self.__local_matrices[child + '__Cinv']
                B = self.__local_matrices[child + '__B']

                M = eye(4)
                try:
                    for dof, val in zip(cbone.dof, frame[child]):
                        val = pl.deg2rad(val)
                        R = eye(4)
                        if dof == 'rx':
                            R = Rx(val)
                        elif dof == 'ry':
                            R = Ry(val)
                        elif dof == 'rz':
                            R = Rz(val)

                        M = dot(R, M)
                except Exception:
                    pass  # No DOF data for this bone.

                L = C.dot(M).dot(Cinv).dot(B)
                A = dot(P, L)
                cpos = dot(A, [0, 0, 0, 1]) + rpos

                if child[0] == 'rtoes' or child == 'ltoes':
                    continue

                lines.append([ppos[:3], cpos[:3]])
                draw_line_to_children(child, cpos, A, rpos)

        transf = frame['root'].copy()
        if root_offset is not None:
            transf[0:3] += root_offset

        R = dot(Rz(pl.deg2rad(transf[5])), dot(Ry(pl.deg2rad(transf[4])), Rx(pl.deg2rad(transf[3]))))
        B = T(transf[0:3])
        rpos = dot(B, [0, 0, 0, 1]) / 0.45 * Skeleton.SCALE
        draw_line_to_children('root', rpos, R, rpos)

        return lines
