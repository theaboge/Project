"""
The Skeleton and Bone classes: bone hierarchy, bone lengths/axes, and the
local transform matrices used to turn joint angles into rotations
(`precompute_local_matrices`, used by `convert.animation_to_SO3`).
"""

from collections import defaultdict
from ._animation import *
from .convert import Rx, Ry, Rz, T  # shared with convert.py, see that file -- no longer duplicated here


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
