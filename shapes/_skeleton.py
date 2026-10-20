"""
The Skeleton and Bone classes: a bone hierarchy keyed by name, each bone
carrying only what `animation_to_so3.animation_to_SO3` actually reads from it.
"""


class Bone(object):
    """
    A parsed `.asf` bone: only the name and DOF are kept, since that's all `animation_to_so3.animation_to_SO3` actually reads from it.
    """

    def __init__(self):
        self.name = ''
        self.dof = []


class Skeleton(object):
    """
    A parsed `.asf` skeleton: a dict of `Bone`s keyed by name.
    """

    def __init__(self, description=""):
        self._description = description
        self.bones = dict()
