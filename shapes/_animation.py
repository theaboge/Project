"""
The Animation class: a parsed `.amc` motion as a sequence of frames.
"""


class Animation(object):

    """
    A parsed `.amc` animation: a list of frames, each a dict mapping
    channel name -> float array of values.

    The frames are stored in the `_frames` attribute, and the optional
    description is stored in `_description`.
    """

    def __init__(self, frames, description=""):

        """
        Initialize an Animation with a list of frames and an optional description.

        Parameters
        ----------
        frames : list of dict
            Each frame is a dict mapping channel name -> float array of values.
        description : str, optional
            Stored as the animation's description.
        """

        self._frames = frames
        self._description = description
        self.channel_order = ['ltibia', 'root', 'lfoot', 'rthumb',
                'upperback', 'rfoot', 'head', 'rradius', 'lthumb', 'rfingers',
                'lhand', 'rfemur', 'lfemur', 'lradius',
                'lwrist', 'rtibia', 'lowerneck', 'thorax', 'lclavicle',
                'rclavicle', 'upperneck', 'rtoes', 'lowerback', 'rhumerus',
                'rhand', 'lfingers', 'lhumerus', 'rwrist']

    def get_frames(self):
        """
        Return a list of all frames in the animation.

        Returns
        -------
        list of dict
            Each frame is a dict mapping channel name -> float array of values.
        """
        return self._frames
