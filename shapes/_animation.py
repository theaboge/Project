"""
The Animation class: holds a parsed `.amc` motion as a sequence of frames
(each frame a dict of channel name -> angle values) plus basic operations
on them (move to origin, crop, step through frames).
"""

from pylab import *


class Animation(object):
    """A parsed motion-capture animation: a list of frames plus a fixed
    channel ordering used when converting frames to SO(3) (see
    `convert.animation_to_SO3`)."""

    def __init__(self, frames, description=""):
        self._frames = frames
        self._description = description
        self._index = 0
        self._offset = zeros(3)
        self.channel_order = ['ltibia', 'root', 'lfoot', 'rthumb',
                'upperback', 'rfoot', 'head', 'rradius', 'lthumb', 'rfingers',
                'lhand', 'rfemur', 'lfemur', 'lradius',
                'lwrist', 'rtibia', 'lowerneck', 'thorax', 'lclavicle',
                'rclavicle', 'upperneck', 'rtoes', 'lowerback', 'rhumerus',
                'rhand', 'lfingers', 'lhumerus', 'rwrist']

    def move_root_to_origin(self):
        """Shift every frame's root translation so the animation starts
        at the origin (translation only, not rotation)."""
        self._offset = -self._frames[0]['root'][0:3]

        for frame in self._frames:
            frame['root'][0:3] += self._offset

    def num_frames(self):
        """Number of frames in the animation."""
        return len(self._frames)

    def crop(self, start, end):
        """Keep only frames `[start, end]` (inclusive)."""
        self._frames = self._frames[start:end + 1]

    def step(self, continuous=False):
        """Advance the playback index by one frame, wrapping to 0 at the
        end. If `continuous`, accumulates a translation offset so looped
        playback doesn't jump back to the start position. Used by the
        (unused in the current pipeline) animation viewer."""
        if not self._frames:
            return
        if self._index >= len(self._frames) - 1:
            self._index = 0
            if continuous:
                self._offset += self._frames[-1]['root'][0:3] - self._frames[0]['root'][0:3]
        else:
            self._index += 1

    def get_current_frame(self):
        """The frame at the current playback index, and the accumulated
        offset. Used by the (unused) animation viewer."""
        if not self._frames or self._index < 0:
            return None

        return self._frames[self._index], self._offset

    def get_frame(self, index):
        """The frame at a specific index."""
        return self._frames[index]

    def get_frames(self):
        """All frames, in order."""
        return self._frames

    def set_frames(self, frames):
        """Replace all frames."""
        self._frames = frames
