# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Global configuration parameters for astropop."""

from functools import wraps as _wraps


def _optional(*disable_options):
    """Return None without calling a function when any setting is enabled.

    Settings are attribute names on `AstropopConfig`, evaluated at call time.
    For a property, decorate its getter below ``@property`` so its setter
    remains available. The wrapped function's arguments and metadata are kept.
    """
    def decorate(func):
        @_wraps(func)
        def optional(*args, **kwargs):
            if any(getattr(AstropopConfig, name) for name in disable_options):
                return None
            return func(*args, **kwargs)
        return optional
    return decorate


class AstropopConfig:
    """Global configuration parameters for astropop.

    Set these class attributes directly before loading or creating frames;
    do not instantiate this class. To avoid allocating optional arrays, the
    settings must be applied before FrameData construction or read_framedata.
    Changing them only before a reduction does not free existing arrays.
    Settings are process-wide and are not thread-local. Disabling storage
    hides arrays on existing frames without deleting them.
    Setters and copies honor the current settings; ignored input arrays are
    not validated or stored. Restore settings after temporary use.

    Attributes
    ----------
    FRAMEDATA_DISABLE_UNCERTAINTY : bool
        Ignore FrameData uncertainty inputs and assignments, omit new arrays,
        return None from uncertainty getters even on existing frames, and skip
        uncertainty extraction, propagation and estimation in image reductions.
        Default is False.
    FRAMEDATA_DISABLE_FLAGS : bool
        Ignore FrameData flag and mask inputs and assignments, omit new arrays,
        and return None from flag and mask getters even on existing frames.
        Default is False.
    """
    FRAMEDATA_DISABLE_UNCERTAINTY = False
    FRAMEDATA_DISABLE_FLAGS = False
