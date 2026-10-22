# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Global configuration parameters for astropop."""

from __future__ import annotations

import typing as _t


from functools import wraps as _wraps

_P = _t.ParamSpec('_P')
_R = _t.TypeVar('_R')


def _optional(
    *disable_options: str,
) -> _t.Callable[[_t.Callable[_P, _R]], _t.Callable[_P, _R | None]]:
    """Return None without calling a function when any setting is enabled.

    Settings are attribute names on `AstropopConfig`, evaluated at call time.
    For a property, decorate its getter below ``@property`` so its setter
    remains available. The wrapped function's arguments and metadata are kept.
    """
    def decorate(func: _t.Callable[_P, _R]) -> _t.Callable[_P, _R | None]:
        """Wrap a callable while preserving its arguments and metadata."""
        @_wraps(func)
        def optional(*args: _P.args, **kwargs: _P.kwargs) -> _R | None:
            """Return None for an enabled disable-setting, otherwise call the function."""
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
    FRAMEDATA_DISABLE_UNCERTAINTY
        Ignore FrameData uncertainty inputs and assignments, omit new arrays,
        return None from uncertainty getters even on existing frames, and skip
        uncertainty extraction, propagation and estimation in image reductions.
        Default is False.
    FRAMEDATA_DISABLE_FLAGS
        Ignore FrameData flag and mask inputs and assignments, omit new arrays,
        and return None from flag and mask getters even on existing frames.
        Default is False.

    Examples
    --------
    >>> from astropop.config import AstropopConfig
    >>> from astropop.framedata import FrameData
    >>> previous = AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY
    >>> try:
    ...     # Configure allocation before creating or reading frames.
    ...     AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = True
    ...     frame = FrameData([[100.]], uncertainty=2.)
    ...     print(frame.uncertainty)
    ... finally:
    ...     AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = previous
    None
    """
    FRAMEDATA_DISABLE_UNCERTAINTY: _t.ClassVar[bool] = False
    FRAMEDATA_DISABLE_FLAGS: _t.ClassVar[bool] = False
