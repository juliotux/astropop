"""Shared type contracts; these aliases do not perform runtime validation."""

from __future__ import annotations

from os import PathLike as _PathLike
from types import EllipsisType as _EllipsisType
from typing import TYPE_CHECKING as _TYPE_CHECKING, TypeAlias as _TypeAlias
from typing import Annotated as _Annotated, Union as _Union
from collections.abc import Callable as _Callable, Mapping as _Mapping

import numpy as _np
from numpy.typing import ArrayLike as _ArrayLike, NDArray as _NDArray
from astropy import units as _u
from astropy.io import fits as _fits
from astropy.nddata import CCDData as _CCDData

if _TYPE_CHECKING:
    from .framedata import FrameData
    from .math.physical import QFloat

PathLike: _TypeAlias = str | _PathLike[str]
NumberArray: _TypeAlias = float | int | _np.number | _NDArray
UnitLike: _TypeAlias = str | _u.UnitBase | _u.Quantity | None
HeaderLike: _TypeAlias = _fits.Header | _Mapping[str, object]
ImageHDU: _TypeAlias = _fits.PrimaryHDU | _fits.ImageHDU | _fits.CompImageHDU
# Semantic shape metadata; NumPy typing does not enforce dimensionality.
ImageLike: _TypeAlias = _Annotated[_ArrayLike, "2D pixel array (rows, columns)"]
ImageInput: _TypeAlias = _Union[
    ImageLike, 'FrameData', 'QFloat', _u.Quantity, _CCDData,
    ImageHDU, _fits.HDUList, PathLike,
]
QuantityInput: _TypeAlias = _Union[
    _ArrayLike, 'QFloat', 'FrameData', _CCDData, _u.Quantity, _u.UnitBase,
]
Index: _TypeAlias = int | slice | _ArrayLike | _EllipsisType | tuple[object, ...]
Axis: _TypeAlias = int | tuple[int, ...] | None
Window: _TypeAlias = tuple[slice, slice] | None
Statistic: _TypeAlias = str | _Callable[..., _ArrayLike]
