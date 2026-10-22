# Licensed under a 3-clause BSD style license - see LICENSE.rst


"""Calibrate images and apply pixel-level corrections to FrameData."""
from __future__ import annotations

import typing as _t
from .._typing import ImageInput as _ImageInput, QuantityInput as _QuantityInput
from ..framedata import FrameData
import os
import numpy as np
import astroscrappy

from ..logger import logger
from .imarith import imarith
from ..math.physical import convert_to_qfloat
from ..framedata import check_framedata, PixelMaskFlags


__all__ = ['cosmics_lacosmic', 'gain_correct', 'subtract_bias',
           'subtract_dark', 'flat_correct', 'trim_image']


# TODO: replace ccdproc functions by built-in, skiping units
# block_reduce = ccdproc.block_reduce
# block_replicate = ccdproc.block_replicate


def cosmics_lacosmic(
    frame: _ImageInput,
    inplace: bool=False,
    **lacosmic_kwargs: _t.Any,
) -> FrameData:
    """
    Remove cosmic rays with LAcosmic. From astroscrappy package.

    Notes
    -----
    * Detailed information about the Laplacian Cosmic Ray Identification method
      can be found on Dokkum,P.G. (2001) - PASP 113, 1420 (2001)
      https://arxiv.org/pdf/astro-ph/0108003.pdf.

    * If ``frame`` is not a `~astropop.framedata.FrameData` instance,
      inplies in ``inplace=False``, and a new `~astropop.framedata.FrameData`
      instance will be created.

    Parameters
    ----------
    frame
        2D image to clean with LACosmic.
    inplace
        If True, the operations will be performed inplace in the ``frame``.
    **lacosmic_kwargs
        Options forwarded to astroscrappy.detect_cosmics, including gain,
        readnoise, rejection thresholds and iteration count. Existing FrameData
        masks are not forwarded automatically; supply inmask explicitly if needed.

    Returns
    -------
    result
        New cosmic-rays corrected `~astropop.framedata.FrameData` instance if not ``inplace``,
        else the ``image`` `~astropop.framedata.FrameData` instance.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.framedata import FrameData, PixelMaskFlags as F
    >>> from astropop.image.processing import cosmics_lacosmic
    >>> pixels = np.full((32, 32), 100.)
    >>> pixels[16, 16] = 10000.
    >>> frame = FrameData(pixels, unit='adu')
    >>> cleaned = cosmics_lacosmic(frame, gain=1., readnoise=4., niter=1)
    >>> print(bool(cleaned.mask_flags(F.COSMIC_RAY)[16, 16]))
    True
    """
    frame = check_framedata(frame)

    mask, dat = astroscrappy.detect_cosmics(frame.data, **lacosmic_kwargs)

    if inplace:
        ccd = frame
    else:
        ccd = frame.copy()

    ccd.data = dat
    ccd.add_flags(PixelMaskFlags.INTERPOLATED | PixelMaskFlags.COSMIC_RAY,
                  where=mask)
    ccd.header['HIERARCH astropop lacosmic'] = True
    return ccd


def gain_correct(
    image: _ImageInput,
    gain: _QuantityInput,
    inplace: bool=False,
) -> FrameData:
    """
    Process the gain correction of an image.

    Notes
    -----
    * The gain is implemented as a multiplier of the original image.

    * If ``image`` is not a `~astropop.framedata.FrameData` instance,
      inplies in ``inplace=False``, and a new `~astropop.framedata.FrameData`
      instance will be created.

    Parameters
    ----------
    image
        Values to perform the operation. `~astropy.units.Quantity`, numerical
        values and `~astropy.nddata.CCDData` are also suported.
    gain
        Gain to be applied on the image numerical values.
    inplace
        If True, the operations will be performed inplace in the ``image``.

    Returns
    -------
    result
        New gain corrected `~astropop.framedata.FrameData` instance if not ``inplace``, else the
        ``image`` `~astropop.framedata.FrameData` instance.

    Examples
    --------
    >>> from astropy import units as u
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.processing import gain_correct
    >>> frame = FrameData([[100.]], unit='adu', uncertainty=3.)
    >>> electrons = gain_correct(frame, 2 * u.electron / u.adu)
    >>> print(electrons.data.tolist(), electrons.unit)
    [[200.0]] electron
    >>> print(electrons.uncertainty.tolist())
    [[6.0]]
    """
    gain = convert_to_qfloat(gain)
    nim = imarith(image, gain, '*', inplace=inplace, merge_headers='first',
                  merge_flags='no_merge')
    nim.header['HIERARCH astropop gain_corrected'] = True
    nim.header['HIERARCH astropop gain_corrected_value'] = gain.nominal
    nim.header['HIERARCH astropop gain_corrected_unit'] = str(gain.unit)

    return nim


def subtract_bias(
    image: _ImageInput,
    master_bias: _ImageInput,
    inplace: bool=False,
) -> FrameData:
    """
    Subtract a master_bias frame from a FrameData.

    Notes
    -----
    * This function will just subtract a master bias image from the original
      image. The master bias calculation, by handling several bias images,
      must be done previously.

    * If ``image`` is not a `~astropop.framedata.FrameData` instance,
      inplies in ``inplace=False``, and a new `~astropop.framedata.FrameData`
      instance will be created.

    Parameters
    ----------
    image
        Image to perform the bias correction. `~astropy.units.Quantity`,
        numerical values and `~astropy.nddata.CCDData` are also suported.
    master_bias
        Master bias image to be subtracted from the ``image``.
    inplace
        If True, the operations will be performed inplace in the ``image``.

    Returns
    -------
    result
        New bias corrrected `~astropop.framedata.FrameData` instance if ``inplace``, else the
        ``image`` `~astropop.framedata.FrameData` instance.

    Examples
    --------
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.processing import subtract_bias
    >>> science = FrameData([[110.]], unit='adu', uncertainty=3.)
    >>> bias = FrameData([[10.]], unit='adu', uncertainty=4.)
    >>> corrected = subtract_bias(science, bias)
    >>> print(corrected.data.tolist(), corrected.uncertainty.tolist())
    [[100.0]] [[5.0]]
    """
    master_bias = check_framedata(master_bias)
    nim = imarith(image, master_bias, '-', inplace=inplace,
                  merge_headers='first', merge_flags='no_merge')

    nim.header['HIERARCH astropop bias_corrected'] = True
    name = master_bias.origin_filename
    if name is not None:
        nim.header['HIERARCH astropop bias_corrected_file'] = name

    return nim


def subtract_dark(
    image: _ImageInput,
    master_dark: _ImageInput,
    dark_exposure: _QuantityInput,
    image_exposure: _QuantityInput,
    inplace: bool=False,
) -> FrameData:
    """
    Subtract master_dark frame from a FrameData.

    Notes
    -----
    * This function will just subtract a master dark frame from the original
      image. The master dark calculation, by handling several dark images,
      must be done previously.

    * Different exposure times among dark image and `~astropop.framedata.FrameData` are handled by
      a multiplying the Master dark image by ``image_exposure/dark_exposure``

    * If ``image`` is not a `~astropop.framedata.FrameData` instance,
      inplies in ``inplace=False``, and a new `~astropop.framedata.FrameData`
      instance will be created.

    Parameters
    ----------
    image
        Image to perform the dark correction. `~astropy.units.Quantity`,
        numerical values and `~astropy.nddata.CCDData` are also suported.
    master_dark
        Master dark image to be subtracted from the ``image``.
    dark_exposure
        Exposure time of the Master dark.
    image_exposure
        Exposure time of the ``image``.
    inplace
        If True, the operations will be performed inplace in the ``image``.

    Returns
    -------
    result
        New dark corrrected `~astropop.framedata.FrameData` instance if ``inplace``, else the
        ``image`` `~astropop.framedata.FrameData` instance.

    Examples
    --------
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.processing import subtract_dark
    >>> science = FrameData([[100.]], unit='adu')
    >>> dark = FrameData([[5.]], unit='adu')

    >>> # Both inputs are bias-subtracted; scale the dark from 10 to 20 seconds.
    >>> corrected = subtract_dark(science, dark, dark_exposure=10,
    ...                           image_exposure=20)
    >>> print(corrected.data.tolist())
    [[90.0]]
    """
    image = check_framedata(image)
    master_dark = check_framedata(master_dark)
    scale = image_exposure/dark_exposure
    if scale != 1:
        logger.debug('Scaling dark by %s factor to match image'
                     ' exposure.', scale)
        master_dark = imarith(master_dark, scale, "*", inplace=False)

    nim = imarith(image, master_dark, '-', inplace=inplace,
                  merge_headers='first', merge_flags='no_merge')

    nim.header['HIERARCH astropop dark_corrected'] = True
    nim.header['HIERARCH astropop dark_corrected_scale'] = scale
    name = master_dark.origin_filename
    if name is not None:
        name = os.path.basename(name)
        nim.header['HIERARCH astropop dark_corrected_file'] = name

    return nim


def flat_correct(
    image: _ImageInput,
    master_flat: _ImageInput,
    min_value: float | None=None,
    norm_value: _QuantityInput | None=None,
    inplace: bool=False,
) -> FrameData:
    """
    Divide the image by a flat field frame.

    Parameters
    ----------
    image
        Image to perform the flat field correction. `~astropy.units.Quantity`,
        numerical values and `~astropy.nddata.CCDData` are also suported.
    master_flat
        Master flat field image used as the divisor of the ``image``.
    min_value
        Replace smaller master-flat data values by this bound before
        normalization. None leaves values unchanged; this does not mask them.
    norm_value
        Divide the master flat by this explicit value before correcting the
        image. None performs no normalization.
    inplace
        If True, the operations will be performed inplace in the ``image``.

    Returns
    -------
    result
        New flat field corrected `~astropop.framedata.FrameData` instance if ``inplace``, else the
        ``image`` `~astropop.framedata.FrameData` instance.

    Examples
    --------
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.processing import flat_correct
    >>> science = FrameData([[80., 120.]], unit='adu')
    >>> flat = FrameData([[0.8, 1.2]])

    >>> # Divide by a dimensionless response normalized to unity.
    >>> corrected = flat_correct(science, flat)
    >>> print(corrected.data.tolist(), corrected.unit)
    [[100.0, 100.0]] adu
    """
    master_flat = check_framedata(master_flat, copy=True)
    image = check_framedata(image)

    if min_value is not None:
        logger.debug('Set lower flat value to %s', min_value)
        mask = master_flat.data < min_value
        master_flat.data[np.where(mask)] = min_value

    if norm_value is not None:
        logger.debug('Normalizing flat with %s value.', norm_value)
        master_flat = imarith(master_flat, norm_value, '/', inplace=False)

    nim = imarith(image, master_flat, '/', inplace=inplace,
                  merge_headers='first', merge_flags='no_merge')

    nim.header['HIERARCH astropop flat_corrected'] = True

    name = master_flat.origin_filename
    if name is not None:
        name = os.path.basename(name)
        nim.header['HIERARCH astropop flat_corrected_file'] = name

    return nim


def trim_image(
    image: _ImageInput,
    x_slice: slice | None=None,
    y_slice: slice | None=None,
    inplace: bool=False,
) -> FrameData:
    """Trim an image to a given section. Uses python slice standard.

    Parameters
    ----------
    image
        Image to be trimmed. `~astropy.units.Quantity`, numerical values and
        `~astropy.nddata.CCDData` are also suported.
    x_slice, y_slice
        Section to be trimmed in x and y axes.
    inplace
        If True, the operations will be performed inplace in the ``image``.

    Returns
    -------
    result
        Trimmed `~astropop.framedata.FrameData` instance.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.processing import trim_image
    >>> frame = FrameData(np.arange(12).reshape(3, 4))

    >>> # x selects columns and y selects rows, with exclusive upper bounds.
    >>> cutout = trim_image(frame, x_slice=slice(1, 3), y_slice=slice(0, 2))
    >>> print(cutout.data.tolist(), frame.shape)
    [[1.0, 2.0], [5.0, 6.0]] (3, 4)
    """
    image = check_framedata(image, copy=False)
    if not inplace:
        image = image.copy()

    # shape is always (y, x)
    x_slice = slice(0, image.shape[1]) if x_slice is None else x_slice
    y_slice = slice(0, image.shape[0]) if y_slice is None else y_slice

    section = (y_slice, x_slice)

    # trim the arrays
    data = image.data[section]
    uncertainty = image.uncertainty
    if uncertainty is not None:
        uncertainty = uncertainty[section]
    flags = image.flags
    if flags is not None:
        flags = flags[section]

    image.data = data
    image.uncertainty = uncertainty
    image.flags = flags

    # fix WCS if existing
    if image.wcs is not None:
        image.wcs = image.wcs.slice(section)

    str_slice = ''
    for s in section[::-1]:
        str_slice += ':'.join(str(i) if i is not None else ''
                              for i in [s.start, s.stop])
        str_slice += ','
    str_slice = str_slice[:-1]
    image.meta['HIERARCH astropop trimmed_section'] = str_slice

    return image
