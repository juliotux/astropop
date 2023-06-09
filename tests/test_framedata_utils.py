# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from astropop.framedata import check_framedata, read_framedata, FrameData, \
                               PixelMaskFlags
from astropop.math import QFloat
from astropy.io import fits
from astropy import units as u
from astropy.nddata import CCDData, StdDevUncertainty
from astropop.testing import *

from .test_framedata import create_framedata, _random_array, \
                            DEFAULT_DATA_SIZE, DEFAULT_HEADER


class Test_CheckRead_FrameData:
    # TODO: test .fz files and CompImageHDU

    def test_check_framedata_framedata(self):
        frame = create_framedata()
        fc = check_framedata(frame)
        fr = read_framedata(frame)
        # default must not copy
        assert_is(fc, frame)
        assert_is(fr, frame)

    def test_check_framedata_framedata_copy(self):
        frame = create_framedata()
        fc = check_framedata(frame, copy=True)
        fr = read_framedata(frame, copy=True)

        for f in (fc, fr):
            assert_is_not(f, frame)
            assert_is_instance(f, FrameData)
            assert_equal(f.data, frame.data)
            assert_equal(f.meta, frame.meta)
            assert_equal(f.uncertainty, frame.uncertainty)
            assert_equal(f.mask, frame.mask)
            assert_equal(f.flags, frame.flags)
            assert_equal(f.unit, frame.unit)
            assert_equal(f.history, frame.history)
            assert_equal(f.wcs, frame.wcs)

    def test_check_framedata_framedata_copy_with_dtype(self):
        frame = create_framedata(uncertainty=1)  # default is float64
        fc = check_framedata(frame, copy=True, dtype=np.float32)
        fr = read_framedata(frame, copy=True, dtype=np.float32)

        for f in (fc, fr):
            assert_is_not(f, frame)
            assert_is_instance(f, FrameData)
            assert_almost_equal(f.data, frame.data)
            assert_equal(f.meta, frame.meta)
            assert_equal(f.uncertainty, frame.uncertainty)
            assert_equal(f.mask, frame.mask)
            assert_equal(f.unit, frame.unit)
            assert_equal(f.history, frame.history)
            assert_equal(f.wcs, frame.wcs)
            assert_equal(f.data.dtype, np.float32)
            assert_equal(f.uncertainty.dtype, np.float32)

    def test_check_framedata_fits_hdu(self):
        # meta is messed by fits
        data = _random_array.copy()
        hdu = fits.PrimaryHDU(data)
        fc = check_framedata(hdu)
        fr = read_framedata(hdu)
        for f in (fc, fr):
            assert_is_instance(f, FrameData)
            assert_equal(f.data, data)
            assert_equal(f.unit, u.dimensionless_unscaled)
            assert_is_none(f._unct)
            assert_false(np.any(f.mask))
            assert_equal(f.flags, np.zeros_like(data, dtype=np.uint8))

    def test_check_framedata_fits_hdul_simple(self):
        # meta is messed by fits
        data = _random_array.copy()
        hdul = fits.HDUList([fits.PrimaryHDU(data)])
        fc = check_framedata(hdul)
        fr = read_framedata(hdul)
        for f in (fc, fr):
            assert_is_instance(f, FrameData)
            assert_equal(f.data, data)
            assert_equal(f.unit, u.dimensionless_unscaled)
            assert_is_none(f._unct)
            assert_false(np.any(f.mask))
            assert_equal(f.flags, np.zeros_like(data, dtype=np.uint8))

    def test_check_framedata_fits_hdul_defaults(self):
        data = _random_array.copy()
        header = fits.Header({'bunit': 'adu'})
        data_hdu = fits.PrimaryHDU(data, header=header)
        uncert_hdu = fits.ImageHDU(np.ones((DEFAULT_DATA_SIZE,
                                            DEFAULT_DATA_SIZE)), name='UNCERT')
        mask = np.zeros((DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)).astype('uint8')
        mask[1:3, 1:3] = 1
        mask_hdu = fits.ImageHDU(mask, name='MASK')
        hdul = fits.HDUList([data_hdu, uncert_hdu, mask_hdu])

        expect_flags = np.zeros((DEFAULT_DATA_SIZE,
                                 DEFAULT_DATA_SIZE)).astype('uint8')
        expect_flags[1:3, 1:3] = (PixelMaskFlags.MASKED |
                                  PixelMaskFlags.UNSPECIFIED).value

        fc = check_framedata(hdul)
        fr = read_framedata(hdul)
        for f in (fc, fr):
            assert_is_instance(f, FrameData)
            assert_equal(f.data, data)
            assert_equal(f.unit, 'adu')
            assert_equal(f.uncertainty, np.ones((DEFAULT_DATA_SIZE,
                                                 DEFAULT_DATA_SIZE)))
            assert_equal(f.mask, mask)
            assert_equal(f.flags, expect_flags)

    def test_check_framedata_fits_hdul_keywords(self):
        uncert_name = 'ASTROPOP_UNCERT'
        mask_name = 'ASTROPOP_MASK'
        data_name = 'ASTROPOP_DATA'
        p_hdu = fits.PrimaryHDU()
        data = _random_array.copy()
        header = fits.Header({'astrunit': 'adu'})
        data_hdu = fits.ImageHDU(data, header=header, name=data_name)
        uncert_hdu = fits.ImageHDU(np.ones((DEFAULT_DATA_SIZE,
                                            DEFAULT_DATA_SIZE)),
                                   name=uncert_name)
        mask = np.zeros((DEFAULT_DATA_SIZE,
                         DEFAULT_DATA_SIZE)).astype('uint8')
        mask[1:3, 1:3] = 1
        mask_hdu = fits.ImageHDU(mask, name=mask_name)
        hdul = fits.HDUList([p_hdu, data_hdu, uncert_hdu, mask_hdu])

        fc = check_framedata(hdul, hdu=data_name, unit=None,
                             hdu_uncertainty=uncert_name, hdu_mask=mask_name,
                             unit_key='astrunit')
        fr = read_framedata(hdul, hdu=data_name, unit=None,
                            hdu_uncertainty=uncert_name, hdu_mask=mask_name,
                            unit_key='astrunit')
        for f in (fc, fr):
            assert_is_instance(f, FrameData)
            assert_equal(f.data, data)
            assert_equal(f.unit, 'adu')
            assert_equal(f.uncertainty, np.ones((DEFAULT_DATA_SIZE,
                                                 DEFAULT_DATA_SIZE)))
            assert_equal(f.mask, mask)

    def test_check_framedata_fits_unit(self):
        data = _random_array.copy()
        hdu = fits.PrimaryHDU(data)
        fc = check_framedata(hdu, unit='adu')
        fr = read_framedata(hdu, unit='adu')
        for f in (fc, fr):
            assert_is_instance(f, FrameData)
            assert_equal(f.data, data)
            assert_equal(f.unit, 'adu')

    def test_check_framedata_fitsfile(self, tmp_path):
        tmp = tmp_path / 'fest_check_framedata.fits'
        tmpstr = str(tmp)

        uncert_name = 'ASTROPOP_UNCERT'
        mask_name = 'ASTROPOP_MASK'
        data_name = 'ASTROPOP_DATA'
        p_hdu = fits.PrimaryHDU()
        data = _random_array.copy()
        header = fits.Header({'astrunit': 'adu'})
        data_hdu = fits.ImageHDU(data, header=header, name=data_name)
        uncert_hdu = fits.ImageHDU(np.ones((DEFAULT_DATA_SIZE,
                                            DEFAULT_DATA_SIZE)),
                                   name=uncert_name)
        mask = np.zeros((DEFAULT_DATA_SIZE,
                         DEFAULT_DATA_SIZE)).astype('uint8')
        mask[1:3, 1:3] = 1
        mask_hdu = fits.ImageHDU(mask, name=mask_name)
        hdul = fits.HDUList([p_hdu, data_hdu, uncert_hdu, mask_hdu])
        hdul.writeto(tmpstr)

        # FIXME: must work with both string and pathlike
        # but astropy seems to not read the path lile
        n = tmpstr
        for mmap in [False, True]:
            fc = check_framedata(n, hdu=data_name, unit=None,
                                 hdu_uncertainty=uncert_name,
                                 hdu_mask=mask_name,
                                 unit_key='astrunit',
                                 use_memmap_backend=mmap)
            fr = read_framedata(n, hdu=data_name, unit=None,
                                hdu_uncertainty=uncert_name,
                                hdu_mask=mask_name,
                                unit_key='astrunit',
                                use_memmap_backend=mmap)
            for f in (fc, fr):
                assert_is_instance(f, FrameData)
                assert_equal(f.data, data)
                assert_equal(f.unit, 'adu')
                assert_equal(f.uncertainty,
                             np.ones((DEFAULT_DATA_SIZE,
                                      DEFAULT_DATA_SIZE)))
                assert_equal(f.mask, mask)
                assert_equal(f._memmapping, mmap)

    def test_check_framedata_ccddata(self):
        data = _random_array.copy()
        header = DEFAULT_HEADER.copy()
        unit = 'adu'
        uncert = 0.1*_random_array
        mask = np.zeros((DEFAULT_DATA_SIZE,
                         DEFAULT_DATA_SIZE)).astype('uint8')
        mask[1:3, 1:3] = 1
        ccd = CCDData(data, unit=unit,
                      uncertainty=StdDevUncertainty(uncert, unit=unit),
                      mask=mask, meta=header)

        for mmap in [False, True]:
            fc = check_framedata(ccd, use_memmap_backend=mmap)
            fr = read_framedata(ccd, use_memmap_backend=mmap)
            for f in (fc, fr):
                assert_is_instance(f, FrameData)
                assert_equal(f.data, data)
                assert_equal(f.unit, unit)
                assert_equal(f.uncertainty, uncert)
                assert_equal(f.mask, mask)
                assert_equal(f._memmapping, mmap)

    def test_check_framedata_quantity(self):
        data = _random_array.copy()*u.Unit('adu')
        for mmap in [False, True]:
            fc = check_framedata(data, use_memmap_backend=mmap)
            fr = read_framedata(data, use_memmap_backend=mmap)
            for f in (fc, fr):
                assert_is_instance(f, FrameData)
                assert_equal(f.data, _random_array)
                assert_equal(f.unit, 'adu')
                assert_is_none(f._unct)
                assert_false(np.any(f.mask))
                assert_equal(f.meta, {})
                assert_is_none(f.wcs)
                assert_equal(f._memmapping, mmap)

    def test_check_framedata_nparray(self):
        data = _random_array.copy()

        for mmap in [False, True]:
            fc = check_framedata(data, use_memmap_backend=mmap)
            fr = read_framedata(data, use_memmap_backend=mmap)
            for f in (fc, fr):
                assert_is_instance(f, FrameData)
                assert_equal(f.data, _random_array)
                assert_equal(f.unit, u.dimensionless_unscaled)
                assert_is_none(f._unct)
                assert_false(np.any(f.mask))
                assert_equal(f.meta, {})
                assert_is_none(f.wcs)
                assert_equal(f._memmapping, mmap)

    def test_check_framedata_qfloat(self):
        data = _random_array.copy()
        unit = 'adu'
        uncert = 0.1*np.ones_like(_random_array)
        qf = QFloat(data, uncert, unit)

        for mmap in [False, True]:
            fc = check_framedata(qf, use_memmap_backend=mmap)
            fr = read_framedata(qf, use_memmap_backend=mmap)
            for f in (fc, fr):
                assert_is_instance(f, FrameData)
                assert_equal(f.data, data)
                assert_equal(f.unit, unit)
                assert_equal(f._unct, uncert)
                assert_false(np.any(f.mask))
                assert_equal(f.meta, {})
                assert_is_none(f.wcs)
                assert_equal(f._memmapping, mmap)

    def test_check_framedata_invalid(self):
        # None should fail
        with pytest.raises(TypeError):
            check_framedata(None)
        with pytest.raises(TypeError):
            read_framedata(None)
