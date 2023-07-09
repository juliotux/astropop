# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropop.framedata import FrameData
from astropop.testing import *
from .test_framedata import create_framedata


class TestFrameData2FITS:
    def test_to_hdu_defaults(self):
        frame = create_framedata()
        frame.meta = {'observer': 'Edwin Hubble'}
        frame.uncertainty = np.random.rand(*frame.shape)
        fits_hdulist = frame.to_hdu()
        assert_is_instance(fits_hdulist, fits.HDUList)
        for k, v in frame.meta.items():
            assert_equal(fits_hdulist[0].header[k], v)
        assert_equal(fits_hdulist[0].data, frame.data)
        assert_equal(fits_hdulist["UNCERT"].data,
                     frame.uncertainty)
        assert_equal(fits_hdulist["MASK"].data, frame.mask)
        assert_equal(fits_hdulist[0].header['BUNIT'], 'adu')

    def test_to_hdu_no_uncert_no_mask_names(self):
        frame = create_framedata()
        frame.meta = {'observer': 'Edwin Hubble'}
        frame.uncertainty = np.random.rand(*frame.shape)
        frame.mask_pixels(np.zeros(frame.shape, dtype=bool))
        fits_hdulist = frame.to_hdu(hdu_uncertainty=None, hdu_mask=None)
        assert_is_instance(fits_hdulist, fits.HDUList)
        for k, v in frame.meta.items():
            assert_equal(fits_hdulist[0].header[k], v)
        assert_equal(fits_hdulist[0].data, frame.data)
        with pytest.raises(KeyError):
            fits_hdulist['UNCERT']
        with pytest.raises(KeyError):
            fits_hdulist['MASK']
        assert_equal(fits_hdulist[0].header['BUNIT'], 'adu')

    def test_to_hdu_wcs(self):
        frame = create_framedata()
        wcs = WCS(naxis=2)
        wcsh = wcs.to_header(relax=True)
        frame.wcs = wcs
        hdul = frame.to_hdu()
        assert_is_instance(hdul, fits.HDUList)
        for k in wcsh:
            assert_equal(hdul[0].header[k], wcsh[k])

    def test_to_hdu_history(self):
        frame = create_framedata()
        frame.history = 'test 1'
        frame.history = ['test 2', 'test']

        hdul = frame.to_hdu()
        assert_is_instance(hdul, fits.HDUList)
        assert_equal(hdul[0].header['history'], ['test 1', 'test 2', 'test'])

    def test_to_hdu_comment(self):
        frame = create_framedata()
        frame.comment = 'test 1'
        frame.comment = ['test 2', 'test']

        hdul = frame.to_hdu()
        assert_is_instance(hdul, fits.HDUList)
        assert_equal(hdul[0].header['comment'], ['test 1', 'test 2', 'test'])

    def test_to_hdu_none_history(self):
        frame = create_framedata()
        frame._history = None

        hdul = frame.to_hdu()
        assert_is_instance(hdul, fits.HDUList)
        assert_not_in('HISTORY', hdul[0].header)

    def test_to_hdu_none_comment(self):
        frame = create_framedata()
        frame._comment = None

        hdul = frame.to_hdu()
        assert_is_instance(hdul, fits.HDUList)
        assert_not_in('COMMENT', hdul[0].header)

    def test_write_unit_to_hdu(self):
        frame = create_framedata()
        ccd_unit = frame.unit
        hdulist = frame.to_hdu()
        assert_true('bunit' in hdulist[0].header)
        assert_equal(hdulist[0].header['bunit'].strip(), ccd_unit.to_string())


class TestFrameDataWriteFits:
    def test_write_fits_defaults(self, tmpdir):
        fn = tmpdir.join('test.fits')
        frame = create_framedata()
        frame.meta = {'observer': 'Edwin Hubble'}
        frame.uncertainty = np.random.rand(*frame.shape)
        frame.mask_pixels(np.zeros(frame.shape, dtype=bool))
        frame.write(fn)
        hdul = fits.open(fn)
        assert_equal(hdul[0].data, frame.data)
        assert_equal(hdul["UNCERT"].data, frame.uncertainty)
        assert_equal(hdul["MASK"].data, frame.mask)
        assert_equal(hdul[0].header['BUNIT'], 'adu')

    def test_write_fits_no_uncert_no_mask_names(self, tmpdir):
        fn = tmpdir.join('test.fits')
        frame = create_framedata()
        frame.meta = {'observer': 'Edwin Hubble'}
        frame.uncertainty = np.random.rand(*frame.shape)
        frame.mask_pixels(np.zeros(frame.shape, dtype=bool))
        frame.write(fn, hdu_uncertainty=None, hdu_mask=None)
        hdul = fits.open(fn)
        assert_equal(hdul[0].data, frame.data)
        with pytest.raises(KeyError):
            hdul['UNCERT']
        with pytest.raises(KeyError):
            hdul['MASK']
        assert_equal(hdul[0].header['BUNIT'], 'adu')

    def test_write_fits_wcs(self, tmpdir):
        fn = tmpdir.join('test.fits')
        frame = create_framedata()
        wcs = WCS(naxis=2)
        wcsh = wcs.to_header(relax=True)
        frame.wcs = wcs
        frame.write(fn)
        hdul = fits.open(fn)
        for k in wcsh:
            assert_equal(hdul[0].header[k], wcsh[k])

    def test_write_no_std_units_default_true(self, tmpdir):
        fn = tmpdir.join('test.fits')
        frame = create_framedata()
        frame.unit = 'electron'
        frame.write(fn)
        hdul = fits.open(fn)
        assert_equal(hdul[0].header['BUNIT'], 'electron')

    def test_write_no_std_units_false(self, tmpdir):
        fn = tmpdir.join('test.fits')
        frame = create_framedata()
        frame.unit = 'electron'
        with pytest.raises(ValueError):
            frame.write(fn, no_fits_standard_units=False)


class TestFrameData2CCDData:
    shape = (10, 10)
    meta = {'OBSERVER': 'testing', 'very long keyword': 1}
    unit = 'adu'

    @property
    def mask(self):
        mask = np.zeros(self.shape, dtype=bool)
        mask[1:3, 1:3] = 1
        return mask

    @property
    def frame(self):
        data = 100*np.ones(self.shape)
        return FrameData(data, unit=self.unit, meta=self.meta)

    def test_simple_export(self):
        frame = self.frame
        ccd = frame.to_ccddata()

        assert_equal(ccd.data, 100*np.ones(self.shape))
        assert_equal(ccd.unit, self.unit)
        for i in self.meta.keys():
            assert_equal(ccd.meta[i], self.meta[i])
        assert_equal(ccd.mask, np.zeros(self.shape, dtype=bool))
        assert_is_none(ccd.uncertainty)

    def test_export_with_mask(self):
        frame = self.frame
        frame.mask_pixels(self.mask)

        ccd = frame.to_ccddata()
        assert_equal(ccd.data, 100*np.ones(self.shape))
        assert_equal(ccd.unit, self.unit)
        for i in self.meta.keys():
            assert_equal(ccd.meta[i], self.meta[i])
        assert_equal(ccd.mask, self.mask)
        assert_is_none(ccd.uncertainty, str(ccd.uncertainty))

    def test_export_with_uncertainty(self):
        frame = self.frame
        frame.uncertainty = 1

        ccd = frame.to_ccddata()
        assert_equal(ccd.data, 100*np.ones(self.shape))
        assert_equal(ccd.unit, self.unit)
        for i in self.meta.keys():
            assert_equal(ccd.meta[i], self.meta[i])
        assert_equal(ccd.mask, np.zeros(self.shape, dtype=bool))
        assert_equal(ccd.uncertainty.array, np.ones(self.shape))
        assert_equal(ccd.uncertainty.unit, self.unit)
        assert_equal(ccd.uncertainty.uncertainty_type, 'std')

    def test_export_full(self):
        frame = self.frame
        frame.uncertainty = 1
        frame.mask_pixels(self.mask)

        ccd = frame.to_ccddata()
        assert_equal(ccd.data, 100*np.ones(self.shape))
        assert_equal(ccd.unit, self.unit)
        for i in self.meta.keys():
            assert_equal(ccd.meta[i], self.meta[i])
        assert_equal(ccd.mask, self.mask)
        assert_equal(ccd.uncertainty.array, np.ones(self.shape))
        assert_equal(ccd.uncertainty.unit, self.unit)
        assert_equal(ccd.uncertainty.uncertainty_type, 'std')
