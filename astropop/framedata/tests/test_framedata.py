# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Some parts stolen from Astropy CCDData testing bench

import pytest
import tempfile
import os
import numpy as np
from astropop.framedata.framedata import FrameData, setup_filename, \
                                         extract_units
from astropy.io import fits
from astropy.utils import NumpyRNGContext
from astropy import units as u
from astropy.wcs import WCS, FITSFixedWarning
from astropy.tests.helper import catch_warnings
from astropop.testing import assert_almost_equal, assert_equal, assert_true, \
                             assert_false, assert_is_instance, assert_in, \
                             assert_not_in, assert_is


# TODO: test history


DEFAULT_DATA_SIZE = 100
DEFAULT_HEADER = {'observer': 'astropop', 'very long key': 2}

with NumpyRNGContext(123):
    _random_array = np.random.normal(size=[DEFAULT_DATA_SIZE,
                                           DEFAULT_DATA_SIZE])


def create_framedata():
    data = _random_array.copy()
    fake_meta = DEFAULT_HEADER.copy()
    frame = FrameData(data, unit=u.Unit('adu'))
    frame.header = fake_meta
    return frame


@pytest.mark.parametrize("dunit,unit,expected",
                         [('adu', None, 'adu'),
                          ('adu', 'adu', 'adu'),
                          (None, 'adu', 'adu'),
                          ('adu', 'm', 'raise'),
                          (None, None, None)])
def test_extract_units(dunit, unit, expected):
    d = np.array([0, 1, 2, 3, 4])
    if dunit is not None:
        d = d*u.Unit(dunit)
    if expected == 'raise':
        with pytest.raises(ValueError):
            extract_units(d, unit)
    else:
        eunit = extract_units(d, unit)
        if expected is not None:
            expected = u.Unit(expected)
        assert_equal(eunit, expected)


def test_setup_filename(tmpdir):
    temp = os.path.abspath(tmpdir)
    fname = 'test_filename.npy'
    test_obj = FrameData(np.zeros(2), unit='adu',
                         cache_filename='test_filename.npy',
                         cache_folder=temp)

    assert_equal(setup_filename(test_obj), os.path.join(temp, fname))
    assert_true(tmpdir.exists())
    # Manual set filename
    ntemp = tempfile.mkstemp(suffix='.npy')[1]
    # with obj and manual filename, keep object
    assert_equal(setup_filename(test_obj, filename=ntemp),
                 os.path.join(temp, fname))
    test_obj.cache_filename = None
    assert_equal(setup_filename(test_obj, filename=ntemp),
                 os.path.join(temp, os.path.basename(ntemp)))
    # same for cache folder
    test_obj.cache_filename = fname
    assert_equal(setup_filename(test_obj, filename=ntemp),
                 os.path.join(temp, fname))
    test_obj.cache_folder = None
    cache = os.path.join(tmpdir, 'astropop_testing')
    assert_equal(setup_filename(test_obj, cache_folder=cache),
                 os.path.join(cache, fname))
    assert_true(os.path.isdir(cache))
    os.removedirs(cache)

    # now, with full random
    test_obj.cache_filename = None
    test_obj.cache_folder = None
    sfile = setup_filename(test_obj)
    dirname = os.path.dirname(sfile)
    filename = os.path.basename(sfile)
    assert_equal(dirname, test_obj.cache_folder)
    assert_equal(filename, test_obj.cache_filename)
    assert_true(os.path.exists(dirname))


# TODO:
@pytest.mark.skip('Copy not implemented.')
def test_copy():
    frame = create_framedata()
    ccd_copy = frame.copy()
    assert_equal(ccd_copy.data, frame.data)
    assert_equal(ccd_copy.unit, frame.unit)
    assert_equal(ccd_copy.meta, frame.meta)


class Test_FrameData_Creation():
    def test_framedata_cration_array(self):
        a = _random_array.copy()
        meta = DEFAULT_HEADER.copy()
        unit = 'adu'
        f = FrameData(a, unit=unit, meta=meta, dtype='float64')
        assert_almost_equal(a, f.data)
        assert_true(f.unit is u.adu)
        assert_true(np.issubdtype(f.dtype, np.float64))
        assert_equal(f.meta['observer'], meta['observer'])
        assert_equal(f.meta['very long key'], meta['very long key'])

    def test_framedata_cration_array_uncertainty(self):
        a = _random_array.copy()
        b = _random_array.copy()
        meta = DEFAULT_HEADER.copy()
        unit = 'adu'
        f = FrameData(a, unit=unit, meta=meta, uncertainty=b, u_dtype='float32')
        assert_almost_equal(a, f.data)
        assert_almost_equal(b, f.uncertainty)
        assert_equal(f.unit, u.adu)
        assert_true(np.issubdtype(f.uncertainty.dtype, np.float32))
        assert_equal(f.meta['observer'], meta['observer'])
        assert_equal(f.meta['very long key'], meta['very long key'])


    def test_framedata_cration_array_mask(self):
        a = _random_array.copy()
        b = np.zeros(_random_array.shape)
        meta = DEFAULT_HEADER.copy()
        unit = 'adu'
        f = FrameData(a, unit=unit, meta=meta, mask=b, m_dtype='bool')
        assert_almost_equal(a, f.data)
        assert_almost_equal(b, f.mask)
        assert_equal(f.unit, u.adu)
        assert_true(np.issubdtype(f.mask.dtype, np.bool))
        assert_equal(f.meta['observer'], meta['observer'])
        assert_equal(f.meta['very long key'], meta['very long key'])

    def test_framedata_cration_array_mask_flags(self):
        a = _random_array.copy()
        b = np.zeros(_random_array.shape).astype('int16')
        for i in range(8):
            b[i, i] = 1 << i
        meta = DEFAULT_HEADER.copy()
        unit = 'adu'
        f = FrameData(a, unit=unit, meta=meta, mask=b, m_dtype='uint8')
        assert_almost_equal(a, f.data)
        assert_almost_equal(b, f.mask)
        assert_equal(f.unit, u.adu)
        assert_true(np.issubdtype(f.mask.dtype, np.uint8))
        assert_equal(f.meta['observer'], meta['observer'])
        assert_equal(f.meta['very long key'], meta['very long key'])

    def test_framedata_empty(self):
        with pytest.raises(TypeError):
            # empty initializer should fail
            FrameData()

    def test_frame_simple(self):
        framedata = create_framedata()
        assert_equal(framedata.shape, (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE))
        assert_equal(framedata.size, DEFAULT_DATA_SIZE * DEFAULT_DATA_SIZE)
        assert_is(framedata.dtype, np.dtype(float))

    def test_frame_init_with_string_electron_unit(self):
        framedata = FrameData(np.zeros([2, 2]), unit="electron")
        assert_is(framedata.unit, u.electron)


class Test_FrameData_WCS():
    def test_wcs_invalid(self):
        frame = create_framedata()
        with pytest.raises(TypeError):
            frame.wcs = 5

    def test_wcs_assign(self):
        frame = create_framedata()
        wcs = WCS(naxis=2)
        frame.wcs = wcs
        assert_equal(frame.wcs, wcs)


class Test_FrameData_Meta():
    def test_framedata_meta_header(self):
        header = DEFAULT_HEADER.copy()
        meta = {'testing1': 'a', 'testing2': 'b'}
        header.update({'testing1': 'c'})

        a = FrameData([1, 2, 3], unit='', meta=meta, header=header)
        assert_equal(type(a.meta), dict)
        assert_equal(type(a.header), dict)
        assert_equal(a.meta['testing1'], 'c')  # Header priority
        assert_equal(a.meta['testing2'], 'b')
        assert_equal(a.header['testing1'], 'c')  # Header priority
        assert_equal(a.header['testing2'], 'b')
        for k in DEFAULT_HEADER.keys():
            assert_equal(a.header[k], DEFAULT_HEADER[k])
            assert_equal(a.meta[k], DEFAULT_HEADER[k])

    def test_framedata_meta_is_case_sensitive(self):
        frame = create_framedata()
        key = 'SoMeKEY'
        lkey = key.lower()
        ukey = key.upper()
        frame.meta[key] = 10
        assert_not_in(lkey, frame.meta)
        assert_not_in(ukey, frame.meta)
        assert_in(key, frame.meta)

    def test_metafromheader(self):
        hdr = fits.header.Header()
        hdr['observer'] = 'Edwin Hubble'
        hdr['exptime'] = '3600'

        d1 = FrameData(np.ones((5, 5)), meta=hdr, unit=u.electron)
        assert_equal(d1.meta['OBSERVER'], 'Edwin Hubble')
        assert_equal(d1.header['OBSERVER'], 'Edwin Hubble')

    def test_metafromdict(self):
        dic = {'OBSERVER': 'Edwin Hubble', 'EXPTIME': 3600}
        d1 = FrameData(np.ones((5, 5)), meta=dic, unit=u.electron)
        assert_equal(d1.meta['OBSERVER'], 'Edwin Hubble')

    def test_header2meta(self):
        hdr = fits.header.Header()
        hdr['observer'] = 'Edwin Hubble'
        hdr['exptime'] = '3600'

        d1 = FrameData(np.ones((5, 5)), unit=u.electron)
        d1.header = hdr
        assert_equal(d1.meta['OBSERVER'], 'Edwin Hubble')
        assert_equal(d1.header['OBSERVER'], 'Edwin Hubble')

    def test_metafromstring_fail(self):
        hdr = 'this is not a valid header'
        with pytest.raises(ValueError):
            FrameData(np.ones((5, 5)), meta=hdr, unit=u.adu)

    def test_framedata_meta_is_not_fits_header(self):
        frame = create_framedata()
        frame.meta = {'OBSERVER': 'Edwin Hubble'}
        assert_false(isinstance(frame.meta, fits.Header))


class Test_FrameData_Uncertainty():
    def test_setting_uncertainty_with_array(self):
        frame = create_framedata()
        frame.uncertainty = None
        fake_uncertainty = np.sqrt(np.abs(frame.data))
        frame.uncertainty = fake_uncertainty.copy()
        assert_equal(frame.uncertainty, fake_uncertainty)
        assert_equal(frame.unit, u.adu)

    def test_setting_uncertainty_with_scalar(self):
        uncertainty = 10
        frame = create_framedata()
        frame.uncertainty = None
        frame.uncertainty = uncertainty
        fake_uncertainty = np.zeros_like(frame.data)
        fake_uncertainty[:] = uncertainty
        assert_equal(frame.uncertainty, fake_uncertainty)
        assert_equal(frame.unit, u.adu)

    def test_setting_uncertainty_with_quantity(self):
        uncertainty = 10*u.adu
        frame = create_framedata()
        frame.uncertainty = None
        frame.uncertainty = uncertainty
        fake_uncertainty = np.zeros_like(frame.data)
        fake_uncertainty[:] = uncertainty.value
        assert_equal(frame.uncertainty, fake_uncertainty)
        assert_equal(frame.unit, u.adu)

    def test_setting_uncertainty_wrong_shape_raises_error(self):
        frame = create_framedata()
        with pytest.raises(ValueError):
            frame.uncertainty = np.zeros([3, 4])

    def test_setting_bad_uncertainty_raises_error(self):
        frame = create_framedata()
        with pytest.raises(TypeError):
            # Uncertainty is supposed to be an instance of NDUncertainty
            frame.uncertainty = 'not a uncertainty'

    def test_none_uncertainty_returns_zeros(self):
        frame = create_framedata()
        assert_equal(frame.uncertainty, np.zeros((DEFAULT_DATA_SIZE,
                                                  DEFAULT_DATA_SIZE)))


class Test_FrameData_FITS():
    def test_to_hdu_defaults(self):
        frame = create_framedata()
        frame.meta = {'observer': 'Edwin Hubble'}
        frame.uncertainty = np.random.rand(*frame.shape)
        frame.mask = np.zeros(frame.shape)
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
        frame.mask = np.zeros(frame.shape)
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

    def test_write_unit_to_hdu(self):
        frame = create_framedata()
        ccd_unit = frame.unit
        hdulist = frame.to_hdu()
        assert_true('bunit' in hdulist[0].header)
        assert_equal(hdulist[0].header['bunit'].strip(), ccd_unit.to_string())

    # TODO:
    @pytest.mark.skip('Wait Fits Implementation')
    def test_initialize_from_FITS_bad_keyword_raises_error(self, tmpdir):
        # There are two fits.open keywords that are not permitted in ccdproc:
        #     do_not_scale_image_data and scale_back
        frame = create_framedata()
        filename = tmpdir.join('test.fits').strpath
        frame.write_fits(filename)
        with pytest.raises(TypeError):
            FrameData.read_fits(filename, unit=frame.unit,
                                do_not_scale_image_data=True)
        with pytest.raises(TypeError):
            FrameData.read_fits(filename, unit=frame.unit, scale_back=True)
