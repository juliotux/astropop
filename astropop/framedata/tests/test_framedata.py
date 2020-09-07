# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Some parts stolen from Astropy CCDData testing bench

import pytest
import tempfile
import os
import numpy as np
import numpy.testing as npt
import pytest_check as check
from astropop.framedata.framedata import FrameData, setup_filename, \
                                         extract_units
from astropy.io import fits
from astropy.utils import NumpyRNGContext
from astropy import units as u
from astropy.wcs import WCS, FITSFixedWarning
from astropy.tests.helper import catch_warnings


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
        check.is_true(eunit == expected)


def test_setup_filename(tmpdir):
    temp = os.path.abspath(tmpdir)
    fname = 'test_filename.npy'
    test_obj = FrameData(np.zeros(2), unit='adu',
                         cache_filename='test_filename.npy',
                         cache_folder=temp)

    check.equal(setup_filename(test_obj), os.path.join(temp, fname))
    check.is_true(tmpdir.exists())
    # Manual set filename
    ntemp = tempfile.mkstemp(suffix='.npy')[1]
    # with obj and manual filename, keep object
    check.equal(setup_filename(test_obj, filename=ntemp),
                os.path.join(temp, fname))
    test_obj.cache_filename = None
    check.equal(setup_filename(test_obj, filename=ntemp),
                os.path.join(temp, os.path.basename(ntemp)))
    # same for cache folder
    test_obj.cache_filename = fname
    check.equal(setup_filename(test_obj, filename=ntemp),
                os.path.join(temp, fname))
    test_obj.cache_folder = None
    cache = os.path.join(tmpdir, 'astropop_testing')
    check.equal(setup_filename(test_obj, cache_folder=cache),
                os.path.join(cache, fname))
    check.is_true(os.path.isdir(cache))
    os.removedirs(cache)

    # now, with full random
    test_obj.cache_filename = None
    test_obj.cache_folder = None
    sfile = setup_filename(test_obj)
    dirname = os.path.dirname(sfile)
    filename = os.path.basename(sfile)
    check.equal(dirname, test_obj.cache_folder)
    check.equal(filename, test_obj.cache_filename)
    check.is_true(os.path.exists(dirname))


def test_framedata_cration_array():
    a = _random_array.copy()
    meta = DEFAULT_HEADER.copy()
    unit = 'adu'
    f = FrameData(a, unit=unit, meta=meta, dtype='float64')
    npt.assert_array_almost_equal(a, f.data)
    check.is_true(f.unit is u.adu)
    check.is_true(np.issubdtype(f.dtype, np.float64))
    check.equal(f.meta['observer'], meta['observer'])
    check.equal(f.meta['very long key'], meta['very long key'])


def test_framedata_cration_array_uncertainty():
    a = _random_array.copy()
    b = _random_array.copy()
    meta = DEFAULT_HEADER.copy()
    unit = 'adu'
    f = FrameData(a, unit=unit, meta=meta, uncertainty=b, u_dtype='float32')
    npt.assert_array_almost_equal(a, f.data)
    npt.assert_array_almost_equal(b, f.uncertainty)
    check.is_true(f.unit is u.adu)
    check.is_true(f.uncertainty.unit is u.adu)
    check.is_true(np.issubdtype(f.uncertainty.dtype, np.float32))
    check.equal(f.meta['observer'], meta['observer'])
    check.equal(f.meta['very long key'], meta['very long key'])


def test_framedata_cration_array_mask():
    a = _random_array.copy()
    b = np.zeros(_random_array.shape)
    meta = DEFAULT_HEADER.copy()
    unit = 'adu'
    f = FrameData(a, unit=unit, meta=meta, mask=b, m_dtype='bool')
    npt.assert_array_almost_equal(a, f.data)
    npt.assert_array_almost_equal(b, f.mask)
    check.is_true(f.unit is u.adu)
    check.is_true(np.issubdtype(f.mask.dtype, np.bool))
    check.equal(f.meta['observer'], meta['observer'])
    check.equal(f.meta['very long key'], meta['very long key'])


def test_framedata_cration_array_mask_flags():
    a = _random_array.copy()
    b = np.zeros(_random_array.shape).astype('int16')
    for i in range(8):
        b[i, i] = 1 << i
    meta = DEFAULT_HEADER.copy()
    unit = 'adu'
    f = FrameData(a, unit=unit, meta=meta, mask=b, m_dtype='uint8')
    npt.assert_array_almost_equal(a, f.data)
    npt.assert_array_almost_equal(b, f.mask)
    check.is_true(f.unit is u.adu)
    check.is_true(np.issubdtype(f.mask.dtype, np.uint8))
    check.equal(f.meta['observer'], meta['observer'])
    check.equal(f.meta['very long key'], meta['very long key'])


def test_framedata_empty():
    with pytest.raises(TypeError):
        # empty initializer should fail
        FrameData()  # noqa


def test_framedata_meta_header():
    header = DEFAULT_HEADER.copy()
    meta = {'testing1': 'a', 'testing2': 'b'}
    header.update({'testing1': 'c'})

    a = FrameData([1, 2, 3], unit='', meta=meta, header=header)
    check.equal(type(a.meta), dict)
    check.equal(type(a.header), dict)
    check.equal(a.meta['testing1'], 'c')  # Header priority
    check.equal(a.meta['testing2'], 'b')
    check.equal(a.header['testing1'], 'c')  # Header priority
    check.equal(a.header['testing2'], 'b')
    for k in DEFAULT_HEADER.keys():
        check.equal(a.header[k], DEFAULT_HEADER[k])
        check.equal(a.meta[k], DEFAULT_HEADER[k])


def test_frame_simple():
    framedata = create_framedata()
    check.equal(framedata.shape, (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE))
    check.equal(framedata.size, DEFAULT_DATA_SIZE * DEFAULT_DATA_SIZE)
    check.is_true(framedata.dtype is np.dtype(float))


def test_frame_init_with_string_electron_unit():
    framedata = FrameData(np.zeros([2, 2]), unit="electron")
    check.is_true(framedata.unit is u.electron)


def test_framedata_meta_is_case_sensitive():
    frame = create_framedata()
    key = 'SoMeKEY'
    lkey = key.lower()
    ukey = key.upper()
    frame.meta[key] = 10
    check.is_true(lkey not in frame.meta)
    check.is_true(ukey not in frame.meta)
    check.is_true(key in frame.meta)


def test_metafromheader():
    hdr = fits.header.Header()
    hdr['observer'] = 'Edwin Hubble'
    hdr['exptime'] = '3600'

    d1 = FrameData(np.ones((5, 5)), meta=hdr, unit=u.electron)
    check.equal(d1.meta['OBSERVER'], 'Edwin Hubble')
    check.equal(d1.header['OBSERVER'], 'Edwin Hubble')


def test_metafromdict():
    dic = {'OBSERVER': 'Edwin Hubble', 'EXPTIME': 3600}
    d1 = FrameData(np.ones((5, 5)), meta=dic, unit=u.electron)
    check.equal(d1.meta['OBSERVER'], 'Edwin Hubble')


def test_header2meta():
    hdr = fits.header.Header()
    hdr['observer'] = 'Edwin Hubble'
    hdr['exptime'] = '3600'

    d1 = FrameData(np.ones((5, 5)), unit=u.electron)
    d1.header = hdr
    check.equal(d1.meta['OBSERVER'], 'Edwin Hubble')
    check.equal(d1.header['OBSERVER'], 'Edwin Hubble')


def test_metafromstring_fail():
    hdr = 'this is not a valid header'
    with pytest.raises(ValueError):
        FrameData(np.ones((5, 5)), meta=hdr, unit=u.adu)


def test_framedata_meta_is_not_fits_header():
    frame = create_framedata()
    frame.meta = {'OBSERVER': 'Edwin Hubble'}
    check.is_false(isinstance(frame.meta, fits.Header))


def test_setting_uncertainty_with_array():
    frame = create_framedata()
    frame.uncertainty = None
    fake_uncertainty = np.sqrt(np.abs(frame.data))
    frame.uncertainty = fake_uncertainty.copy()
    np.testing.assert_array_equal(frame.uncertainty, fake_uncertainty)
    check.is_true(frame.uncertainty.unit is u.adu)


def test_setting_uncertainty_with_scalar():
    uncertainty = 10
    frame = create_framedata()
    frame.uncertainty = None
    frame.uncertainty = uncertainty
    fake_uncertainty = np.zeros_like(frame.data)
    fake_uncertainty[:] = uncertainty
    np.testing.assert_array_equal(frame.uncertainty, fake_uncertainty)
    check.is_true(frame.uncertainty.unit is u.adu)


def test_setting_uncertainty_with_quantity():
    uncertainty = 10*u.adu
    frame = create_framedata()
    frame.uncertainty = None
    frame.uncertainty = uncertainty
    fake_uncertainty = np.zeros_like(frame.data)
    fake_uncertainty[:] = uncertainty.value
    np.testing.assert_array_equal(frame.uncertainty, fake_uncertainty)
    check.is_true(frame.uncertainty.unit is u.adu)


def test_setting_uncertainty_wrong_shape_raises_error():
    frame = create_framedata()
    with pytest.raises(ValueError):
        frame.uncertainty = np.zeros([3, 4])


def test_to_hdu_defaults():
    frame = create_framedata()
    frame.meta = {'observer': 'Edwin Hubble'}
    frame.uncertaint = np.random.rand(*frame.shape)
    frame.mask = np.zeros(frame.shape)
    fits_hdulist = frame.to_hdu()
    check.is_instance(fits_hdulist, fits.HDUList)
    for k, v in frame.meta.items():
        check.equal(fits_hdulist[0].header[k], v)
    np.testing.assert_array_equal(fits_hdulist[0].data, frame.data)
    np.testing.assert_array_equal(fits_hdulist["UNCERT"].data,
                                  frame.uncertainty)
    np.testing.assert_array_equal(fits_hdulist["MASK"].data, frame.mask)
    check.equal(fits_hdulist[0].header['BUNIT'], 'adu')


def test_to_hdu_no_uncert_no_mask():
    frame = create_framedata()
    frame.meta = {'observer': 'Edwin Hubble'}
    frame.uncertaint = np.random.rand(*frame.shape)
    frame.mask = np.zeros(frame.shape)
    fits_hdulist = frame.to_hdu(hdu_uncertainty=None, hdu_mask=None)
    check.is_instance(fits_hdulist, fits.HDUList)
    for k, v in frame.meta.items():
        check.equal(fits_hdulist[0].header[k], v)
    np.testing.assert_array_equal(fits_hdulist[0].data, frame.data)
    with pytest.raises(KeyError):
        fits_hdulist['UNCERT']
    with pytest.raises(KeyError):
        fits_hdulist['MASK']
    check.equal(fits_hdulist[0].header['BUNIT'], 'adu')


# TODO:
@pytest.mark.skip('Wait Fits Implementation')
def test_initialize_from_FITS(tmpdir):
    frame = create_framedata()
    hdu = fits.PrimaryHDU(frame.data, header=fits.Header(frame.header))
    hdulist = fits.HDUList([hdu])
    filename = tmpdir.join('afile.fits').strpath
    hdulist.writeto(filename)
    cd = FrameData.read_fits(filename, unit=u.electron)
    check.equal(cd.shape, (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE))
    check.equal(cd.size, DEFAULT_DATA_SIZE * DEFAULT_DATA_SIZE)
    check.is_true(np.issubdtype(cd.data.dtype, np.floating))
    for k, v in hdu.header.items():
        check.equal(cd.meta[k], v)


# TODO:
@pytest.mark.skip('Wait Fits Implementation')
def test_initialize_from_FITS_memmap(tmpdir):
    frame = create_framedata()
    hdu = fits.PrimaryHDU(frame.data, header=fits.Header(frame.header))
    hdulist = fits.HDUList([hdu])
    filename = tmpdir.join('afile.fits').strpath
    hdulist.writeto(filename)
    # Same with memmap
    cd1 = FrameData.read_fits(filename, unit=u.electron,
                              use_memmap_backend=True)
    check.equal(cd1.shape, (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE))
    check.equal(cd1.size, DEFAULT_DATA_SIZE * DEFAULT_DATA_SIZE)
    check.is_true(np.issubdtype(cd1.data.dtype, np.floating))
    for k, v in hdu.header.items():
        check.equal(cd1.meta[k], v)
    check.is_instance(cd1.data, np.memmap)


# TODO:
@pytest.mark.skip('Wait Fits Implementation')
def test_initialize_from_fits_with_unit_in_header(tmpdir):
    fake_img = np.zeros([2, 2])
    hdu = fits.PrimaryHDU(fake_img)
    hdu.header['bunit'] = u.adu.to_string()
    filename = tmpdir.join('afile.fits').strpath
    hdu.writeto(filename)
    ccd = FrameData.read_fits(filename)
    # ccd should pick up the unit adu from the fits header...did it?
    check.is_true(ccd.unit is u.adu)

    # An explicit unit in the read overrides any unit in the FITS file
    ccd2 = FrameData.read_fits(filename, unit="photon")
    check.is_true(ccd2.unit is u.photon)


# TODO:
@pytest.mark.skip('Wait Fits Implementation')
def test_initialize_from_fits_with_ADU_in_header(tmpdir):
    fake_img = np.zeros([2, 2])
    hdu = fits.PrimaryHDU(fake_img)
    hdu.header['bunit'] = 'ADU'
    filename = tmpdir.join('afile.fits').strpath
    hdu.writeto(filename)
    ccd = FrameData.read_fits(filename)
    # ccd should pick up the unit adu from the fits header...did it?
    check.is_true(ccd.unit is u.adu)


# TODO:
@pytest.mark.skip('Wait Fits Implementation')
def test_initialize_from_fits_with_invalid_unit_in_header(tmpdir):
    hdu = fits.PrimaryHDU(np.ones((2, 2)))
    hdu.header['bunit'] = 'definetely-not-a-unit'
    filename = tmpdir.join('afile.fits').strpath
    hdu.writeto(filename)
    with pytest.raises(ValueError):
        FrameData.read_fits(filename)


# TODO:
@pytest.mark.skip('Wait Fits Implementation')
def test_initialize_from_fits_with_data_in_different_extension(tmpdir):
    fake_img = np.arange(4).reshape(2, 2)
    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU(fake_img)
    hdus = fits.HDUList([hdu1, hdu2])
    filename = tmpdir.join('afile.fits').strpath
    hdus.writeto(filename)
    with catch_warnings(FITSFixedWarning) as w:
        ccd = FrameData.read_fits(filename, unit='adu')
    check.equal(len(w), 0)
    np.testing.assert_array_equal(ccd.data, fake_img)
    # FIXME: why?
    # check.equal(hdu2.header + hdu1.header, ccd.header)


# TODO:
@pytest.mark.skip('Wait Fits Implementation')
def test_initialize_from_fits_with_extension(tmpdir):
    fake_img1 = np.zeros([2, 2])
    fake_img2 = np.arange(4).reshape(2, 2)
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(fake_img1)
    hdu2 = fits.ImageHDU(fake_img2)
    hdus = fits.HDUList([hdu0, hdu1, hdu2])
    filename = tmpdir.join('afile.fits').strpath
    hdus.writeto(filename)
    ccd = FrameData.read_fits(filename, hdu=2, unit='adu')
    np.testing.assert_array_equal(ccd.data, fake_img2)


def test_write_unit_to_hdu():
    frame = create_framedata()
    ccd_unit = frame.unit
    hdulist = frame.to_hdu()
    check.is_true('bunit' in hdulist[0].header)
    check.equal(hdulist[0].header['bunit'].strip(), ccd_unit.to_string())


# TODO:
@pytest.mark.skip('Wait Fits Implementation')
def test_initialize_from_FITS_bad_keyword_raises_error(tmpdir):
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


# TODO:
@pytest.mark.skip('Wait Fits Implementation')
def test_framedata_writer(tmpdir):
    frame = create_framedata()
    filename = tmpdir.join('test.fits').strpath
    frame.write_fits(filename)
    ccd_disk = FrameData.read_fits(filename)
    np.testing.assert_array_equal(frame.data, ccd_disk.data)


# TODO:
@pytest.mark.skip('Wait Fits Implementation')
def test_fromMEF(tmpdir):
    frame = create_framedata()
    hdu = frame.to_hdu()[0]
    hdu2 = fits.PrimaryHDU(2 * frame.data)
    hdulist = fits.HDUList(hdu)
    hdulist.append(hdu2)
    filename = tmpdir.join('afile.fits').strpath
    hdulist.writeto(filename)
    # by default, we reading from the first extension
    cd = FrameData.read_fits(filename, unit=u.electron)
    np.testing.assert_array_equal(cd.data, frame.data)
    # but reading from the second should work too
    cd = FrameData.read_fits(filename, hdu=1, unit=u.electron)
    np.testing.assert_array_equal(cd.data, 2 * frame.data)


# TODO:
@pytest.mark.skip('Wait Fits Implementation')
def test_setting_bad_uncertainty_raises_error():
    frame = create_framedata()
    with pytest.raises(TypeError):
        # Uncertainty is supposed to be an instance of NDUncertainty
        frame.uncertainty = 'not a uncertainty'


# TODO:
@pytest.mark.skip('Wait Fits Implementation')
def test_copy():
    frame = create_framedata()
    ccd_copy = frame.copy()
    np.testing.assert_array_equal(ccd_copy.data, frame.data)
    check.equal(ccd_copy.unit, frame.unit)
    check.equal(ccd_copy.meta, frame.meta)


def test_wcs_invalid():
    frame = create_framedata()
    with pytest.raises(TypeError):
        frame.wcs = 5


def test_wcs_assign():
    frame = create_framedata()
    wcs = WCS(naxis=2)
    frame.wcs = wcs
    check.equal(frame.wcs, wcs)
