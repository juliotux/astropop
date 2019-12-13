# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Some parts stolen from Astropy CCDData testing bench

import pytest
import tempfile
import textwrap
import os
import numpy as np
import numpy.testing as npt
from astropop.framedata import FrameData, shape_consistency, unit_consistency, \
                               setup_filename, framedata_read_fits, \
                               framedata_to_hdu, extract_units
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
    """
    Return a FrameData object of size DEFAULT_DATA_SIZE x DEFAULT_DATA_SIZE
    with units of ADU.
    """
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
    d = np.array([0,1,2,3,4])
    if dunit is not None:
        d = d*u.Unit(dunit)
    if expected == 'raise':
        with pytest.raises(ValueError):
            extract_units(d, unit)
    else:
        eunit = extract_units(d, unit)
        if expected is not None:
            expected = u.Unit(expected)
        assert eunit is expected


def test_setup_filename(tmpdir):
    temp = os.path.abspath(tmpdir)
    fname = 'test_filename.npy'
    test_obj = FrameData(np.zeros(2), unit='adu',
                         cache_filename='test_filename.npy',
                         cache_folder=temp)

    assert setup_filename(test_obj) == os.path.join(temp, fname)
    assert tmpdir.exists()
    # Manual set filename
    ntemp = tempfile.mkstemp(suffix='.npy')[1]
    # with obj and manual filename, keep object
    assert setup_filename(test_obj, filename=ntemp) == os.path.join(temp, fname)
    test_obj.cache_filename = None
    assert setup_filename(test_obj, filename=ntemp) == os.path.join(temp, os.path.basename(ntemp))
    # same for cache folder
    test_obj.cache_filename = fname
    assert setup_filename(test_obj, filename=ntemp) == os.path.join(temp, fname)
    test_obj.cache_folder = None
    cache = os.path.join(tmpdir, 'astropop_testing')
    assert setup_filename(test_obj, cache_folder=cache) == os.path.join(cache, fname)
    assert os.path.isdir(cache)
    os.removedirs(cache)

    # now, with full random
    test_obj.cache_filename = None
    test_obj.cache_folder = None
    sfile = setup_filename(test_obj)
    dirname = os.path.dirname(sfile)
    filename = os.path.basename(sfile)
    assert dirname == test_obj.cache_folder
    assert filename == test_obj.cache_filename
    assert os.path.exists(dirname)


def test_framedata_cration_array():
    a = _random_array.copy()
    meta = DEFAULT_HEADER.copy()
    unit = 'adu'
    f = FrameData(a, unit=unit, meta=meta, dtype='float64')
    npt.assert_array_almost_equal(a, f.data)
    assert f.unit is u.adu
    assert np.issubdtype(f.dtype, np.float64)
    assert f.meta['observer'] == meta['observer']
    assert f.meta['very long key'] == meta['very long key']
    

def test_framedata_cration_array_uncertainty():
    a = _random_array.copy()
    b = _random_array.copy()
    meta = DEFAULT_HEADER.copy()
    unit = 'adu'
    f = FrameData(a, unit=unit, meta=meta, uncertainty=b, u_unit=unit, u_dtype='float32')
    npt.assert_array_almost_equal(a, f.data)
    npt.assert_array_almost_equal(b, f.uncertainty)
    assert f.unit is u.adu
    assert f.uncertainty.unit is u.adu
    assert np.issubdtype(f.uncertainty.dtype, np.float32)
    assert f.meta['observer'] == meta['observer']
    assert f.meta['very long key'] == meta['very long key']
    

def test_framedata_cration_array_mask():
    a = _random_array.copy()
    b = np.zeros(_random_array.shape)
    meta = DEFAULT_HEADER.copy()
    unit = 'adu'
    f = FrameData(a, unit=unit, meta=meta, mask=b, m_dtype='bool')
    npt.assert_array_almost_equal(a, f.data)
    npt.assert_array_almost_equal(b, f.mask)
    assert f.unit is u.adu
    assert np.issubdtype(f.mask.dtype, np.bool)
    assert f.meta['observer'] == meta['observer']
    assert f.meta['very long key'] == meta['very long key']
 

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
    assert f.unit is u.adu
    assert np.issubdtype(f.mask.dtype, np.uint8)
    assert f.meta['observer'] == meta['observer']
    assert f.meta['very long key'] == meta['very long key']


def test_framedata_empty():
    with pytest.raises(TypeError):
        FrameData()  # empty initializer should fail


def test_framedata_meta_header():
    header = DEFAULT_HEADER.copy()
    meta = {'testing1': 'a', 'testing2': 'b'}
    header.update({'testing1': 'c'})

    a = FrameData([1, 2, 3], unit='', meta=meta, header=header)
    assert type(a.meta) == dict
    assert type(a.header) == dict
    assert a.meta['testing1'] == 'c'  # Header priority
    assert a.meta['testing2'] == 'b'
    assert a.header['testing1'] == 'c'  # Header priority
    assert a.header['testing2'] == 'b'
    for k in DEFAULT_HEADER.keys():
        assert a.header[k] == DEFAULT_HEADER[k]
        assert a.meta[k] == DEFAULT_HEADER[k]


def test_frame_simple():
    framedata = create_framedata()
    assert framedata.shape == (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)
    assert framedata.size == DEFAULT_DATA_SIZE * DEFAULT_DATA_SIZE
    assert framedata.dtype is np.dtype(float)


def test_frame_init_with_string_electron_unit():
    framedata = FrameData(np.zeros([2, 2]), unit="electron")
    assert framedata.unit is u.electron


def test_framedata_meta_is_case_sensitive():
    ccd_data = create_framedata()
    key = 'SoMeKEY'
    lkey = key.lower()
    ukey = key.upper()
    ccd_data.meta[key] = 10
    assert lkey not in ccd_data.meta
    assert ukey not in ccd_data.meta
    assert key in ccd_data.meta


def test_metafromheader():
    hdr = fits.header.Header()
    hdr['observer'] = 'Edwin Hubble'
    hdr['exptime'] = '3600'

    d1 = FrameData(np.ones((5, 5)), meta=hdr, unit=u.electron)
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'
    assert d1.header['OBSERVER'] == 'Edwin Hubble'


def test_metafromdict():
    dic = {'OBSERVER': 'Edwin Hubble', 'EXPTIME': 3600}
    d1 = FrameData(np.ones((5, 5)), meta=dic, unit=u.electron)
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'


def test_header2meta():
    hdr = fits.header.Header()
    hdr['observer'] = 'Edwin Hubble'
    hdr['exptime'] = '3600'

    d1 = FrameData(np.ones((5, 5)), unit=u.electron)
    d1.header = hdr
    assert d1.meta['OBSERVER'] == 'Edwin Hubble'
    assert d1.header['OBSERVER'] == 'Edwin Hubble'


def test_metafromstring_fail():
    hdr = 'this is not a valid header'
    with pytest.raises(ValueError):
        FrameData(np.ones((5, 5)), meta=hdr, unit=u.adu)


def test_framedata_meta_is_not_fits_header():
    ccd_data = create_framedata()
    ccd_data.meta = {'OBSERVER': 'Edwin Hubble'}
    assert not isinstance(ccd_data.meta, fits.Header)


def test_setting_uncertainty_with_array():
    ccd_data = create_framedata()
    ccd_data.uncertainty = None
    fake_uncertainty = np.sqrt(np.abs(ccd_data.data))
    ccd_data.uncertainty = fake_uncertainty.copy()
    np.testing.assert_array_equal(ccd_data.uncertainty, fake_uncertainty)


def test_setting_uncertainty_with_scalar():
    uncertainty = 10
    ccd_data = create_framedata()
    ccd_data.uncertainty = None
    ccd_data.uncertainty = uncertainty
    fake_uncertainty = np.zeros_like(ccd_data.data)
    fake_uncertainty[:] = uncertainty
    np.testing.assert_array_equal(ccd_data.uncertainty, fake_uncertainty)


def test_setting_uncertainty_with_quantity():
    uncertainty = 10*u.electron
    ccd_data = create_framedata()
    ccd_data.uncertainty = None
    ccd_data.uncertainty = uncertainty
    fake_uncertainty = np.zeros_like(ccd_data.data)
    fake_uncertainty[:] = uncertainty.value
    np.testing.assert_array_equal(ccd_data.uncertainty, fake_uncertainty)
    assert ccd_data.uncert_unit is u.electron


def test_setting_uncertainty_wrong_shape_raises_error():
    ccd_data = create_framedata()
    with pytest.raises(ValueError):
        ccd_data.uncertainty = np.zeros([3, 4])


def test_to_hdu():
    ccd_data = create_framedata()
    ccd_data.meta = {'observer': 'Edwin Hubble'}
    fits_hdulist = ccd_data.to_hdu()
    assert isinstance(fits_hdulist, fits.HDUList)
    for k, v in ccd_data.meta.items():
        assert fits_hdulist[0].header[k] == v
    np.testing.assert_array_equal(fits_hdulist[0].data, ccd_data.data)


def test_initialize_from_FITS(tmpdir):
    frame = create_framedata()
    hdu = fits.PrimaryHDU(frame.data, header=fits.Header(frame.header))
    hdulist = fits.HDUList([hdu])
    filename = tmpdir.join('afile.fits').strpath
    hdulist.writeto(filename)
    cd = FrameData.read_fits(filename, unit=u.electron)
    assert cd.shape == (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)
    assert cd.size == DEFAULT_DATA_SIZE * DEFAULT_DATA_SIZE
    assert np.issubdtype(cd.data.dtype, np.floating)
    for k, v in hdu.header.items():
        assert cd.meta[k] == v


def test_initialize_from_FITS_memmap(tmpdir):
    frame = create_framedata()
    hdu = fits.PrimaryHDU(frame.data, header=fits.Header(frame.header))
    hdulist = fits.HDUList([hdu])
    filename = tmpdir.join('afile.fits').strpath
    hdulist.writeto(filename)
    # Same with memmap
    cd1 = FrameData.read_fits(filename, unit=u.electron,
                              use_memmap_backend=True)
    assert cd1.shape == (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)
    assert cd1.size == DEFAULT_DATA_SIZE * DEFAULT_DATA_SIZE
    assert np.issubdtype(cd1.data.dtype, np.floating)
    for k, v in hdu.header.items():
        assert cd1.meta[k] == v
    assert isinstance(cd1.data, np.memmap)


def test_initialize_from_fits_with_unit_in_header(tmpdir):
    fake_img = np.zeros([2, 2])
    hdu = fits.PrimaryHDU(fake_img)
    hdu.header['bunit'] = u.adu.to_string()
    filename = tmpdir.join('afile.fits').strpath
    hdu.writeto(filename)
    ccd = FrameData.read_fits(filename)
    # ccd should pick up the unit adu from the fits header...did it?
    assert ccd.unit is u.adu

    # An explicit unit in the read overrides any unit in the FITS file
    ccd2 = FrameData.read_fits(filename, unit="photon")
    assert ccd2.unit is u.photon


def test_initialize_from_fits_with_ADU_in_header(tmpdir):
    fake_img = np.zeros([2, 2])
    hdu = fits.PrimaryHDU(fake_img)
    hdu.header['bunit'] = 'ADU'
    filename = tmpdir.join('afile.fits').strpath
    hdu.writeto(filename)
    ccd = FrameData.read_fits(filename)
    # ccd should pick up the unit adu from the fits header...did it?
    assert ccd.unit is u.adu


def test_initialize_from_fits_with_invalid_unit_in_header(tmpdir):
    hdu = fits.PrimaryHDU(np.ones((2, 2)))
    hdu.header['bunit'] = 'definetely-not-a-unit'
    filename = tmpdir.join('afile.fits').strpath
    hdu.writeto(filename)
    with pytest.raises(ValueError):
        FrameData.read_fits(filename)


def test_initialize_from_fits_with_data_in_different_extension(tmpdir):
    fake_img = np.arange(4).reshape(2, 2)
    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU(fake_img)
    hdus = fits.HDUList([hdu1, hdu2])
    filename = tmpdir.join('afile.fits').strpath
    hdus.writeto(filename)
    with catch_warnings(FITSFixedWarning) as w:
        ccd = FrameData.read_fits(filename, unit='adu')
    assert len(w) == 0
    np.testing.assert_array_equal(ccd.data, fake_img)
    # FIXME: why?
    # assert hdu2.header + hdu1.header == ccd.header


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
    ccd_data = create_framedata()
    ccd_unit = ccd_data.unit
    hdulist = ccd_data.to_hdu()
    assert 'bunit' in hdulist[0].header
    assert hdulist[0].header['bunit'].strip() == ccd_unit.to_string()


def test_initialize_from_FITS_bad_keyword_raises_error(tmpdir):
    # There are two fits.open keywords that are not permitted in ccdproc:
    #     do_not_scale_image_data and scale_back
    ccd_data = create_framedata()
    filename = tmpdir.join('test.fits').strpath
    ccd_data.write_fits(filename)
    with pytest.raises(TypeError):
        FrameData.read_fits(filename, unit=ccd_data.unit,
                            do_not_scale_image_data=True)
    with pytest.raises(TypeError):
        FrameData.read_fits(filename, unit=ccd_data.unit, scale_back=True)


def test_framedata_writer(tmpdir):
    ccd_data = create_framedata()
    filename = tmpdir.join('test.fits').strpath
    ccd_data.write_fits(filename)
    ccd_disk = FrameData.read_fits(filename)
    np.testing.assert_array_equal(ccd_data.data, ccd_disk.data)


def test_fromMEF(tmpdir):
    ccd_data = create_framedata()
    hdu = ccd_data.to_hdu()[0]
    hdu2 = fits.PrimaryHDU(2 * ccd_data.data)
    hdulist = fits.HDUList(hdu)
    hdulist.append(hdu2)
    filename = tmpdir.join('afile.fits').strpath
    hdulist.writeto(filename)
    # by default, we reading from the first extension
    cd = FrameData.read_fits(filename, unit=u.electron)
    np.testing.assert_array_equal(cd.data, ccd_data.data)
    # but reading from the second should work too
    cd = FrameData.read_fits(filename, hdu=1, unit=u.electron)
    np.testing.assert_array_equal(cd.data, 2 * ccd_data.data)


def test_setting_bad_uncertainty_raises_error():
    ccd_data = create_framedata()
    with pytest.raises(TypeError):
        # Uncertainty is supposed to be an instance of NDUncertainty
        ccd_data.uncertainty = 'not a uncertainty'


def test_copy():
    ccd_data = create_framedata()
    ccd_copy = ccd_data.copy()
    np.testing.assert_array_equal(ccd_copy.data, ccd_data.data)
    assert ccd_copy.unit == ccd_data.unit
    assert ccd_copy.meta == ccd_data.meta


def test_wcs():
    ccd_data = create_framedata()
    ccd_data.wcs = 5
    assert ccd_data.wcs == 5