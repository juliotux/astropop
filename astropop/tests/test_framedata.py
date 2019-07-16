# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Some parts stolen from Astropy CCDData testing bench

import pytest
import tempfile
import os
import numpy as np
import numpy.testing as npt
from astropop.framedata import FrameData, create_array_memmap, \
                               delete_array_memmap, ensure_bool_mask, \
                               setup_filename, framedata_read_fits, \
                               framedata_to_hdu
from astropy.io import fits
from astropy.utils import NumpyRNGContext
from astropy import units as u
from astropy.wcs import WCS, FITSFixedWarning
from astropy.tests.helper import catch_warnings


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
    frame = FrameData(data, unit=u.adu)
    frame.header = fake_meta
    return frame


def test_create_and_delete_memmap(tmpdir):
    # Creation
    f = os.path.join(tmpdir, 'testarray.npy')
    g = os.path.join(tmpdir, 'test2array.npy')
    a = np.ones((30, 30), dtype='f8')
    b = create_array_memmap(f, a)
    c = create_array_memmap(g, a, dtype=bool)
    assert isinstance(b, np.memmap)
    assert isinstance(c, np.memmap)
    npt.assert_array_equal(a, b)
    npt.assert_allclose(a, c)
    assert os.path.exists(f)
    assert os.path.exists(g)

    # Deletion
    # Since for the uses the object is overwritten, we do it here too
    b = delete_array_memmap(b)
    c = delete_array_memmap(c)
    assert not isinstance(b, np.memmap)
    assert not isinstance(b, np.memmap)
    assert isinstance(b, np.ndarray)
    assert isinstance(c, np.ndarray)
    npt.assert_array_equal(a, b)
    npt.assert_allclose(a, c)
    assert not os.path.exists(f)
    assert not os.path.exists(g)

    # None should not raise errors
    create_array_memmap('dummy', None)
    delete_array_memmap(None)


def test_ensure_bool_mask_bool(tmpdir):
    # Bool array
    b_array = np.zeros(2, dtype=bool)
    mask = ensure_bool_mask(b_array)
    assert np.dtype(mask.dtype) is np.dtype(bool)
    npt.assert_array_equal(mask, b_array)


def test_ensure_bool_mask_integer(tmpdir):
    # Integer array
    i_array = np.zeros(2, dtype='i4')
    mask = ensure_bool_mask(i_array)
    assert np.dtype(mask.dtype) is np.dtype(bool)
    npt.assert_array_almost_equal(mask, i_array)


def test_ensure_bool_mask_float(tmpdir):
    # Float array
    f_array = np.zeros(2, dtype='f8')
    mask = ensure_bool_mask(f_array)
    assert np.dtype(mask.dtype) is np.dtype(bool)
    npt.assert_array_almost_equal(mask, f_array)


@pytest.mark.skip("Bug in numpy. Fixed in 1.17.0")
def test_ensure_bool_mask_memmap(tmpdir):
    mmap = tmpdir.join('memmap.npy')
    filename = str(mmap)
    m_array = create_array_memmap(filename, np.zeros((10, 10)))
    mask = ensure_bool_mask(m_array)
    assert np.dtype(mask.dtype) is np.dtype(bool)
    npt.assert_array_almost_equal(mask, m_array)
    del m_array
    os.remove(filename)


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
    assert setup_filename(test_obj, filename=ntemp) == os.path.join(temp, fname)  # noqa
    test_obj.cache_filename = None
    assert setup_filename(test_obj, filename=ntemp) == os.path.join(temp, os.path.basename(ntemp))  # noqa
    # same for cache folder
    test_obj.cache_filename = fname
    assert setup_filename(test_obj, filename=ntemp) == os.path.join(temp, fname)  # noqa
    test_obj.cache_folder = None
    cache = '/tmp/astropop_testing'
    assert setup_filename(test_obj, cache_folder=cache) == os.path.join(cache, fname)  # noqa
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
    f = FrameData(a, unit=unit, meta=meta)
    npt.assert_equal(a, f.data)
    assert f.unit is u.adu
    assert f.meta['observer'] == meta['observer']
    assert f.meta['very long key'] == meta['very long key']


def test_framedata_empty():
    with pytest.raises(TypeError):
        FrameData()  # empty initializer should fail


def test_framedata_must_have_unit():
    with pytest.raises(ValueError):
        FrameData(np.zeros([2, 2]))


def test_framedata_unit_cannot_be_set_to_none():
    frame = create_framedata()
    with pytest.raises(TypeError):
        frame.unit = None


def test_framedata_meta_header_conflict():
    with pytest.raises(ValueError) as exc:
        FrameData([1, 2, 3], unit='', meta={1: 1}, header={2: 2})
        assert "can't have both header and meta." in str(exc.value)


def test_frame_simple():
    framedata = create_framedata()
    assert framedata.shape == (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)
    assert framedata.size == DEFAULT_DATA_SIZE * DEFAULT_DATA_SIZE
    assert framedata.dtype is np.dtype(float)


def test_frame_init_with_string_electron_unit():
    framedata = FrameData(np.zeros([2, 2]), unit="electron")
    assert framedata.unit is u.electron


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
    # why?
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


def test_framedata_meta_is_case_sensitive():
    ccd_data = create_framedata()
    key = 'SoMeKEY'
    lkey = key.lower()
    ukey = key.upper()
    ccd_data.meta[key] = 10
    assert lkey not in ccd_data.meta
    assert ukey not in ccd_data.meta
    assert key in ccd_data.meta


def test_ccddata_meta_is_not_fits_header():
    ccd_data = create_framedata()
    ccd_data.meta = {'OBSERVER': 'Edwin Hubble'}
    assert not isinstance(ccd_data.meta, fits.Header)


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


@pytest.mark.skip('To be implemented')
def test_setting_bad_uncertainty_raises_error():
    ccd_data = create_framedata()
    with pytest.raises(TypeError):
        # Uncertainty is supposed to be an instance of NDUncertainty
        ccd_data.uncertainty = 'not a uncertainty'


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


@pytest.mark.skip('To be implemented yet')
def test_copy():
    ccd_data = create_framedata()
    ccd_copy = ccd_data.copy()
    np.testing.assert_array_equal(ccd_copy.data, ccd_data.data)
    assert ccd_copy.unit == ccd_data.unit
    assert ccd_copy.meta == ccd_data.meta
