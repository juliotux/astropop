# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Some parts stolen from Astropy CCDData testing bench

import pytest
import tempfile
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
