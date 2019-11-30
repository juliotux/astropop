# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Some parts stolen from Astropy CCDData testing bench

import pytest
import tempfile
import os
import numpy as np
import numpy.testing as npt
from astropop.framedata import FrameData, ensure_bool_mask, \
                               shape_consistency, unit_consistency, \
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
    frame = FrameData(data, unit=u.adu)
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
