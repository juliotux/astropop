# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import os
import pytest
import numpy as np
from astropop.framedata import FrameData, PixelMaskFlags
from astropy.io import fits
from astropy.utils import NumpyRNGContext
from astropy import units as u
from astropy.wcs import WCS
from astropop.testing import *


DEFAULT_DATA_SIZE = 20
DEFAULT_HEADER = {'observer': 'astropop', 'very long key': 2}

with NumpyRNGContext(123):
    _random_array = np.random.normal(size=[DEFAULT_DATA_SIZE,
                                           DEFAULT_DATA_SIZE])
    _random_array = _random_array.astype(np.float64)


def create_framedata(**kwargs):
    data = _random_array.copy()
    fake_meta = DEFAULT_HEADER.copy()
    fake_meta.update(kwargs.pop('meta', {}))
    frame = FrameData(data, unit=u.Unit('adu'), meta=fake_meta, **kwargs)
    return frame


class TestFrameDataCreationMetas:
    def test_framedata_cration_array_with_meta(self):
        a = _random_array.copy()
        meta = DEFAULT_HEADER.copy()
        f = FrameData(a, meta=meta)
        assert_equal(f.history, [])
        assert_equal(f.comment, [])
        assert_is_none(f.wcs)
        assert_equal(f.meta['observer'], meta['observer'])
        assert_equal(f.meta['very long key'], meta['very long key'])

    def test_framedata_creation_array_meta_with_history(self):
        a = _random_array.copy()
        meta = DEFAULT_HEADER.copy()
        hist = ['test1', 'test2']
        meta['history'] = hist
        f = FrameData(a, meta=meta)
        assert_equal(f.history, hist)
        assert_equal(f.comment, [])
        assert_is_none(f.wcs)
        assert_equal(f.meta['observer'], meta['observer'])
        assert_equal(f.meta['very long key'], meta['very long key'])

    def test_framedata_creation_array_meta_with_comment(self):
        a = _random_array.copy()
        meta = DEFAULT_HEADER.copy()
        comment = ['test1', 'test2']
        meta['comment'] = comment
        f = FrameData(a, meta=meta)
        assert_equal(f.history, [])
        assert_equal(f.comment, comment)
        assert_is_none(f.wcs)
        assert_equal(f.meta['observer'], meta['observer'])
        assert_equal(f.meta['very long key'], meta['very long key'])

    def test_framedata_creation_array_meta_with_wcs(self):
        a = _random_array.copy()
        meta = DEFAULT_HEADER.copy()
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.crval = [10, 10]
        wcs.wcs.crpix = [10, 10]
        wcs.wcs.cdelt = [1, 1]
        meta.update(wcs.to_header())
        f = FrameData(a, meta=meta)
        assert_equal(f.history, [])
        assert_equal(f.comment, [])
        assert_is_instance(f.wcs, WCS)
        assert_equal(f.wcs.wcs.ctype, ['RA---TAN', 'DEC--TAN'])
        assert_equal(f.wcs.wcs.crval, [10, 10])
        assert_equal(f.wcs.wcs.crpix, [10, 10])
        assert_equal(f.wcs.wcs.cdelt, [1, 1])
        assert_equal(f.meta['observer'], meta['observer'])
        assert_equal(f.meta['very long key'], meta['very long key'])
        for i in wcs.to_header().keys():
            assert_not_in(i, f.meta)

    def test_framedata_creation_array_with_wcs_and_meta(self):
        a = _random_array.copy()
        meta = DEFAULT_HEADER.copy()
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.crval = [10, 10]
        wcs.wcs.crpix = [10, 10]
        wcs.wcs.cdelt = [1, 1]
        f = FrameData(a, meta=meta, wcs=wcs)
        assert_equal(f.history, [])
        assert_equal(f.comment, [])
        assert_is_instance(f.wcs, WCS)
        assert_equal(f.wcs.wcs.ctype, ['RA---TAN', 'DEC--TAN'])
        assert_equal(f.wcs.wcs.crval, [10, 10])
        assert_equal(f.wcs.wcs.crpix, [10, 10])
        assert_equal(f.wcs.wcs.cdelt, [1, 1])
        assert_equal(f.meta['observer'], meta['observer'])
        assert_equal(f.meta['very long key'], meta['very long key'])
        for i in wcs.to_header().keys():
            assert_not_in(i, f.meta)

    def test_framedata_creation_with_meta_wcs_conflict(self):
        a = _random_array.copy()
        meta = DEFAULT_HEADER.copy()
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.crval = [10, 10]
        wcs.wcs.crpix = [10, 10]
        wcs.wcs.cdelt = [1, 1]
        wcs2 = WCS(naxis=2)
        wcs2.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs2.wcs.crval = [20, 20]
        wcs2.wcs.crpix = [20, 20]
        wcs2.wcs.cdelt = [2, 2]
        meta.update(wcs2.to_header())
        with pytest.raises(ValueError, match='meta and wcs offer a WCS.'):
            f = FrameData(a, meta=meta, wcs=wcs)

    def test_frame_init_with_wcs(self):
        a = _random_array.copy()
        unit = 'adu'
        wcs = WCS(naxis=2)
        f = FrameData(a, unit=unit, wcs=wcs)
        # there is a bug if doing f.wcs == wcs
        assert_is_instance(f.wcs, WCS)
        for i in wcs.to_header().keys():
            assert_not_in(i, f.meta)
            assert_equal(f.wcs.to_header()[i], wcs.to_header()[i])

    def test_framedata_meta_header(self):
        header = DEFAULT_HEADER.copy()
        meta = {'testing1': 'a', 'testing2': 'b'}
        header.update({'testing1': 'c'})
        header = fits.Header(header)

        a = FrameData([[1], [2], [3]], unit='', meta=meta, header=header)
        assert_is_instance(a.meta, fits.Header)
        assert_is_instance(a.header, fits.Header)
        assert_is(a.meta, a.header)
        assert_equal(a.meta['testing1'], 'c')  # Header priority
        assert_equal(a.meta['testing2'], 'b')
        assert_equal(a.header['testing1'], 'c')  # Header priority
        assert_equal(a.header['testing2'], 'b')
        for k in DEFAULT_HEADER:
            assert_equal(a.header[k], DEFAULT_HEADER[k])
            assert_equal(a.meta[k], DEFAULT_HEADER[k])

    def test_framedata_meta_is_not_case_sensitive(self):
        frame = create_framedata()
        key = 'SoMeKEY'
        lkey = key.lower()
        ukey = key.upper()
        frame.meta[key] = 10
        assert_in(lkey, frame.meta)
        assert_in(ukey, frame.meta)
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
        d1.meta = hdr
        assert_equal(d1.meta['OBSERVER'], 'Edwin Hubble')
        assert_equal(d1.header['OBSERVER'], 'Edwin Hubble')

    def test_metafromstring_fail(self):
        hdr = 'this is not a valid header'
        with pytest.raises(TypeError):
            FrameData(np.ones((5, 5)), meta=hdr, unit=u.adu)

    def test_framedata_meta_history(self):
        frame = create_framedata(meta={'history': 'testing history'})
        assert_not_in('history', frame.meta)
        assert_equal(frame.history, ['testing history'])


class TestFrameDataCreationData:
    def test_framedata_empty(self):
        with pytest.raises(TypeError):
            # empty initializer should fail
            FrameData()

    def test_frame_simple(self):
        framedata = create_framedata()
        assert_equal(framedata.shape, (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE))
        assert_equal(framedata.size, DEFAULT_DATA_SIZE * DEFAULT_DATA_SIZE)
        assert_is(framedata.dtype, np.dtype(float))

    def test_framedata_cration_array(self):
        a = _random_array.copy()
        f = FrameData(a)
        assert_almost_equal(a, f.data)
        assert_equal(f.dtype, a.dtype)
        assert_equal(f.unit, u.dimensionless_unscaled)
        assert_equal(f.meta, {})
        assert_equal(f.history, [])
        assert_equal(f.comment, [])
        assert_equal(f.flags, np.zeros_like(a, dtype=np.uint8))
        assert_is_none(f.wcs)
        assert_is_none(f.uncertainty)

    def test_framedata_cration_array_with_unit(self):
        a = _random_array.copy()
        f = FrameData(a, unit=u.adu)
        assert_equal(f.unit, u.adu)
        assert_equal(f.meta, {})
        assert_equal(f.history, [])
        assert_equal(f.comment, [])
        assert_is_none(f.wcs)

    def test_framedata_cration_array_mask(self):
        a = _random_array.copy()
        b = np.zeros_like(_random_array, dtype=bool)
        b[1, 1] = True
        expected_flags = np.zeros_like(_random_array, dtype=np.uint8)
        expected_flags[1, 1] = (PixelMaskFlags.MASKED |
                                PixelMaskFlags.UNSPECIFIED).value
        unit = 'adu'
        f = FrameData(a, unit=unit, mask=b)
        assert_almost_equal(a, f.data)
        assert_equal(b, f.mask)
        assert_equal(expected_flags, f.flags)
        assert_equal(f.unit, u.adu)

    def test_framedata_cration_array_mask_and_flags(self):
        a = _random_array.copy()
        msk = np.zeros_like(_random_array, dtype=bool)
        msk[1, 1] = True
        flags = np.zeros_like(_random_array, dtype=np.uint8)
        flags[2, 2] = (PixelMaskFlags.SATURATED |
                       PixelMaskFlags.INTERPOLATED).value
        expected_flags = np.zeros_like(_random_array, dtype=np.uint8)
        expected_flags[1, 1] = (PixelMaskFlags.MASKED |
                                PixelMaskFlags.UNSPECIFIED).value
        expected_flags[2, 2] = (PixelMaskFlags.SATURATED |
                                PixelMaskFlags.INTERPOLATED).value

        unit = 'adu'
        f = FrameData(a, unit=unit, mask=msk, flags=flags)
        assert_almost_equal(a, f.data)
        assert_equal(msk, f.mask)
        assert_equal(expected_flags, f.flags)
        assert_equal(f.unit, u.adu)

    def test_frame_init_with_string_electron_unit(self):
        framedata = FrameData(np.zeros([2, 2]), unit="electron")
        assert_is(framedata.unit, u.electron)


class TestFrameDataMemMap:
    def test_framedata_memmap_default(self):
        # default is created with None unct and flags
        frame = create_framedata()
        assert_is_not_instance(frame._data, np.memmap)
        assert_is_none(frame._unct)
        assert_is_not_instance(frame._flags, np.memmap)
        assert_false(frame._memmapping)
        frame.enable_memmap()
        fname = os.path.join(frame.cache_folder, frame.cache_filename)
        assert_is_instance(frame._data, np.memmap)
        assert_is_none(frame._unct)
        assert_is_instance(frame._flags, np.memmap)
        assert_true(frame._memmapping)
        assert_equal(frame._data.filename, fname+'.data')
        assert_equal(frame._flags.filename, fname+'.flags')

    def test_framedata_memmap_setname(self, tmpdir):
        frame = create_framedata()
        c_folder = tmpdir.strpath
        c_file = 'test'
        frame.enable_memmap(filename=c_file, cache_folder=c_folder)
        fname = os.path.join(c_folder, c_file)
        assert_is_instance(frame._data, np.memmap)
        assert_is_none(frame._unct)
        assert_is_instance(frame._flags, np.memmap)
        assert_true(frame._memmapping)
        assert_equal(frame._data.filename, fname+'.data')
        assert_equal(frame._flags.filename, fname+'.flags')

    def test_framedata_disable_memmap(self):
        frame = create_framedata()
        assert_is_not_instance(frame._data, np.memmap)
        assert_is_none(frame._unct)
        assert_is_not_instance(frame._flags, np.memmap)
        assert_false(frame._memmapping)
        frame.enable_memmap()
        assert_is_instance(frame._data, np.memmap)
        assert_is_none(frame._unct)
        assert_is_instance(frame._flags, np.memmap)
        assert_true(frame._memmapping)
        frame.disable_memmap()
        assert_is_not_instance(frame._data, np.memmap)
        assert_is_none(frame._unct)
        assert_is_not_instance(frame._flags, np.memmap)
        assert_false(frame._memmapping)

    def test_framedata_memmap_with_uncertainty(self):
        frame = create_framedata()
        frame.uncertainty = np.ones_like(frame.data)
        assert_is_not_instance(frame._data, np.memmap)
        assert_is_not_instance(frame._unct, np.memmap)
        assert_is_not_instance(frame._flags, np.memmap)
        assert_false(frame._memmapping)
        frame.enable_memmap()
        fname = os.path.join(frame.cache_folder, frame.cache_filename)
        assert_is_instance(frame._data, np.memmap)
        assert_is_instance(frame._unct, np.memmap)
        assert_is_instance(frame._flags, np.memmap)
        assert_true(frame._memmapping)
        assert_equal(frame._data.filename, fname+'.data')
        assert_equal(frame._unct.filename, fname+'.unct')
        assert_equal(frame._flags.filename, fname+'.flags')

    def test_framedata_disable_memmap_with_uncertainty(self):
        frame = create_framedata()
        frame.uncertainty = np.ones_like(frame.data)
        assert_is_not_instance(frame._data, np.memmap)
        assert_is_not_instance(frame._unct, np.memmap)
        assert_is_not_instance(frame._flags, np.memmap)
        assert_false(frame._memmapping)
        frame.enable_memmap()
        assert_is_instance(frame._data, np.memmap)
        assert_is_instance(frame._unct, np.memmap)
        assert_is_instance(frame._flags, np.memmap)
        assert_true(frame._memmapping)
        frame.disable_memmap()
        assert_is_not_instance(frame._data, np.memmap)
        assert_is_not_instance(frame._unct, np.memmap)
        assert_is_not_instance(frame._flags, np.memmap)
        assert_false(frame._memmapping)
