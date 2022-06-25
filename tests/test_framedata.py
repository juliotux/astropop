# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import tempfile
import os
import numpy as np
from astropop.framedata.framedata import setup_filename, extract_units, \
                                         shape_consistency, \
                                         uncertainty_unit_consistency
from astropop.framedata import FrameData, check_framedata, read_framedata
from astropop.math import QFloat
from astropy.io import fits
from astropy.utils import NumpyRNGContext
from astropy import units as u
from astropy.wcs import WCS
from astropy.nddata import CCDData, StdDevUncertainty
from astropop.testing import *


DEFAULT_DATA_SIZE = 100
DEFAULT_HEADER = {'observer': 'astropop', 'very long key': 2}

with NumpyRNGContext(123):
    _random_array = np.random.normal(size=[DEFAULT_DATA_SIZE,
                                           DEFAULT_DATA_SIZE])
    _random_array = _random_array.astype(np.float64)


def create_framedata(**kwargs):
    data = _random_array.copy()
    fake_meta = DEFAULT_HEADER.copy()
    frame = FrameData(data, unit=u.Unit('adu'), **kwargs)
    frame.meta = fake_meta
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


class Test_FrameData_Shape_Consistency():
    shape = (12, 15)

    def test_all_ok(self):
        data = np.zeros(self.shape)
        unct = np.ones(self.shape)*0.1
        mask = np.zeros(self.shape)
        mask[1:2, 3:4] = 1
        d, u, m = shape_consistency(data, unct, mask)
        assert_equal(d, data)
        assert_equal(u, unct)
        assert_equal(m, mask)

    def test_only_unct(self):
        data = np.zeros(self.shape)
        unct = np.ones(self.shape)*0.1
        d, u, m = shape_consistency(data, unct, mask=None)
        assert_equal(d, data)
        assert_equal(u, unct)
        assert_is_none(m)

    def test_only_mask(self):
        data = np.zeros(self.shape)
        mask = np.zeros(self.shape)
        mask[1:2, 3:4] = 1
        d, u, m = shape_consistency(data, None, mask)
        assert_equal(d, data)
        assert_is_none(u)
        assert_equal(m, mask)

    def test_no_data(self):
        # raises with uncertainty
        with pytest.raises(ValueError):
            shape_consistency(None, 1, None)
        # raises with mask
        with pytest.raises(ValueError):
            shape_consistency(None, None, 1)

    def test_all_none(self):
        # all none must return all none
        d, u, m = shape_consistency()
        assert_is_none(d)
        assert_is_none(u)
        assert_is_none(m)

        # same for only mask=False
        d, u, m = shape_consistency(mask=False)
        assert_is_none(d)
        assert_is_none(u)
        assert_false(m)

    def test_single_value_uncert(self):
        data = np.zeros(self.shape)
        unct = 0.1
        d, u, m = shape_consistency(data, unct)
        assert_equal(d, data)
        assert_equal(u, np.ones(self.shape)*unct)
        assert_is_none(m)

    def test_single_value_mask(self):
        data = np.zeros(self.shape)
        d, u, m = shape_consistency(data, None, False)
        assert_equal(d, data)
        assert_is_none(u)
        assert_equal(m, np.zeros(self.shape, dtype=bool))

        d, u, m = shape_consistency(data, None, True)
        assert_equal(d, data)
        assert_is_none(u)
        assert_equal(m, np.ones(self.shape, dtype=bool))

    def test_wrong_shape_uncertainty(self):
        data = np.zeros(self.shape)
        unct = 0.1*np.ones((2, 2))
        with pytest.raises(ValueError):
            shape_consistency(data, unct)

    def test_wrong_shape_mask(self):
        data = np.zeros(self.shape)
        mask = np.ones((2, 2))
        with pytest.raises(ValueError):
            shape_consistency(data, None, mask)


class Test_Uncertainty_Unit_Consitency():
    def test_all_ok_quantity(self):
        unit = 'adu'
        unct = 0.1*u.Unit('adu')
        un = uncertainty_unit_consistency(unit, unct)
        assert_equal(un, np.array(0.1))

    def test_all_ok_number(self):
        # if uncertainty has no unit, it is returned
        unit = 'adu'
        unct = 0.1
        un = uncertainty_unit_consistency(unit, unct)
        assert_equal(un, np.array(0.1))

    def test_convert_unit(self):
        unit = 'm'
        unct = 1000*u.Unit('cm')
        un = uncertainty_unit_consistency(unit, unct)
        assert_equal(un, np.array(10))

    def test_incompatible_units(self):
        unit = 'm'
        unct = 1000*u.Unit('adu')
        with pytest.raises(u.UnitConversionError):
            uncertainty_unit_consistency(unit, unct)


class Test_FrameData_Setup_Filename():
    fname = 'test_filename.npy'

    def frame(self, path):
        temp = os.path.abspath(path)
        return FrameData(np.zeros(2), unit='adu',
                         cache_filename=self.fname,
                         cache_folder=temp)

    def test_not_framedata(self):
        with pytest.raises(ValueError):
            setup_filename(np.array(None))

    def test_simple(self, tmp_path):
        temp = str(tmp_path)
        frame = self.frame(temp)
        cache_file = os.path.join(temp, self.fname)
        assert_equal(setup_filename(frame), cache_file)

    def test_manual_filename_with_full_path(self, tmp_path):
        temp = str(tmp_path)
        frame = self.frame(temp)
        ntemp = tempfile.mkstemp(suffix='.npy', dir=temp)[1]
        assert_equal(setup_filename(frame, filename=ntemp), ntemp)

    def test_manual_filename_without_full_path(self, tmp_path):
        temp = str(tmp_path)
        frame = self.frame(temp)
        ntemp = 'testing.npy'
        cache_file = os.path.join(temp, ntemp)
        assert_equal(setup_filename(frame, filename=ntemp), cache_file)

    def test_manual_cache_folder_without_file(self, tmp_path):
        temp = str(tmp_path)
        frame = self.frame(temp)
        ntemp = os.path.dirname(tempfile.mkstemp(suffix='.npy')[1])
        cache_file = os.path.join(ntemp, self.fname)
        assert_equal(setup_filename(frame, cache_folder=ntemp), cache_file)

    def test_manual_folder_and_file(self, tmp_path):
        temp = str(tmp_path)
        frame = self.frame(temp)
        nfile = '/no-existing/testing.file.npy'
        ndir = os.path.dirname(tempfile.mkstemp(suffix='.npy')[1])
        cache_file = os.path.join(ndir, os.path.basename(nfile))
        assert_equal(setup_filename(frame, cache_folder=ndir, filename=nfile),
                     cache_file)


class Test_CheckRead_FrameData():
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
            assert_true(f._unct.empty)
            assert_false(np.any(f._mask))

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
            assert_true(f._unct.empty)
            assert_false(np.any(f._mask))

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

        fc = check_framedata(hdul)
        fr = read_framedata(hdul)
        for f in (fc, fr):
            assert_is_instance(f, FrameData)
            assert_equal(f.data, data)
            assert_equal(f.unit, 'adu')
            assert_equal(f.uncertainty, np.ones((DEFAULT_DATA_SIZE,
                                                 DEFAULT_DATA_SIZE)))
            assert_equal(f.mask, mask)

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
                assert_true(f._unct.empty)
                assert_false(np.any(f._mask))
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
                assert_true(f._unct.empty)
                assert_false(np.any(f._mask))
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
                assert_false(np.any(f._mask))
                assert_equal(f.meta, {})
                assert_is_none(f.wcs)
                assert_equal(f._memmapping, mmap)

    def test_check_framedata_invalid(self):
        # None should fail
        with pytest.raises(TypeError):
            check_framedata(None)
        with pytest.raises(TypeError):
            read_framedata(None)


class Test_FrameData_Copy():
    def test_copy_simple(self):
        frame = create_framedata()
        ccd_copy = frame.copy()
        assert_equal(ccd_copy.data, frame.data)
        assert_equal(ccd_copy.unit, frame.unit)
        assert_equal(ccd_copy.meta, frame.meta)
        assert_true(ccd_copy._unct.empty)
        assert_false(np.any(ccd_copy.mask))
        # tmp filenames must be created
        assert_is_not_none(ccd_copy.cache_filename)
        assert_is_not_none(ccd_copy.cache_folder)
        # origin must stay none
        assert_is_none(ccd_copy.origin_filename)

    def test_copy_with_incertainty(self):
        frame = create_framedata()
        frame.uncertainty = 1.0
        ccd_copy = frame.copy()
        assert_equal(ccd_copy.data, frame.data)
        assert_equal(ccd_copy.unit, frame.unit)
        assert_equal(ccd_copy.meta, frame.meta)
        assert_equal(ccd_copy.uncertainty, np.ones(frame.shape))

    def test_copy_history(self):
        frame = create_framedata()
        frame.history = 'frame copy tested by astropop'
        ccd_copy = frame.copy()
        assert_equal(ccd_copy.history, ['frame copy tested by astropop'])
        assert_not_in('history', ccd_copy.meta)
        assert_not_in('HISTORY', ccd_copy.meta)

    def test_copy_fnames(self, tmpdir):
        frame = create_framedata(cache_filename='testing',
                                 cache_folder=tmpdir.strpath,
                                 origin_filename='/dummy/dummy.dummy')
        ccd_copy = frame.copy()
        assert_equal(ccd_copy.cache_filename, 'testing_copy')
        assert_equal(ccd_copy.cache_folder, tmpdir.strpath)
        assert_equal(ccd_copy.origin_filename, '/dummy/dummy.dummy')

    def test_copy_wcs(self):
        wcs = WCS(naxis=2)
        frame = create_framedata(wcs=wcs)
        ccd_copy = frame.copy()
        assert_is_instance(ccd_copy.wcs, WCS)
        for i in wcs.to_header().keys():
            assert_not_in(i, ccd_copy.meta)
            assert_equal(ccd_copy.wcs.to_header()[i], wcs.to_header()[i])

    def test_copy_with_dtype(self):
        frame = create_framedata(uncertainty=0.5)  # default is float64
        f = frame.copy(np.float32)
        assert_is_not(f, frame)
        assert_almost_equal(f.data, frame.data)
        assert_equal(f.uncertainty, frame.uncertainty)
        assert_equal(f.data.dtype, np.float32)
        assert_equal(f.uncertainty.dtype, np.float32)

    def test_copy_astype(self):
        # copy using astype
        frame = create_framedata(uncertainty=0.5)  # default is float64
        f = frame.astype(np.float32)
        assert_is_not(f, frame)
        assert_almost_equal(f.data, frame.data)
        assert_equal(f.uncertainty, frame.uncertainty)
        assert_equal(f.data.dtype, np.float32)
        assert_equal(f.uncertainty.dtype, np.float32)

    def test_copy_filenames(self, tmpdir):
        tmp = tmpdir.strpath
        frame = create_framedata(cache_folder=tmp, cache_filename='testcopy')
        frame.uncertainty = 1.0
        f = frame.copy()
        f.enable_memmap()
        expect = os.path.join(tmp, 'testcopy'+'_copy')
        assert_path_exists(expect+'.data')
        assert_path_exists(expect+'.unct')
        assert_path_exists(expect+'.mask')


class Test_FrameData_History():
    def test_framedata_set_history(self):
        frame = create_framedata()
        frame.history = 'once upon a time'
        frame.history = 'a small fits file'
        frame.history = ['got read', 'by astropop']
        frame.history = ('and stay in', 'computer memory.')

        assert_equal(len(frame.history), 6)
        assert_equal(frame.history[0], 'once upon a time')
        assert_equal(frame.history[1], 'a small fits file')
        assert_equal(frame.history[2], 'got read')
        assert_equal(frame.history[3], 'by astropop')
        assert_equal(frame.history[4], 'and stay in')
        assert_equal(frame.history[5],  'computer memory.')


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
        f = FrameData(a, unit=unit, meta=meta, uncertainty=b,
                      u_dtype='float32')
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

    def test_frame_init_memmap(self):
        a = _random_array.copy()
        unit = 'adu'
        wcs = WCS(naxis=2)

        # default is not memmapped
        f = FrameData(a, unit=unit, wcs=wcs)
        assert_false(f._memmapping)
        assert_false(f._data.memmap)
        assert_false(f._unct.memmap)
        assert_false(f._mask.memmap)

        # with true, shuold be
        f = FrameData(a, unit=unit, wcs=wcs, use_memmap_backend=True)
        assert_true(f._memmapping)
        assert_true(f._data.memmap)
        assert_true(f._unct.memmap)
        assert_true(f._mask.memmap)

        # with explicit false, should not be
        f = FrameData(a, unit=unit, wcs=wcs, use_memmap_backend=False)
        assert_false(f._memmapping)
        assert_false(f._data.memmap)
        assert_false(f._unct.memmap)
        assert_false(f._mask.memmap)

    def test_init_with_masked_data(self):
        a = _random_array.copy()
        mask = np.zeros_like(a, dtype=bool)
        mask[1:2, 3:4] = 1
        a = np.ma.array(a, mask=mask)
        unit = 'adu'
        wcs = WCS(naxis=2)
        f = FrameData(a, unit=unit, wcs=wcs)
        assert_equal(f.mask, mask)

        # setting mask too
        mask2 = np.zeros_like(a, dtype=bool)
        mask2[3:4, 3:4] = 1
        expect = np.zeros_like(a, dtype=bool)
        expect[1:2, 3:4] = 1
        expect[3:4, 3:4] = 1
        f = FrameData(a, unit=unit, wcs=wcs, mask=mask2)
        assert_equal(f.mask, expect)


class Test_FrameData_WCS():
    def test_wcs_invalid(self):
        frame = create_framedata()
        with pytest.raises(TypeError):
            frame.wcs = 5

    def test_wcs_assign(self):
        wcs = WCS(naxis=2)
        frame = create_framedata()
        frame.wcs = wcs
        assert_equal(frame.wcs, wcs)


class Test_FrameData_Meta():
    def test_framedata_meta_header(self):
        header = DEFAULT_HEADER.copy()
        meta = {'testing1': 'a', 'testing2': 'b'}
        header.update({'testing1': 'c'})

        a = FrameData([1, 2, 3], unit='', meta=meta, header=header)
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

    def test_framedata_meta_is_fits_header(self):
        frame = create_framedata()
        frame.meta = {'OBSERVER': 'Edwin Hubble'}
        assert_is_instance(frame.meta, fits.Header)

    def test_framedata_meta_history(self):
        frame = create_framedata(meta={'history': 'testing history'})
        assert_not_in('history', frame.meta)
        assert_equal(frame.history, ['testing history'])

    def test_framedata_wcs_not_in_meta(self):
        wcs = dict(WCS(naxis=2).to_header())
        frame = create_framedata(meta=wcs)
        for i in wcs.keys():
            assert_not_in(i, frame.meta)

    def test_framedata_set_wcs_none(self):
        frame = create_framedata()
        frame.wcs = None
        assert_equal(frame.wcs, None)

    def test_framedata_set_wcs(self):
        frame = create_framedata()
        wcs = WCS(naxis=2)
        frame.wcs = wcs
        assert_equal(frame.wcs, wcs)

    def test_framedata_set_wcs_error(self):
        frame = create_framedata()
        with pytest.raises(TypeError,
                           match='wcs setter value must be a WCS instance.'):
            frame.wcs = 1


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

    def test_none_uncertainty_returns_empty(self):
        frame = create_framedata()
        assert_equal(frame.uncertainty, np.array(None))
        assert_true(frame.uncertainty.empty)

    def test_get_uncertainty_empty_return_none(self):
        # test the get_uncertainty method with return_none=True
        frame = create_framedata()
        assert_is_none(frame.get_uncertainty(True), None)

    def test_get_uncertainty_empty_return_zero(self):
        # test the get_uncertainty method with return_none=False
        frame = create_framedata()
        shp = (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)
        assert_equal(frame.get_uncertainty(False), np.zeros(shp))

    def test_get_uncertainty_non_empty_return_none(self):
        # test the get_uncertainty method with return_none=True
        shp = (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)
        frame = create_framedata(uncertainty=np.ones(shp))
        assert_equal(frame.get_uncertainty(True), np.ones(shp))

    def test_get_uncertainty_non_empty_return_zero(self):
        # test the get_uncertainty method with return_none=False
        shp = (DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)
        frame = create_framedata(uncertainty=np.ones(shp))
        assert_equal(frame.get_uncertainty(False), np.ones(shp))


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

    def test_write_unit_to_hdu(self):
        frame = create_framedata()
        ccd_unit = frame.unit
        hdulist = frame.to_hdu()
        assert_true('bunit' in hdulist[0].header)
        assert_equal(hdulist[0].header['bunit'].strip(), ccd_unit.to_string())


class Test_FrameData_CCDData():
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
        frame.mask = self.mask

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
        frame.mask = self.mask

        ccd = frame.to_ccddata()
        assert_equal(ccd.data, 100*np.ones(self.shape))
        assert_equal(ccd.unit, self.unit)
        for i in self.meta.keys():
            assert_equal(ccd.meta[i], self.meta[i])
        assert_equal(ccd.mask, self.mask)
        assert_equal(ccd.uncertainty.array, np.ones(self.shape))
        assert_equal(ccd.uncertainty.unit, self.unit)
        assert_equal(ccd.uncertainty.uncertainty_type, 'std')


class Test_FrameData_MemMap():
    def test_framedata_memmap_default(self):
        frame = create_framedata()
        assert_false(frame._data.memmap)
        assert_false(frame._unct.memmap)
        assert_false(frame._mask.memmap)
        assert_false(frame._memmapping)
        frame.enable_memmap()
        fname = os.path.join(frame.cache_folder, frame.cache_filename)
        assert_true(frame._data.memmap)
        assert_true(frame._unct.memmap)
        assert_true(frame._mask.memmap)
        assert_true(frame._memmapping)
        assert_equal(frame._data.filename, fname+'.data')
        assert_equal(frame._unct.filename, fname+'.unct')
        assert_equal(frame._mask.filename, fname+'.mask')

    def test_framedata_memmap_setname(self, tmpdir):
        frame = create_framedata()
        assert_false(frame._data.memmap)
        assert_false(frame._unct.memmap)
        assert_false(frame._mask.memmap)
        assert_false(frame._memmapping)
        c_folder = tmpdir.strpath
        c_file = 'test'
        frame.enable_memmap(filename=c_file, cache_folder=c_folder)
        fname = os.path.join(c_folder, c_file)
        assert_true(frame._data.memmap)
        assert_true(frame._unct.memmap)
        assert_true(frame._mask.memmap)
        assert_true(frame._memmapping)
        assert_equal(frame._data.filename, fname+'.data')
        assert_equal(frame._unct.filename, fname+'.unct')
        assert_equal(frame._mask.filename, fname+'.mask')

    def test_framedata_disable_memmap(self):
        frame = create_framedata()
        assert_false(frame._data.memmap)
        assert_false(frame._unct.memmap)
        assert_false(frame._mask.memmap)
        assert_false(frame._memmapping)
        frame.enable_memmap()
        assert_true(frame._data.memmap)
        assert_true(frame._unct.memmap)
        assert_true(frame._mask.memmap)
        assert_true(frame._memmapping)
        frame.disable_memmap()
        assert_false(frame._data.memmap)
        assert_false(frame._unct.memmap)
        assert_false(frame._mask.memmap)
        assert_false(frame._memmapping)


class Test_FrameData_GetSet():
    def test_set_data_no_unit(self):
        frame = create_framedata()
        frame.data = 1
        assert_equal(frame.data, 1)
        assert_equal(frame.unit, 'adu')

    def test_set_data_with_unit(self):
        frame = create_framedata()
        frame.data = 1*u.Unit('s')
        assert_equal(frame.data, 1)
        assert_equal(frame.unit, 's')


class Test_FrameData_MathProps():
    def test_framedata_median_without_unit(self):
        frame = FrameData([5, 1, 3, 4, 1])
        assert_equal(frame.median(), 3)

    def test_framedata_median_with_unit(self):
        frame = FrameData([5, 1, 3, 4, 1], unit='adu')
        assert_equal(frame.median(), 3*u.Unit('adu'))

    def test_framedata_mean_without_unit(self):
        frame = FrameData([2, 1, 3, 8, 6])
        assert_equal(frame.mean(), 4)

    def test_framedata_mean_with_unit(self):
        frame = FrameData([2, 1, 3, 8, 6], unit='adu')
        assert_equal(frame.mean(), 4*u.Unit('adu'))

    def test_framedata_min_without_unit(self):
        frame = FrameData([2, 1, 3, 8, 6])
        assert_equal(frame.min(), 1)

    def test_framedata_min_with_unit(self):
        frame = FrameData([2, 1, 3, 8, 6], unit='adu')
        assert_equal(frame.min(), 1*u.Unit('adu'))

    def test_framedata_max_without_unit(self):
        frame = FrameData([1, 2, 3, 8, 6])
        assert_equal(frame.max(), 8)

    def test_framedata_max_with_unit(self):
        frame = FrameData([1, 2, 3, 8, 6], unit='adu')
        assert_equal(frame.max(), 8*u.Unit('adu'))

    def test_framedata_std_without_unit(self):
        frame = FrameData(np.arange(10))
        res = frame.std()
        assert_almost_equal(res, 2.8722813232690143)

    def test_framedata_std_with_unit(self):
        frame = FrameData(np.arange(10), unit='adu')
        res = frame.std()
        assert_almost_equal(res.value, 2.8722813232690143)
        assert_equal(res.unit, u.adu)

    def test_framedata_statistics_without_unit(self):
        frame = FrameData(np.arange(9))
        res = frame.statistics()
        assert_equal(res['mean'], 4)
        assert_equal(res['median'], 4)
        assert_equal(res['min'], 0)
        assert_equal(res['max'], 8)
        assert_almost_equal(res['std'], 2.581988897471611)

    def test_framedata_statistics_with_unit(self):
        frame = FrameData(np.arange(9), unit='adu')
        res = frame.statistics()
        assert_equal(res['mean'].value, 4)
        assert_equal(res['median'].value, 4)
        assert_equal(res['min'].value, 0)
        assert_equal(res['max'].value, 8)
        assert_almost_equal(res['std'].value, 2.581988897471611)
        assert_equal(res['mean'].unit, u.adu)
        assert_equal(res['median'].unit, u.adu)
        assert_equal(res['min'].unit, u.adu)
        assert_equal(res['max'].unit, u.adu)
