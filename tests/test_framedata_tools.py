# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
import tempfile
import os
from astropop.framedata.framedata import setup_filename, extract_units, \
                                         shape_consistency, \
                                         uncertainty_unit_consistency, \
                                         FrameData
from astropy import units as u
from astropop.testing import *


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


class Test_FrameData_Shape_Consistency:
    shape = (12, 15)

    def test_all_ok(self):
        data = np.zeros(self.shape)
        unct = np.ones(self.shape)*0.1
        mask = np.zeros(self.shape)
        mask[1:2, 3:4] = 1
        flags = np.zeros(self.shape)
        d, u, m, f = shape_consistency(data, unct, mask, flags)
        assert_equal(d, data)
        assert_equal(u, unct)
        assert_equal(m, mask)
        assert_equal(f, flags)

    def test_only_unct(self):
        data = np.zeros(self.shape)
        unct = np.ones(self.shape)*0.1
        d, u, m, f = shape_consistency(data, unct)
        assert_equal(d, data)
        assert_equal(u, unct)
        assert_is_none(m)
        assert_is_none(f)

    def test_only_mask(self):
        data = np.zeros(self.shape)
        mask = np.zeros(self.shape)
        mask[1:2, 3:4] = 1
        d, u, m, f = shape_consistency(data, None, mask)
        assert_equal(d, data)
        assert_is_none(u)
        assert_equal(m, mask)
        assert_is_none(f)

    def test_no_data(self):
        # raises with uncertainty
        with pytest.raises(ValueError):
            shape_consistency(None, 1, None)
        # raises with mask
        with pytest.raises(ValueError):
            shape_consistency(None, None, 1)
        # raises with flags
        with pytest.raises(ValueError):
            shape_consistency(None, None, None, 1)

    def test_all_none(self):
        # all none must return all none
        d, u, m, f = shape_consistency()
        assert_is_none(d)
        assert_is_none(u)
        assert_is_none(m)
        assert_is_none(f)

    def test_single_value_uncert(self):
        data = np.zeros(self.shape)
        unct = 0.1
        d, u, m, f = shape_consistency(data, unct)
        assert_equal(d, data)
        assert_equal(u, np.ones(self.shape)*unct)
        assert_is_none(m)
        assert_is_none(f)

    def test_single_value_mask(self):
        # mask will not create a full array
        data = np.zeros(self.shape)
        with pytest.raises(ValueError):
            shape_consistency(data, None, False)

    def test_single_value_flags(self):
        # flags will not create a full array
        data = np.zeros(self.shape)
        with pytest.raises(ValueError):
            shape_consistency(data, None, None, 1)

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

    def test_wrong_shape_flags(self):
        data = np.zeros(self.shape)
        flags = np.ones((2, 2))
        with pytest.raises(ValueError):
            shape_consistency(data, flags=flags)


class Test_Uncertainty_Unit_Consitency:
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


class Test_FrameData_Setup_Filename:
    fname = 'test_filename.npy'

    def frame(self, path):
        temp = os.path.abspath(path)
        return FrameData(np.zeros((2, 2)), unit='adu',
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
