# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import numpy as np
import pytest
from astropop.math.physical import QFloat, UnitsError, units
from packaging import version

from astropop.testing import *

# Testing qfloat compatibility with Numpy ufuncs and array functions.


class TestQFloatNumpyArrayFuncs:
    """Test numpy array functions for numpy comatibility."""

    def test_error_not_handled(self):
        # handled QFloat must be ok
        qf = QFloat([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], "m")
        res = np.sqrt(qf)

        # not handled QFloat must raise
        with pytest.raises(TypeError):
            np.frexp(qf)

    def test_error_only_call_method(self):
        qf = QFloat([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], "m")
        with pytest.raises(TypeError):
            np.sin.at(qf, 0)

    def test_qfloat_np_append(self):
        qf1 = QFloat([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], unit="m")
        qf2 = QFloat([1.0], [0.1], unit="km")
        qf3 = QFloat(1.0, 0.1, unit="km")
        qf4 = QFloat(0, 0)

        qf = np.append(qf1, qf1)
        assert_equal(qf.nominal, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        assert_equal(qf.std_dev, [0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
        assert_equal(qf.unit, qf1.unit)

        # This should work and convert the unit.
        qf = np.append(qf1, qf2)
        assert_equal(qf.nominal, [1.0, 2.0, 3.0, 1000.0])
        assert_equal(qf.std_dev, [0.1, 0.2, 0.3, 100.0])
        assert_equal(qf.unit, qf1.unit)

        # Also this should work and convert the unit in the same way.
        qf = np.append(qf1, qf3)
        assert_equal(qf.nominal, [1.0, 2.0, 3.0, 1000.0])
        assert_equal(qf.std_dev, [0.1, 0.2, 0.3, 100.0])
        assert_equal(qf.unit, qf1.unit)

        # This should fail due to unit
        with pytest.raises(UnitsError):
            qf = np.append(qf1, qf4)

        # Testing with axis
        qf1 = QFloat([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                     [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "m",)
        qf = np.append(qf1, QFloat([[8.0], [9.0]], [[0.8], [0.9]], "m"),
                       axis=1)
        assert_equal(qf.nominal, [[1.0, 2.0, 3.0, 8.0], [4.0, 5.0, 6.0, 9.0]])
        assert_equal(qf.std_dev, [[0.1, 0.2, 0.3, 0.8], [0.4, 0.5, 0.6, 0.9]])
        qf = np.append(qf1, QFloat([[7.0, 8.0, 9.0]], [[0.7, 0.8, 0.9]], "m"),
                       axis=0)
        assert_equal(qf.nominal, [[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0],
                                  [7.0, 8.0, 9.0]])
        assert_equal(qf.std_dev, [[0.1, 0.2, 0.3],
                                  [0.4, 0.5, 0.6],
                                  [0.7, 0.8, 0.9]])

    def test_qfloat_np_around(self):
        # single case
        qf = np.around(QFloat(1.02549, 0.135964))
        assert_equal(qf.nominal, 1)
        assert_equal(qf.std_dev, 0)

        qf = np.around(QFloat(1.02549, 0.135964), decimals=2)
        assert_equal(qf.nominal, 1.03)
        assert_equal(qf.std_dev, 0.14)

        # just check array too
        qf = np.around(QFloat([1.03256, 2.108645], [0.01456, 0.594324]),
                       decimals=2)
        assert_equal(qf.nominal, [1.03, 2.11])
        assert_equal(qf.std_dev, [0.01, 0.59])

    def test_qfloat_np_atleast_1d(self):
        # This function is not implemented, so should raise
        with pytest.raises(TypeError):
            np.atleast_1d(QFloat([1.0, 2.0], [0.1, 0.2], "m"))

    def test_qfloat_np_atleast_2d(self):
        # This function is not implemented, so should raise
        with pytest.raises(TypeError):
            np.atleast_2d(QFloat([1.0, 2.0], [0.1, 0.2], "m"))

    def test_qfloat_np_atleast_3d(self):
        # This function is not implemented, so should raise
        with pytest.raises(TypeError):
            np.atleast_3d(QFloat([1.0, 2.0], [0.1, 0.2], "m"))

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_broadcast(self):
        raise NotImplementedError

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_broadcast_to(self):
        raise NotImplementedError

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_ceil(self):
        raise NotImplementedError

    def test_qfloat_np_clip(self):
        arr = np.arange(10)
        qf = QFloat(arr, arr * 0.1, "m")

        res = np.clip(qf, 2, 8)
        tgt = [2, 2, 2, 3, 4, 5, 6, 7, 8, 8]
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, arr * 0.1)
        assert_equal(qf.unit, res.unit)

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_columnstack(self):
        raise NotImplementedError

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_concatenate(self):
        raise NotImplementedError

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_copyto(self):
        raise NotImplementedError

    def test_qfloat_np_copysign(self):
        arr = np.arange(10)
        qf = QFloat(arr, arr * 0.1, "m")

        res = np.copysign(qf, -1)
        assert_almost_equal(res.nominal, -arr)
        assert_almost_equal(res.std_dev, arr * 0.1)
        assert_equal(qf.unit, res.unit)

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_cross(self):
        raise NotImplementedError

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_cumprod(self):
        raise NotImplementedError

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_cumsum(self):
        raise NotImplementedError

    def test_qfloat_np_delete(self):
        a = np.array([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0]])
        qf = QFloat(a, a * 0.1, "m")
        res1 = np.delete(qf, 1, axis=0)
        assert_almost_equal(res1.nominal, [[1.0, 2.0, 3.0, 4.0],
                                           [9.0, 10.0, 11.0, 12.0]])
        assert_almost_equal(res1.std_dev, [[0.1, 0.2, 0.3, 0.4],
                                           [0.9, 1.0, 1.1, 1.2]])
        assert_equal(res1.unit, qf.unit)

        res2 = np.delete(qf, 1, axis=1)
        assert_almost_equal(res2.nominal, [[1.0, 3.0, 4.0],
                                           [5.0, 7.0, 8.0],
                                           [9.0, 11.0, 12.0]])
        assert_almost_equal(res2.std_dev, [[0.1, 0.3, 0.4],
                                           [0.5, 0.7, 0.8],
                                           [0.9, 1.1, 1.2]])
        assert_equal(res2.unit, qf.unit)

        res3 = np.delete(qf, np.s_[::2], 1)
        assert_almost_equal(res3.nominal,
                            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
        assert_almost_equal(res3.std_dev,
                            [[0.2, 0.4], [0.6, 0.8], [1.0, 1.2]])
        assert_equal(res3.unit, qf.unit)

        res4 = np.delete(qf, [1, 3, 5])
        assert_almost_equal(res4.nominal,
                            [1.0, 3.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        assert_almost_equal(res4.std_dev,
                            [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        assert_equal(res4.unit, qf.unit)

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_diff(self):
        raise NotImplementedError

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_dstack(self):
        raise NotImplementedError

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_ediff1d(self):
        raise NotImplementedError

    def test_qfloat_np_expand_dims(self):
        qf = QFloat(1.0, 0.1, "m")
        res1 = np.expand_dims(qf, axis=0)
        assert_almost_equal(res1.nominal, [1.0])
        assert_almost_equal(res1.std_dev, [0.1])
        assert_equal(res1.unit, qf.unit)
        assert_equal(res1.shape, (1,))

        qf = QFloat([1.0, 2.0], [0.1, 0.2], "m")
        res2 = np.expand_dims(qf, axis=0)
        assert_almost_equal(res2.nominal, [[1.0, 2.0]])
        assert_almost_equal(res2.std_dev, [[0.1, 0.2]])
        assert_equal(res2.unit, qf.unit)
        assert_equal(res2.shape, (1, 2))
        res3 = np.expand_dims(qf, axis=1)
        assert_almost_equal(res3.nominal, [[1.0], [2.0]])
        assert_almost_equal(res3.std_dev, [[0.1], [0.2]])
        assert_equal(res3.unit, qf.unit)
        assert_equal(res3.shape, (2, 1))

        if version.parse(np.version.full_version) >= version.parse('1.18.0'):
            res4 = np.expand_dims(qf, axis=(2, 0))
            assert_almost_equal(res4.nominal, [[[1.0], [2.0]]])
            assert_almost_equal(res4.std_dev, [[[0.1], [0.2]]])
            assert_equal(res4.unit, qf.unit)
            assert_equal(res4.shape, (1, 2, 1))

    def test_qfloat_np_flip(self):
        a = np.arange(8).reshape((2, 2, 2))
        qf = QFloat(a, a * 0.1, "m")

        res1 = np.flip(qf)
        assert_equal(res1.nominal, a[::-1, ::-1, ::-1])
        assert_equal(res1.std_dev, a[::-1, ::-1, ::-1] * 0.1)
        assert_equal(res1.unit, qf.unit)

        res2 = np.flip(qf, 0)
        assert_equal(res2.nominal, a[::-1, :, :])
        assert_equal(res2.std_dev, a[::-1, :, :] * 0.1)
        assert_equal(res2.unit, qf.unit)

        res3 = np.flip(qf, 1)
        assert_equal(res3.nominal, a[:, ::-1, :])
        assert_equal(res3.std_dev, a[:, ::-1, :] * 0.1)
        assert_equal(res3.unit, qf.unit)

        res4 = np.flip(qf, 2)
        assert_equal(res4.nominal, a[:, :, ::-1])
        assert_equal(res4.std_dev, a[:, :, ::-1] * 0.1)
        assert_equal(res4.unit, qf.unit)

        # just some static check
        qf = QFloat([[1, 2], [3, 4]], [[0.1, 0.2], [0.3, 0.4]], "m")

        res5 = np.flip(qf)
        assert_equal(res5.nominal, [[4, 3], [2, 1]])
        assert_equal(res5.std_dev, [[0.4, 0.3], [0.2, 0.1]])
        assert_equal(res5.unit, qf.unit)

        res6 = np.flip(qf, 0)
        assert_equal(res6.nominal, [[3, 4], [1, 2]])
        assert_equal(res6.std_dev, [[0.3, 0.4], [0.1, 0.2]])
        assert_equal(res6.unit, qf.unit)

        res7 = np.flip(qf, 1)
        assert_equal(res7.nominal, [[2, 1], [4, 3]])
        assert_equal(res7.std_dev, [[0.2, 0.1], [0.4, 0.3]])
        assert_equal(res7.unit, qf.unit)

    def test_qfloat_np_fliplr(self):
        a = np.arange(8).reshape((2, 2, 2))
        qf = QFloat(a, a * 0.1, "m")
        res = np.fliplr(qf)
        assert_equal(res.nominal, a[:, ::-1, :])
        assert_equal(res.std_dev, a[:, ::-1, :] * 0.1)
        assert_equal(res.unit, qf.unit)

        qf = QFloat([[1, 2], [3, 4]], [[0.1, 0.2], [0.3, 0.4]], "m")
        res = np.fliplr(qf)
        assert_equal(res.nominal, [[2, 1], [4, 3]])
        assert_equal(res.std_dev, [[0.2, 0.1], [0.4, 0.3]])
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_flipud(self):
        a = np.arange(8).reshape((2, 2, 2))
        qf = QFloat(a, a * 0.1, "m")
        res = np.flipud(qf)
        assert_equal(res.nominal, a[::-1, :, :])
        assert_equal(res.std_dev, a[::-1, :, :] * 0.1)
        assert_equal(res.unit, qf.unit)

        qf = QFloat([[1, 2], [3, 4]], [[0.1, 0.2], [0.3, 0.4]], "m")
        res = np.flipud(qf)
        assert_equal(res.nominal, [[3, 4], [1, 2]])
        assert_equal(res.std_dev, [[0.3, 0.4], [0.1, 0.2]])
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_insert(self):
        a = np.array([[1, 2], [3, 4], [5, 6]])
        qf = QFloat(a, a * 0.1, "m")

        res = np.insert(qf, 5, QFloat(999, 0.1, unit="m"))
        assert_almost_equal(res.nominal, [1, 2, 3, 4, 5, 999, 6])
        assert_almost_equal(res.std_dev, [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.6])
        assert_equal(res.unit, qf.unit)

        res = np.insert(qf, 1, QFloat(999, 0.1, unit="m"), axis=1)
        assert_almost_equal(res.nominal,
                            [[1, 999, 2], [3, 999, 4], [5, 999, 6]])
        assert_almost_equal(res.std_dev, [[0.1, 0.1, 0.2],
                                          [0.3, 0.1, 0.4],
                                          [0.5, 0.1, 0.6]])
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_mean(self):
        a = np.arange(8).reshape((2, 4))
        qf = QFloat(a, a * 0.1, "m")

        res = np.mean(qf)
        assert_almost_equal(res.nominal, np.mean(a))
        assert_almost_equal(res.std_dev, np.std(a)/np.sqrt(a.size))
        assert_equal(res.unit, qf.unit)

        res = np.mean(qf, axis=0)
        assert_almost_equal(res.nominal, np.mean(a, axis=0))
        assert_almost_equal(res.std_dev, np.std(a, axis=0)/np.sqrt(2))
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_median(self):
        a = np.arange(8).reshape((2, 4))
        qf = QFloat(a, a * 0.1, "m")

        res = np.median(qf)
        assert_almost_equal(res.nominal, np.median(a))
        assert_almost_equal(res.std_dev, np.std(a)/np.sqrt(a.size))
        assert_equal(res.unit, qf.unit)

        res = np.median(qf, axis=0)
        assert_almost_equal(res.nominal, np.median(a, axis=0))
        assert_almost_equal(res.std_dev, np.std(a, axis=0)/np.sqrt(2))
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_moveaxis(self):
        arr = np.zeros((3, 4, 5))
        qf = QFloat(arr, unit='m')

        res = np.moveaxis(qf, 0, -1)
        assert_equal(res.shape, (4, 5, 3))
        assert_equal(res.unit, qf.unit)

        res = np.moveaxis(qf, -1, 0)
        assert_equal(res.shape, (5, 3, 4))
        assert_equal(res.unit, qf.unit)

        res = np.moveaxis(qf, (0, 1), (-1, -2))
        assert_equal(res.shape, (5, 4, 3))
        assert_equal(res.unit, qf.unit)

        res = np.moveaxis(qf, [0, 1, 2], [-1, -2, -3])
        assert_equal(res.shape, (5, 4, 3))
        assert_equal(res.unit, qf.unit)

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_nancumprod(self):
        raise NotImplementedError

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_nancumsum(self):
        raise NotImplementedError

    def test_qfloat_np_nanmean(self):
        arr = np.array([1, 2, 1, np.nan, 1, 2, np.nan, 2])
        qf = QFloat(arr, uncertainty=arr*0.1, unit="m")
        res = np.nanmean(qf)
        assert_almost_equal(res.nominal, 1.5)
        assert_almost_equal(res.std_dev,
                            np.nanstd(qf.nominal)/np.sqrt(qf.size-2))

    def test_qfloat_np_nanmedian(self):
        arr = np.array([1, 2, 1, np.nan, 1, 2, np.nan, 2])
        qf = QFloat(arr, uncertainty=arr*0.1, unit="m")
        res = np.nanmedian(qf)
        assert_almost_equal(res.nominal, 1.5)
        assert_almost_equal(res.std_dev,
                            np.nanstd(qf.nominal)/np.sqrt(qf.size-2))

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_nanprod(self):
        raise NotImplementedError

    def test_qfoat_np_nanstd(self):
        arr = np.array([1, 2, 1, np.nan, 1, 2, np.nan, 2])
        qf = QFloat(arr, uncertainty=arr*0.1, unit="m")
        res = np.nanstd(qf)
        assert_almost_equal(res, np.nanstd(arr))

    def test_qfloat_np_nanstd(self):
        arr = np.array([1, 2, 1, np.nan, 1, 2, np.nan, 2])
        qf = QFloat(arr, uncertainty=arr*0.1, unit="m")
        res = np.nanstd(qf)
        assert_almost_equal(res, np.nanstd(arr))

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_nansum(self):
        raise NotImplementedError

    def test_qfloat_np_nanvar(self):
        arr = np.array([1, 2, 1, np.nan, 1, 2, np.nan, 2])
        qf = QFloat(arr, uncertainty=arr*0.1, unit="m")
        res = np.nanvar(qf)
        assert_almost_equal(res, np.nanvar(arr))

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_prod(self):
        raise NotImplementedError

    def test_qfloat_np_ravel(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        tgt = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        qf = QFloat(arr, arr * 0.1, "m")

        res = np.ravel(qf)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_repeat(self):
        arr = np.array([1, 2, 3])
        tgt = np.array([1, 1, 2, 2, 3, 3])
        qf = QFloat(arr, arr * 0.1, "m")

        res = np.repeat(qf, 2)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_reshape(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        tgt = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
        qf = QFloat(arr, arr * 0.1, "m")

        res = np.reshape(qf, (2, 6))
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)
        assert_equal(res.shape, (2, 6))

    def test_qfloat_np_resize(self):
        arr = np.array([[1, 2], [3, 4]])
        qf = QFloat(arr, arr * 0.1, "m")

        shp = (2, 4)
        tgt = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        res = np.resize(qf, shp)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)
        assert_equal(res.shape, shp)

        shp = (4, 2)
        tgt = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        res = np.resize(qf, shp)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)
        assert_equal(res.shape, shp)

        shp = (4, 3)
        tgt = np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]])
        res = np.resize(qf, shp)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)
        assert_equal(res.shape, shp)

        shp = (0,)
        # must fail due to empty array (not real)
        with pytest.raises(TypeError):
            res = np.resize(qf, shp)

    def test_qfloat_np_roll(self):
        arr = np.arange(10)
        qf = QFloat(arr, arr * 0.01, "m")

        off = 2
        tgt = np.array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
        res = np.roll(qf, off)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.01)
        assert_equal(res.unit, qf.unit)

        off = -2
        tgt = np.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
        res = np.roll(qf, off)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.01)
        assert_equal(res.unit, qf.unit)

        arr = np.arange(12).reshape((4, 3))
        qf = QFloat(arr, arr * 0.01, "m")

        ax = 0
        off = 1
        tgt = np.array([[9, 10, 11], [0, 1, 2], [3, 4, 5], [6, 7, 8]])
        res = np.roll(qf, off, axis=ax)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.01)
        assert_equal(res.unit, qf.unit)

        ax = 1
        off = 1
        tgt = np.array([[2, 0, 1], [5, 3, 4], [8, 6, 7], [11, 9, 10]])
        res = np.roll(qf, off, axis=ax)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.01)
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_rollaxis(self):
        arr = np.ones((3, 4, 5, 6))
        qf = QFloat(arr, arr * 0.01, "m")

        res = np.rollaxis(qf, 3, 1)
        assert_equal(res.shape, (3, 6, 4, 5))

        res = np.rollaxis(qf, 2)
        assert_equal(res.shape, (5, 3, 4, 6))

        res = np.rollaxis(qf, 1, 4)
        assert_equal(res.shape, (3, 5, 6, 4))

    def test_qfloat_np_round(self):
        # single case
        qf = np.round(QFloat(1.02549, 0.135964))
        assert_equal(qf.nominal, 1)
        assert_equal(qf.std_dev, 0)

        qf = np.round(QFloat(1.02549, 0.135964), decimals=2)
        assert_equal(qf.nominal, 1.03)
        assert_equal(qf.std_dev, 0.14)

        # just check array too
        qf = np.round(QFloat([1.03256, 2.108645], [0.01456, 0.594324]),
                      decimals=2)
        assert_equal(qf.nominal, [1.03, 2.11])
        assert_equal(qf.std_dev, [0.01, 0.59])

    def test_qfloat_np_rot90(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        b1 = np.array([[2, 5], [1, 4], [0, 3]])
        b2 = np.array([[5, 4, 3], [2, 1, 0]])
        b3 = np.array([[3, 0], [4, 1], [5, 2]])
        b4 = np.array([[0, 1, 2], [3, 4, 5]])
        qf = QFloat(arr, arr * 0.1, "m")

        for k in range(-3, 13, 4):
            res = np.rot90(qf, k=k)
            assert_equal(res.nominal, b1)
            assert_equal(res.std_dev, b1 * 0.1)
            assert_equal(res.unit, qf.unit)
        for k in range(-2, 13, 4):
            res = np.rot90(qf, k=k)
            assert_equal(res.nominal, b2)
            assert_equal(res.std_dev, b2 * 0.1)
            assert_equal(res.unit, qf.unit)
        for k in range(-1, 13, 4):
            res = np.rot90(qf, k=k)
            assert_equal(res.nominal, b3)
            assert_equal(res.std_dev, b3 * 0.1)
            assert_equal(res.unit, qf.unit)
        for k in range(0, 13, 4):
            res = np.rot90(qf, k=k)
            assert_equal(res.nominal, b4)
            assert_equal(res.std_dev, b4 * 0.1)
            assert_equal(res.unit, qf.unit)

        arr = np.arange(8).reshape((2, 2, 2))
        qf = QFloat(arr, arr * 0.1, "m")

        ax = (0, 1)
        tgt = np.array([[[2, 3], [6, 7]], [[0, 1], [4, 5]]])
        res = np.rot90(qf, axes=ax)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)

        ax = (1, 2)
        tgt = np.array([[[1, 3], [0, 2]], [[5, 7], [4, 6]]])
        res = np.rot90(qf, axes=ax)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)

        ax = (2, 0)
        tgt = np.array([[[4, 0], [6, 2]], [[5, 1], [7, 3]]])
        res = np.rot90(qf, axes=ax)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)

        ax = (1, 0)
        tgt = np.array([[[4, 5], [0, 1]], [[6, 7], [2, 3]]])
        res = np.rot90(qf, axes=ax)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_shape(self):
        for shp in [(10,), (11, 12), (11, 12, 13)]:
            qf = QFloat(np.ones(shp), np.ones(shp), "m")
            assert_equal(np.shape(qf), shp)

    def test_qfloat_np_size(self):
        for shp in [(10,), (11, 12), (11, 12, 13)]:
            qf = QFloat(np.ones(shp), np.ones(shp), "m")
            assert_equal(np.size(qf), np.prod(shp))

    def test_qfloat_np_squeeze(self):
        arr = np.array([[[0], [1], [2]]])
        qf = QFloat(arr, arr * 0.01, "m")

        res = np.squeeze(qf)
        assert_equal(res.shape, (3,))
        assert_almost_equal(res.nominal, [0, 1, 2])
        assert_almost_equal(res.std_dev, [0, 0.01, 0.02])
        assert_equal(res.unit, qf.unit)

        res = np.squeeze(qf, axis=0)
        assert_equal(res.shape, (3, 1))
        assert_almost_equal(res.nominal, [[0], [1], [2]])
        assert_almost_equal(res.std_dev, [[0], [0.01], [0.02]])
        assert_equal(res.unit, qf.unit)

        with pytest.raises(ValueError):
            np.squeeze(qf, axis=1)

        res = np.squeeze(qf, axis=2)
        assert_equal(res.shape, (1, 3))
        assert_almost_equal(res.nominal, [[0, 1, 2]])
        assert_almost_equal(res.std_dev, [[0, 0.01, 0.02]])
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_std(self):
        qf = QFloat(np.arange(10), uncertainty=np.arange(10)*0.1)
        assert_almost_equal(np.std(qf), 2.87228, decimal=4)

        # errors do not enter in the account
        qf = QFloat(np.arange(10), uncertainty=np.arange(10))
        assert_almost_equal(np.std(qf), 2.87228, decimal=4)

    def test_qfloat_np_sum(self):
        arr = np.ones(10).reshape((2, 5))
        qf = QFloat(arr, arr*0.1, "m")

        res = np.sum(qf)
        assert_equal(res.nominal, 10)
        assert_equal(res.std_dev, 0.1*np.sqrt(10))
        assert_equal(res.unit, qf.unit)

        res = np.sum(qf, axis=0)
        assert_equal(res.shape, [5])
        assert_almost_equal(res.nominal, np.ones(5)*2)
        assert_almost_equal(res.std_dev, np.ones(5)*np.sqrt(2)*0.1)
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_swapaxes(self):
        arr = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        tgt = np.array([[[0, 4], [2, 6]], [[1, 5], [3, 7]]])
        qf = QFloat(arr, arr * 0.1, "m")

        res = np.swapaxes(qf, 0, 2)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_take(self):
        arr = np.array([1, 2, 3, 4, 5])
        tgt = np.array([2, 3, 5])
        ind = [1, 2, 4]
        qf = QFloat(arr, arr * 0.1, "m")

        res = np.take(qf, ind)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_tile(self):
        arr = np.array([0, 1, 2])
        qf = QFloat(arr, arr * 0.1)

        tile = 2
        tgt = np.array([0, 1, 2, 0, 1, 2])
        res = np.tile(qf, tile)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)

        tile = (2, 2)
        tgt = np.array([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
        res = np.tile(qf, tile)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)

        # More checking
        arr = np.array([[1, 2], [3, 4]])
        qf = QFloat(arr, arr * 0.1)

        tile = 2
        tgt = np.array([[1, 2, 1, 2], [3, 4, 3, 4]])
        res = np.tile(qf, tile)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)

        tile = (2, 1)
        tgt = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        res = np.tile(qf, tile)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)

    def test_qfloat_np_transpose(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        tgt = np.array([[1, 3, 5], [2, 4, 6]])
        qf = QFloat(arr, arr * 0.1, "m")

        res = np.transpose(qf)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_trunc(self):
        raise NotImplementedError

    def test_qfloat_np_var(self):
        qf = QFloat(np.arange(10), uncertainty=np.arange(10)*0.1)
        assert_almost_equal(np.var(qf), 8.25)

        # errors do not enter in the account
        qf = QFloat(np.arange(10), uncertainty=np.arange(10))
        assert_almost_equal(np.var(qf), 8.25)


class TestQFloatNumpyUfuncs:
    """Test numpy array functions for numpy comatibility."""

    @pytest.mark.parametrize('func', [np.abs, np.absolute])
    def test_qfloat_np_absolute(self, func):
        qf1 = QFloat(1.0, 0.1, 'm')
        qf2 = QFloat(-1.0, 0.1, 'm')
        qf3 = QFloat(-5.0, 0.1)
        qf4 = QFloat(-6)
        qf5 = QFloat([1, -1, 2, -2])

        assert_equal(func(qf1), QFloat(1.0, 0.1, 'm'))
        assert_equal(func(qf2), QFloat(1.0, 0.1, 'm'))
        assert_equal(func(qf3), QFloat(5.0, 0.1))
        assert_equal(func(qf4), QFloat(6))
        assert_equal(func(qf5), [1, 1, 2, 2])

        with pytest.raises(NotImplementedError):
            # out argument should fail
            func(qf1, out=[])

    def test_qfloat_np_add(self):
        qf1 = QFloat(2.0, 0.2, 'm')
        qf2 = QFloat(1.0, 0.1, 'm')
        qf3 = QFloat([1, 2, 3], [0.1, 0.2, 0.3], 'm')
        qf4 = QFloat(1.0, 0.1, 's')
        qf5 = QFloat(1.0)

        res = np.add(qf1, qf2)
        assert_equal(res.nominal, 3.0)
        assert_almost_equal(res.std_dev, 0.223606797749979)
        assert_equal(res.unit, units.Unit('m'))

        res = np.add(qf1, qf3)
        assert_equal(res.nominal, [3, 4, 5])
        assert_almost_equal(res.std_dev, [0.2236068, 0.28284271, 0.36055513])
        assert_equal(res.unit, units.Unit('m'))

        res = np.add(qf3, qf1)
        assert_equal(res.nominal, [3, 4, 5])
        assert_almost_equal(res.std_dev, [0.2236068, 0.28284271, 0.36055513])
        assert_equal(res.unit, units.Unit('m'))

        with pytest.raises(UnitsError):
            np.add(qf1, qf4)

        with pytest.raises(UnitsError):
            np.add(qf1, qf5)

        with pytest.raises(UnitsError):
            np.add(qf1, 1.0)

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.add(qf1, qf2, out=[])

    def test_qfloat_np_ceil(self):
        assert_equal(np.ceil(QFloat(1.5, 0.1, 'm')), QFloat(2.0, 0.0, 'm'))
        assert_equal(np.ceil(QFloat(-1.5, 0.1, 'm')), QFloat(-1.0, 0.0, 'm'))
        assert_equal(np.ceil(QFloat(0.2, 0.1, 'm')), QFloat(1.0, 0.0, 'm'))
        assert_equal(np.ceil(QFloat(-0.2, 0.1, 'm')), QFloat(0.0, 0.0, 'm'))

    @pytest.mark.parametrize('func', [np.divide, np.true_divide])
    def test_qfloat_np_divide(self, func):
        qf1 = QFloat(2.0, 0.2, 'm')
        qf2 = QFloat(1.0, 0.1, 'm')
        qf3 = QFloat([1, 2, 4], [0.1, 0.2, 0.4], 'cm')
        qf4 = QFloat(1.0, 0.1, 's')
        qf5 = QFloat(1.0)

        res = func(qf1, qf2)
        assert_equal(res.nominal, 2)
        assert_almost_equal(res.std_dev, 0.28284271)
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = func(qf1, qf3)
        assert_equal(res.nominal, [2, 1, 0.5])
        assert_almost_equal(res.std_dev, [0.28284271, 0.14142136, 0.07071068])
        assert_equal(res.unit, units.Unit('m/cm'))

        res = func(qf3, qf1)
        assert_equal(res.nominal, [0.5, 1, 2])
        assert_almost_equal(res.std_dev, [0.0707107, 0.1414214, 0.2828427])
        assert_equal(res.unit, units.Unit('cm/m'))

        res = func(qf1, qf4)
        assert_equal(res.nominal, 2.0)
        assert_almost_equal(res.std_dev, 0.28284271247461906)
        assert_equal(res.unit, units.Unit('m/s'))

        res = func(qf1, qf5)
        assert_equal(res.nominal, 2.0)
        assert_almost_equal(res.std_dev, 0.2)
        assert_equal(res.unit, units.Unit('m'))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            func(qf1, qf2, out=[])

    def test_qfloat_np_divmod(self):
        qf1 = QFloat(2.0, 0.2, 'm')
        qf2 = QFloat(1.0, 0.1, 'm')
        qf3 = QFloat([1, 2, 4], [0.1, 0.2, 0.4], 'cm')
        qf4 = QFloat(1.0, 0.1, 's')

        res = np.divmod(qf1, qf2)
        assert_equal(res[0], np.floor_divide(qf1, qf2))
        assert_equal(res[1], np.remainder(qf1, qf2))

        res = np.divmod(qf1, qf3)
        assert_equal(res[0], np.floor_divide(qf1, qf3))
        assert_equal(res[1], np.remainder(qf1, qf3))

        res = np.divmod(qf3, qf1)
        assert_equal(res[0], np.floor_divide(qf3, qf1))
        assert_equal(res[1], np.remainder(qf3, qf1))

        res = np.divmod(qf1, qf4)
        assert_equal(res[0], np.floor_divide(qf1, qf4))
        assert_equal(res[1], np.remainder(qf1, qf4))

    def test_qfloat_np_exp(self):
        qf1 = QFloat(2.0, 0.2)
        qf2 = QFloat(1.0, 0.1)
        qf3 = QFloat([1, 2, 4], [0.1, 0.2, 0.4])
        qf4 = QFloat(1.0, 0.1, 's')
        qf5 = QFloat(1.0)

        res = np.exp(qf1)
        assert_equal(res.nominal, np.exp(2.0))
        assert_almost_equal(res.std_dev, 0.2*np.exp(2.0))
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.exp(qf2)
        assert_equal(res.nominal, np.exp(1.0))
        assert_almost_equal(res.std_dev, 0.1*np.exp(1.0))
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.exp(qf3)
        assert_equal(res.nominal, np.exp(qf3.nominal))
        assert_almost_equal(res.std_dev, np.exp(qf3.nominal)*qf3.std_dev)
        assert_equal(res.unit, units.dimensionless_unscaled)

        with pytest.raises(UnitsError, match='log is only defined for '
                           'dimensionless quantities.'):
            res = np.log(qf4)

        res = np.exp(qf5)
        assert_equal(res.nominal, np.exp(1.0))
        assert_almost_equal(res.std_dev, 0.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

    def test_qfloat_np_exp2(self):
        qf1 = QFloat(2.0, 0.2)
        qf2 = QFloat(1.0, 0.1)
        qf3 = QFloat([1, 2, 4], [0.1, 0.2, 0.4])
        qf4 = QFloat(1.0, 0.1, 's')
        qf5 = QFloat(1.0)

        res = np.exp2(qf1)
        assert_equal(res.nominal, np.exp2(2.0))
        assert_almost_equal(res.std_dev, 0.2*np.exp2(2.0)*np.log(2))
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.exp2(qf2)
        assert_equal(res.nominal, np.exp2(1.0))
        assert_almost_equal(res.std_dev, 0.1*np.exp2(1.0)*np.log(2))
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.exp2(qf3)
        assert_equal(res.nominal, np.exp2(qf3.nominal))
        assert_almost_equal(res.std_dev,
                            np.exp2(qf3.nominal)*qf3.std_dev*np.log(2))
        assert_equal(res.unit, units.dimensionless_unscaled)

        with pytest.raises(UnitsError, match='exp2 is only defined for '
                           'dimensionless quantities.'):
            res = np.exp2(qf4)

        res = np.exp2(qf5)
        assert_equal(res.nominal, np.exp2(1.0))
        assert_almost_equal(res.std_dev, 0.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

    def test_qfloat_np_expm1(self):
        qf1 = QFloat(2.0, 0.2)
        qf2 = QFloat(1.0, 0.1)
        qf3 = QFloat([1, 2, 4], [0.1, 0.2, 0.4])
        qf4 = QFloat(1.0, 0.1, 's')
        qf5 = QFloat(1.0)

        res = np.expm1(qf1)
        assert_equal(res.nominal, np.expm1(2.0))
        assert_almost_equal(res.std_dev, 0.2*np.exp(2.0))
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.expm1(qf2)
        assert_equal(res.nominal, np.expm1(1.0))
        assert_almost_equal(res.std_dev, 0.1*np.exp(1.0))
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.expm1(qf3)
        assert_equal(res.nominal, np.expm1(qf3.nominal))
        assert_almost_equal(res.std_dev, np.exp(qf3.nominal)*qf3.std_dev)
        assert_equal(res.unit, units.dimensionless_unscaled)

        with pytest.raises(UnitsError, match='log is only defined for '
                           'dimensionless quantities.'):
            res = np.log(qf4)

        res = np.expm1(qf5)
        assert_equal(res.nominal, np.expm1(1.0))
        assert_almost_equal(res.std_dev, 0.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

    def test_qfloat_np_floor(self):
        assert_equal(np.floor(QFloat(1.5, 0.1, 'm')), QFloat(1.0, 0.0, 'm'))
        assert_equal(np.floor(QFloat(-1.5, 0.1, 'm')), QFloat(-2.0, 0.0, 'm'))
        assert_equal(np.floor(QFloat(0.2, 0.1, 'm')), QFloat(0.0, 0.0, 'm'))
        assert_equal(np.floor(QFloat(-0.2, 0.1, 'm')), QFloat(-1.0, 0.0, 'm'))

    def test_qfloat_np_floor_divide(self):
        qf1 = QFloat(2.0, 0.2, 'm')
        qf2 = QFloat(1.0, 0.1, 'm')
        qf3 = QFloat([1, 2, 4], [0.1, 0.2, 0.4], 'cm')
        qf4 = QFloat(1.0, 0.1, 's')
        qf5 = QFloat(1.0)

        res = np.floor_divide(qf1, qf2)
        assert_equal(res.nominal, 2)
        assert_almost_equal(res.std_dev, 0)
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.floor_divide(qf1, qf3)
        assert_equal(res.nominal, [2, 1, 0])
        assert_almost_equal(res.std_dev, [0, 0, 0])
        assert_equal(res.unit, units.Unit('m/cm'))

        res = np.floor_divide(qf3, qf1)
        assert_equal(res.nominal, [0, 1, 2])
        assert_almost_equal(res.std_dev, [0, 0, 0])
        assert_equal(res.unit, units.Unit('cm/m'))

        res = np.floor_divide(qf1, qf4)
        assert_equal(res.nominal, 2)
        assert_almost_equal(res.std_dev, 0)
        assert_equal(res.unit, units.Unit('m/s'))

        res = np.floor_divide(qf1, qf5)
        assert_equal(res.nominal, 2)
        assert_almost_equal(res.std_dev, 0)
        assert_equal(res.unit, units.Unit('m'))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.floor_divide(qf1, qf2, out=[])

    def test_qfloat_np_hypot(self):
        qf1 = QFloat(3, 0.3, 'm')
        qf2 = QFloat(4, 0.4, 'm')
        qf3 = QFloat(3*np.ones((5, 5)), unit='m')
        qf4 = QFloat(4*np.ones((5, 5)), unit='m')

        res = np.hypot(qf1, qf2)
        assert_equal(res.nominal, 5)
        assert_almost_equal(res.std_dev, 0.36715119501371646)
        assert_equal(res.unit, units.Unit('m'))

        res = np.hypot(qf3, qf4)
        assert_equal(res.nominal, 5*np.ones((5, 5)))
        assert_almost_equal(res.std_dev, np.zeros((5, 5)))
        assert_equal(res.unit, units.Unit('m'))

        res = np.hypot(qf1, qf4)
        assert_equal(res.nominal, 5*np.ones((5, 5)))
        assert_almost_equal(res.std_dev, 0.18*np.ones((5, 5)))
        assert_equal(res.unit, units.Unit('m'))

        with pytest.raises(UnitsError):
            np.hypot(qf1, 1)

        with pytest.raises(UnitsError):
            np.hypot(qf1, QFloat(1, unit='s'))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.multiply(qf1, qf2, out=[])

    def test_qfloat_np_isfinit(self):
        assert_false(np.isfinite(QFloat(np.inf)))
        assert_false(np.isfinite(QFloat(np.nan)))
        assert_true(np.isfinite(QFloat(1)))
        assert_true(np.isfinite(QFloat(1, unit='m')))
        assert_true(np.isfinite(QFloat(1, 0.1, unit='m/s')))

        assert_equal(np.isfinite(QFloat([np.inf, np.nan, 1.0], unit='m')),
                     [False, False, True])

    def test_qfloat_np_isinf(self):
        assert_true(np.isinf(QFloat(np.inf)))
        assert_false(np.isinf(QFloat(np.nan)))
        assert_false(np.isinf(QFloat(1)))
        assert_false(np.isinf(QFloat(1, unit='m')))
        assert_false(np.isinf(QFloat(1, 0.1, unit='m/s')))

        assert_equal(np.isinf(QFloat([np.inf, np.nan, 1.0], unit='m')),
                     [True, False, False])

    def test_qfloat_np_isnan(self):
        assert_false(np.isnan(QFloat(np.inf)))
        assert_true(np.isnan(QFloat(np.nan)))
        assert_false(np.isnan(QFloat(1)))
        assert_false(np.isnan(QFloat(1, unit='m')))
        assert_false(np.isnan(QFloat(1, 0.1, unit='m/s')))

        assert_equal(np.isnan(QFloat([np.inf, np.nan, 1.0], unit='m')),
                     [False, True, False])

    def test_qfloat_np_log(self):
        qf1 = QFloat(2.0, 0.2)
        qf2 = QFloat(1.0, 0.1)
        qf3 = QFloat([1, 2, 4], [0.1, 0.2, 0.4])
        qf4 = QFloat(1.0, 0.1, 's')
        qf5 = QFloat(1.0)

        res = np.log(qf1)
        assert_equal(res.nominal, np.log(2.0))
        assert_almost_equal(res.std_dev, 0.2/2.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.log(qf2)
        assert_equal(res.nominal, np.log(1.0))
        assert_almost_equal(res.std_dev, 0.1/1.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.log(qf3)
        assert_equal(res.nominal, np.log([1, 2, 4]))
        assert_almost_equal(res.std_dev,
                            qf3.std_dev/qf3.nominal)
        assert_equal(res.unit, units.dimensionless_unscaled)

        with pytest.raises(UnitsError, match='log is only defined for '
                           'dimensionless quantities.'):
            res = np.log(qf4)

        res = np.log(qf5)
        assert_equal(res.nominal, np.log(1.0))
        assert_almost_equal(res.std_dev, 0.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

    def test_qfloat_np_log2(self):
        qf1 = QFloat(2.0, 0.2)
        qf2 = QFloat(1.0, 0.1)
        qf3 = QFloat([1, 2, 4], [0.1, 0.2, 0.4])
        qf4 = QFloat(1.0, 0.1, 's')
        qf5 = QFloat(1.0)

        res = np.log2(qf1)
        assert_equal(res.nominal, np.log2(2.0))
        assert_almost_equal(res.std_dev, 0.2/(2.0*np.log(2)))
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.log2(qf2)
        assert_equal(res.nominal, np.log2(1.0))
        assert_almost_equal(res.std_dev, 0.1/np.log(2))
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.log2(qf3)
        assert_equal(res.nominal, np.log2([1, 2, 4]))
        assert_almost_equal(res.std_dev,
                            qf3.std_dev/(qf3.nominal*np.log(2)))
        assert_equal(res.unit, units.dimensionless_unscaled)

        with pytest.raises(UnitsError, match='log2 is only defined for '
                           'dimensionless quantities.'):
            res = np.log2(qf4)

        res = np.log2(qf5)
        assert_equal(res.nominal, np.log2(1.0))
        assert_almost_equal(res.std_dev, 0.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

    def test_qfloat_np_log10(self):
        qf1 = QFloat(2.0, 0.2)
        qf2 = QFloat(1.0, 0.1)
        qf3 = QFloat([1, 2, 4], [0.1, 0.2, 0.4])
        qf4 = QFloat(1.0, 0.1, 's')
        qf5 = QFloat(1.0)

        res = np.log10(qf1)
        assert_equal(res.nominal, np.log10(2.0))
        assert_almost_equal(res.std_dev, 0.2/(2.0*np.log(10)))
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.log10(qf2)
        assert_equal(res.nominal, np.log10(1.0))
        assert_almost_equal(res.std_dev, 0.1/np.log(10))
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.log10(qf3)
        assert_equal(res.nominal, np.log10([1, 2, 4]))
        assert_almost_equal(res.std_dev,
                            qf3.std_dev/(qf3.nominal*np.log(10)))
        assert_equal(res.unit, units.dimensionless_unscaled)

        with pytest.raises(UnitsError, match='log10 is only defined for '
                           'dimensionless quantities.'):
            res = np.log10(qf4)

        res = np.log10(qf5)
        assert_equal(res.nominal, np.log10(1.0))
        assert_almost_equal(res.std_dev, 0.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

    def test_qfloat_np_log1p(self):
        qf1 = QFloat(2.0, 0.2)
        qf2 = QFloat(1.0, 0.1)
        qf3 = QFloat([1, 2, 4], [0.1, 0.2, 0.4])
        qf4 = QFloat(1.0, 0.1, 's')
        qf5 = QFloat(1.0)

        res = np.log(qf1)
        assert_equal(res.nominal, np.log(2.0))
        assert_almost_equal(res.std_dev, 0.2/2.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.log(qf2)
        assert_equal(res.nominal, np.log(1.0))
        assert_almost_equal(res.std_dev, 0.1/1.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

        res = np.log(qf3)
        assert_equal(res.nominal, np.log([1, 2, 4]))
        assert_almost_equal(res.std_dev,
                            qf3.std_dev/qf3.nominal)
        assert_equal(res.unit, units.dimensionless_unscaled)

        with pytest.raises(UnitsError, match='log is only defined for '
                           'dimensionless quantities.'):
            res = np.log(qf4)

        res = np.log(qf5)
        assert_equal(res.nominal, np.log(1.0))
        assert_almost_equal(res.std_dev, 0.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

    def test_qfloat_np_multiply(self):
        qf1 = QFloat(2.0, 0.2, 'm')
        qf2 = QFloat(1.2, 0.1, 'm')
        qf3 = QFloat([1, 2, 4], [0.1, 0.2, 0.4], 'cm')

        res = np.multiply(qf1, 2)
        assert_equal(res.nominal, 4)
        assert_equal(res.std_dev, 0.4)
        assert_equal(res.unit, units.Unit('m'))

        res = np.multiply(qf1, qf2)
        assert_equal(res.nominal, 2.4)
        assert_almost_equal(res.std_dev, 0.3124099870362662)
        assert_equal(res.unit, units.Unit('m2'))

        res = np.multiply(qf1, qf3)
        assert_equal(res.nominal, [2, 4, 8])
        assert_almost_equal(res.std_dev, [0.28284271, 0.56568542, 1.13137085])
        assert_equal(res.unit, units.Unit('m*cm'))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.multiply(qf1, qf2, out=[])

    def test_qfloat_np_negative(self):
        qf1 = QFloat(1.0, 0.1, 'm')
        qf2 = QFloat(-1.0, 0.1, 'm')
        qf3 = QFloat(-5.0, 0.1)
        qf4 = QFloat(6)
        qf5 = QFloat([1, -1, 2, -2])

        assert_equal(np.negative(qf1), QFloat(-1.0, 0.1, 'm'))
        assert_equal(np.negative(qf2), QFloat(1.0, 0.1, 'm'))
        assert_equal(np.negative(qf3), QFloat(5.0, 0.1))
        assert_equal(np.negative(qf4), QFloat(-6))
        assert_equal(np.negative(qf5), QFloat([-1, 1, -2, 2]))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.negative(qf1, out=[])

    def test_qfloat_np_positive(self):
        qf1 = QFloat(1.0, 0.1, 'm')
        qf2 = QFloat(-1.0, 0.1, 'm')
        qf3 = QFloat(-5.0, 0.1)
        qf4 = QFloat(6)
        qf5 = QFloat([1, -1, 2, -2])

        assert_equal(np.positive(qf1), QFloat(1.0, 0.1, 'm'))
        assert_equal(np.positive(qf2), QFloat(-1.0, 0.1, 'm'))
        assert_equal(np.positive(qf3), QFloat(-5.0, 0.1))
        assert_equal(np.positive(qf4), QFloat(6))
        assert_equal(np.positive(qf5), QFloat([1, -1, 2, -2]))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.positive(qf1, out=[])

    @pytest.mark.parametrize('func', [np.power, np.float_power])
    def test_qfloat_np_power(self, func):
        qf1 = QFloat(2.0, 0.1, 'm')
        qf2 = QFloat([2, 3, 4], [0.1, 0.2, 0.3], 'm')
        qf3 = QFloat(2.0, 0.1)
        qf4 = QFloat([2, 3, 4])

        res = func(qf1, 2)
        assert_equal(res.nominal, 4)
        assert_equal(res.std_dev, 0.4)
        assert_equal(res.unit, units.Unit('m2'))

        res = func(qf1, 1.5)
        assert_almost_equal(res.nominal, 2.8284271247461903)
        assert_almost_equal(res.std_dev, 0.2121320343559643)
        assert_equal(res.unit, units.Unit('m(3/2)'))

        res = func(qf2, 2)
        assert_equal(res.nominal, [4, 9, 16])
        assert_almost_equal(res.std_dev, [0.4, 1.2, 2.4])
        assert_equal(res.unit, units.Unit('m2'))

        res = func(qf2, 1.5)
        assert_almost_equal(res.nominal, [2.82842712, 5.19615242, 8])
        assert_almost_equal(res.std_dev, [0.21213203, 0.51961524, 0.9])
        assert_equal(res.unit, units.Unit('m(3/2)'))

        res = func(qf1, qf3)
        assert_equal(res.nominal, 4)
        assert_almost_equal(res.std_dev, 0.4866954717550927)
        assert_equal(res.unit, units.Unit('m2'))

        with pytest.raises(ValueError):
            func(qf1, qf4)

        with pytest.raises(ValueError):
            func(qf2, qf4)

        with pytest.raises(ValueError):
            func(qf4, qf1)

        with pytest.raises(NotImplementedError):
            # out argument should fail
            func(qf1, 2, out=[])

    @pytest.mark.parametrize('func', [np.mod, np.remainder])
    def test_qfloat_np_remainder(self, func):
        qf1 = QFloat(5.0, 0.1, 'm')
        qf2 = QFloat(3.5, 0.1, 'm')
        qf3 = QFloat(1.0, 0.1, 's')
        qf4 = QFloat([1, 2, 3])

        res = func(qf1, 2)
        assert_equal(res.nominal, 1)
        assert_equal(res.std_dev, 0.1)
        assert_equal(res.unit, units.Unit('m'))

        res = func(qf1, qf2)
        assert_equal(res.nominal, 1.5)
        assert_equal(res.std_dev, 0.14142135623730953)
        assert_equal(res.unit, units.Unit('m'))

        res = func(qf1, qf3)
        assert_equal(res.nominal, 0)
        assert_equal(res.std_dev, np.inf)
        assert_equal(res.unit, units.Unit('m'))

        res = func(qf1, qf4)
        assert_equal(res.nominal, [0, 1, 2])
        assert_equal(res.std_dev, [np.nan, 0.1, 0.1])
        assert_equal(res.unit, units.Unit('m'))

        res = func(qf4, 1.5)
        assert_equal(res.nominal, [1, 0.5, 0])
        assert_equal(res.std_dev, [0, 0, np.nan])
        assert_equal(res.unit, units.Unit(''))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            func(qf1, qf2, out=[])

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_rint(self):
        raise NotImplementedError

    def test_qfloat_np_sign(self):
        assert_equal(np.sign(QFloat(1.5, 0.1, 'm')), 1)
        assert_equal(np.sign(QFloat(-1.5, 0.1, 'm')), -1)
        assert_equal(np.sign(QFloat(0.0, 0.1, 'm')), 0)

    def test_qfloat_np_signbit(self):
        assert_equal(np.sign(QFloat(1.5, 0.1, 'm')), 1)
        assert_equal(np.sign(QFloat(-1.5, 0.1, 'm')), -1)
        assert_equal(np.sign(QFloat(0.0, 0.1, 'm')), 0)
        assert_equal(np.sign(QFloat(-0.0, 0.1, 'm')), -0)

    def test_qfloat_np_sqrt(self):
        qf1 = QFloat(4, 0.1, 'm2')
        qf2 = QFloat([9, 100], [0.1, 0.1], 's2')

        res = np.sqrt(qf1)
        assert_equal(res.nominal, 2)
        assert_equal(res.std_dev, 0.025)
        assert_equal(res.unit, units.Unit('m'))

        res = np.sqrt(qf2)
        assert_equal(res.nominal, [3, 10])
        assert_almost_equal(res.std_dev, [0.01666667, 0.005])
        assert_equal(res.unit, units.Unit('s'))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.sqrt(qf1, out=[])

    def test_qfloat_np_square(self):
        qf1 = QFloat(2.0, 0.1, 'm')
        qf2 = QFloat([1, 2, 3], [0.1, 0.2, 0.3], 'cm')

        res = np.square(qf1)
        assert_equal(res.nominal, 4)
        assert_almost_equal(res.std_dev, 0.28284271247461906)
        assert_equal(res.unit, units.Unit('m2'))

        res = np.square(qf2)
        assert_equal(res.nominal, [1, 4, 9])
        assert_almost_equal(res.std_dev, [0.14142136, 0.56568542, 1.27279221])
        assert_equal(res.unit, units.Unit('cm2'))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.square(qf1, out=[])

    def test_qfloat_np_subtract(self):
        qf1 = QFloat(2.0, 0.2, 'm')
        qf2 = QFloat(1.0, 0.1, 'm')
        qf3 = QFloat([1, 2, 3], [0.1, 0.2, 0.3], 'm')
        qf4 = QFloat(1.0, 0.1, 's')
        qf5 = QFloat(1.0)

        res = np.subtract(qf1, qf2)
        assert_equal(res.nominal, 1.0)
        assert_almost_equal(res.std_dev, 0.223606797749979)
        assert_equal(res.unit, units.Unit('m'))

        res = np.subtract(qf1, qf3)
        assert_equal(res.nominal, [1, 0, -1])
        assert_almost_equal(res.std_dev, [0.2236068, 0.28284271, 0.36055513])
        assert_equal(res.unit, units.Unit('m'))

        res = np.subtract(qf3, qf1)
        assert_equal(res.nominal, [-1, 0, 1])
        assert_almost_equal(res.std_dev, [0.2236068, 0.28284271, 0.36055513])
        assert_equal(res.unit, units.Unit('m'))

        with pytest.raises(UnitsError):
            np.subtract(qf1, qf4)

        with pytest.raises(UnitsError):
            np.subtract(qf1, qf5)

        with pytest.raises(UnitsError):
            np.subtract(qf1, 1.0)

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.subtract(qf1, qf2, out=[])

    def test_qfloat_np_trunc(self):
        assert_equal(np.trunc(QFloat(1.5, 0.1, 'm')), QFloat(1, 0.0, 'm'))
        assert_equal(np.trunc(QFloat(-1.5, 0.1, 'm')), QFloat(-1, 0.0, 'm'))


class TestQFloatNumpyUfuncTrigonometric:
    """Test the numpy trigonometric and inverse trigonometric functions."""

    # Both radians and deg2rad must work in the same way
    @pytest.mark.parametrize('func', [np.radians, np.deg2rad])
    def test_qfloat_np_radians(self, func):
        qf = QFloat(180, 0.1, 'degree')
        res = func(qf)
        assert_almost_equal(res.nominal, 3.141592653589793)
        assert_almost_equal(res.std_dev, 0.001745329251994)
        assert_equal(res.unit, units.Unit('rad'))

        qf = QFloat(-180, 0.1, 'degree')
        res = func(qf)
        assert_almost_equal(res.nominal, -3.141592653589793)
        assert_almost_equal(res.std_dev, 0.001745329251994)
        assert_equal(res.unit, units.Unit('rad'))

        qf = QFloat([0, 30, 45, 60, 90], [0.1, 0.2, 0.3, 0.4, 0.5], 'degree')
        res = func(qf)
        assert_almost_equal(res.nominal, [0, 0.52359878, 0.78539816,
                                          1.04719755, 1.57079633])
        assert_almost_equal(res.std_dev, [0.00174533, 0.00349066, 0.00523599,
                                          0.00698132, 0.00872665])
        assert_equal(res.unit, units.Unit('rad'))

        # radian should no change
        qf = QFloat(1.0, 0.1, 'radian')
        res = func(qf)
        assert_equal(res.nominal, 1.0)
        assert_equal(res.std_dev, 0.1)
        assert_equal(res.unit, units.Unit('rad'))

        # Invalid units
        for unit in ('m', None, 'm/s'):
            with pytest.raises(UnitsError):
                func(QFloat(1.0, 0.1, unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            func(qf, out=[])

    # Both degrees and rad2deg must work in the same way
    @pytest.mark.parametrize('func', [np.degrees, np.rad2deg])
    def test_qfloat_np_degrees(self, func):
        qf = QFloat(np.pi, 0.05, 'radian')
        res = func(qf)
        assert_almost_equal(res.nominal, 180.0)
        assert_almost_equal(res.std_dev, 2.8647889756541165)
        assert_equal(res.unit, units.Unit('deg'))

        qf = QFloat(-np.pi, 0.05, 'radian')
        res = func(qf)
        assert_almost_equal(res.nominal, -180.0)
        assert_almost_equal(res.std_dev, 2.8647889756541165)
        assert_equal(res.unit, units.Unit('deg'))

        qf = QFloat([np.pi, np.pi/2, np.pi/4, np.pi/6],
                    [0.01, 0.02, 0.03, 0.04], 'rad')
        res = func(qf)
        assert_almost_equal(res.nominal, [180.0, 90.0, 45.0, 30.0])
        assert_almost_equal(res.std_dev, [0.5729578, 1.14591559,
                                          1.71887339, 2.29183118])
        assert_equal(res.unit, units.Unit('deg'))

        # deg should no change
        qf = QFloat(1.0, 0.1, 'deg')
        res = func(qf)
        assert_equal(res.nominal, 1.0)
        assert_equal(res.std_dev, 0.1)
        assert_equal(res.unit, units.Unit('deg'))

        # Invalid units
        for unit in ('m', None, 'm/s'):
            with pytest.raises(UnitsError):
                func(QFloat(1.0, 0.1, unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            func(qf, out=[])

    def test_qfloat_np_sin(self):
        qf = QFloat(np.pi, 0.05, 'radian')
        res = np.sin(qf)
        assert_almost_equal(res.nominal, 0.0)
        assert_almost_equal(res.std_dev, 0.05)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat(90, 0.05, 'deg')
        res = np.sin(qf)
        assert_almost_equal(res.nominal, 1.0)
        assert_almost_equal(res.std_dev, 0.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat([30, 45, 60], [0.1, 0.2, 0.3], 'deg')
        res = np.sin(qf)
        assert_almost_equal(res.nominal, [0.5, 0.70710678, 0.8660254])
        assert_almost_equal(res.std_dev, [0.0015115, 0.00246827, 0.00261799])
        assert_equal(res.unit, units.dimensionless_unscaled)

        for unit in ['m', 'm/s', None]:
            with pytest.raises(UnitsError):
                np.sin(QFloat(1.0, unit=unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.sin(qf, out=[])

    def test_qfloat_np_cos(self):
        qf = QFloat(180, 0.05, 'deg')
        res = np.cos(qf)
        assert_almost_equal(res.nominal, -1.0)
        assert_almost_equal(res.std_dev, 0.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat(np.pi/2, 0.05, 'rad')
        res = np.cos(qf)
        assert_almost_equal(res.nominal, 0.0)
        assert_almost_equal(res.std_dev, 0.05)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat([30, 45, 60], [0.1, 0.2, 0.3], 'deg')
        res = np.cos(qf)
        assert_almost_equal(res.nominal, [0.8660254, 0.70710678, 0.5])
        assert_almost_equal(res.std_dev, [0.00087266, 0.00246827, 0.0045345])
        assert_equal(res.unit, units.dimensionless_unscaled)

        for unit in ['m', 'm/s', None]:
            with pytest.raises(UnitsError):
                np.cos(QFloat(1.0, unit=unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.cos(qf, out=[])

    def test_qfloat_np_tan(self):
        qf = QFloat(45, 0.05, 'deg')
        res = np.tan(qf)
        assert_almost_equal(res.nominal, 1.0)
        assert_almost_equal(res.std_dev, 0.0017453292519943294)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat(np.pi/4, 0.05, 'rad')
        res = np.tan(qf)
        assert_almost_equal(res.nominal, 1.0)
        assert_almost_equal(res.std_dev, 0.1)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat([0, 30, 60], [0.1, 0.2, 0.3], 'deg')
        res = np.tan(qf)
        assert_almost_equal(res.nominal, [0, 0.57735027, 1.73205081])
        assert_almost_equal(res.std_dev, [0.00174533, 0.00465421, 0.02094395])
        assert_equal(res.unit, units.dimensionless_unscaled)

        for unit in ['m', 'm/s', None]:
            with pytest.raises(UnitsError):
                np.tan(QFloat(1.0, unit=unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.tan(qf, out=[])

    def test_qfloat_np_sinh(self):
        qf = QFloat(0, 0.05, 'radian')
        res = np.sinh(qf)
        assert_almost_equal(res.nominal, 0.0)
        assert_almost_equal(res.std_dev, 0.05)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat(np.pi, 0.05, 'radian')
        res = np.sinh(qf)
        assert_almost_equal(res.nominal, 11.548739357257748)
        assert_almost_equal(res.std_dev, 0.5795976637760759)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat(90, 0.05, 'deg')
        res = np.sinh(qf)
        assert_almost_equal(res.nominal, 2.3012989023072947)
        assert_almost_equal(res.std_dev, 0.002189671298638268)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat([30, 45, 60], [0.1, 0.2, 0.3], 'deg')
        res = np.sinh(qf)
        assert_almost_equal(res.nominal, [0.5478535, 0.86867096, 1.24936705])
        assert_almost_equal(res.std_dev, [0.0019901, 0.0046238, 0.0083791])
        assert_equal(res.unit, units.dimensionless_unscaled)

        for unit in ['m', 'm/s', None]:
            with pytest.raises(UnitsError):
                np.sinh(QFloat(1.0, unit=unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.sinh(qf, out=[])

    def test_qfloat_np_cosh(self):
        qf = QFloat(0, 0.05, 'radian')
        res = np.cosh(qf)
        assert_almost_equal(res.nominal, 1.0)
        assert_almost_equal(res.std_dev, 0.0)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat(np.pi, 0.05, 'radian')
        res = np.cosh(qf)
        assert_almost_equal(res.nominal, 11.591953275521519)
        assert_almost_equal(res.std_dev, 0.5774369678628875)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat(90, 0.05, 'deg')
        res = np.cosh(qf)
        assert_almost_equal(res.nominal, 2.5091784786580567)
        assert_almost_equal(res.std_dev, 0.0020082621458896)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat([30, 45, 60], [0.1, 0.2, 0.3], 'deg')
        res = np.cosh(qf)
        assert_almost_equal(res.nominal, [1.14023832, 1.32460909, 1.60028686])
        assert_almost_equal(res.std_dev, [0.00095618, 0.00303223, 0.00654167])
        assert_equal(res.unit, units.dimensionless_unscaled)

        for unit in ['m', 'm/s', None]:
            with pytest.raises(UnitsError):
                np.cosh(QFloat(1.0, unit=unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.cosh(qf, out=[])

    def test_qfloat_np_tanh(self):
        qf = QFloat(0, 0.05, 'radian')
        res = np.tanh(qf)
        assert_almost_equal(res.nominal, 0.0)
        assert_almost_equal(res.std_dev, 0.05)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat(np.pi, 0.05, 'radian')
        res = np.tanh(qf)
        assert_almost_equal(res.nominal, 0.99627207622075)
        assert_almost_equal(res.std_dev, 0.00037209750714)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat(90, 0.05, 'deg')
        res = np.tanh(qf)
        assert_almost_equal(res.nominal, 0.9171523356672744)
        assert_almost_equal(res.std_dev, 0.0001386067128590)
        assert_equal(res.unit, units.dimensionless_unscaled)

        qf = QFloat([30, 45, 60], [0.1, 0.2, 0.3], 'deg')
        res = np.tanh(qf)
        assert_almost_equal(res.nominal, [0.48047278, 0.6557942, 0.78071444])
        assert_almost_equal(res.std_dev, [0.00134241, 0.00198944, 0.00204457])
        assert_equal(res.unit, units.dimensionless_unscaled)

        for unit in ['m', 'm/s', None]:
            with pytest.raises(UnitsError):
                np.tanh(QFloat(1.0, unit=unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.tanh(qf, out=[])

    def test_qfloat_np_arcsin(self):
        qf = QFloat(np.sqrt(2)/2, 0.01)
        res = np.arcsin(qf)
        assert_almost_equal(res.nominal, 0.7853981633974484)
        assert_almost_equal(res.std_dev, 0.0141421356237309)
        assert_equal(res.unit, units.Unit('rad'))

        qf = QFloat([0, 0.5, 1], [0.01, 0.2, 0.3])
        res = np.arcsin(qf)
        assert_almost_equal(res.nominal, [0, 0.52359878, 1.57079633])
        assert_almost_equal(res.std_dev, [0.01, 0.23094011, np.inf])
        assert_equal(res.unit, units.Unit('rad'))

        # Invalid units
        for unit in ['m', 'm/s', 'rad', 'deg']:
            with pytest.raises(UnitsError):
                np.arcsin(QFloat(1.0, unit=unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.arcsin(qf, out=[])

    def test_qfloat_np_arccos(self):
        qf = QFloat(np.sqrt(2)/2, 0.01)
        res = np.arccos(qf)
        assert_almost_equal(res.nominal, 0.7853981633974484)
        assert_almost_equal(res.std_dev, 0.0141421356237309)
        assert_equal(res.unit, units.Unit('rad'))

        qf = QFloat([0, 0.5, 1], [0.01, 0.2, 0.3])
        res = np.arccos(qf)
        assert_almost_equal(res.nominal, [1.57079633, 1.04719755, 0])
        assert_almost_equal(res.std_dev, [0.01, 0.23094011, np.inf])
        assert_equal(res.unit, units.Unit('rad'))

        # Invalid units
        for unit in ['m', 'm/s', 'rad', 'deg']:
            with pytest.raises(UnitsError):
                np.arccos(QFloat(1.0, unit=unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.arccos(qf, out=[])

    def test_qfloat_np_arctan(self):
        qf = QFloat(1.0, 0.01)
        res = np.arctan(qf)
        assert_almost_equal(res.nominal, 0.7853981633974484)
        assert_almost_equal(res.std_dev, 0.005)
        assert_equal(res.unit, units.Unit('rad'))

        qf = QFloat([0, 0.5, 1], [0.01, 0.2, 0.3])
        res = np.arctan(qf)
        assert_almost_equal(res.nominal, [0, 0.4636476, 0.7853982])
        assert_almost_equal(res.std_dev, [0.01, 0.16, 0.15])
        assert_equal(res.unit, units.Unit('rad'))

        # Invalid units
        for unit in ['m', 'm/s', 'rad', 'deg']:
            with pytest.raises(UnitsError):
                np.arctan(QFloat(1.0, unit=unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.arctan(qf, out=[])

    def test_qfloat_np_arcsinh(self):
        qf = QFloat(0.0, 0.01)
        res = np.arcsinh(qf)
        assert_almost_equal(res.nominal, 0.0)
        assert_almost_equal(res.std_dev, 0.01)
        assert_equal(res.unit, units.Unit('rad'))

        qf = QFloat([0.5, 1.0, 10], [0.01, 0.2, 0.3])
        res = np.arcsinh(qf)
        assert_almost_equal(res.nominal, [0.4812118, 0.8813736, 2.998223])
        assert_almost_equal(res.std_dev, [0.0089443, 0.1414214, 0.0298511])
        assert_equal(res.unit, units.Unit('rad'))

        # Invalid units
        for unit in ['m', 'm/s', 'rad', 'deg']:
            with pytest.raises(UnitsError):
                np.arcsinh(QFloat(1.0, unit=unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.arcsinh(qf, out=[])

    def test_qfloat_np_arccosh(self):
        qf = QFloat(1.0, 0.01)
        res = np.arccosh(qf)
        assert_almost_equal(res.nominal, 0.0)
        # assert_almost_equal(res.std_dev, np.inf)
        assert_equal(res.unit, units.Unit('rad'))

        qf = QFloat([1.5, 5.0, 10], [0.01, 0.2, 0.3])
        res = np.arccosh(qf)
        assert_almost_equal(res.nominal, [0.9624237, 2.2924317, 2.9932228])
        assert_almost_equal(res.std_dev, [0.0089443, 0.0408248, 0.0301511])
        assert_equal(res.unit, units.Unit('rad'))

        # Invalid units
        for unit in ['m', 'm/s', 'rad', 'deg']:
            with pytest.raises(UnitsError):
                np.arccosh(QFloat(1.0, unit=unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.arccosh(qf, out=[])

    def test_qfloat_np_arctanh(self):
        qf = QFloat(0.0, 0.01)
        res = np.arctanh(qf)
        assert_almost_equal(res.nominal, 0.0)
        assert_almost_equal(res.std_dev, 0.01)
        assert_equal(res.unit, units.Unit('rad'))

        qf = QFloat([0.1, 0.5, 1.0], [0.01, 0.2, 0.3])
        res = np.arctanh(qf)
        assert_almost_equal(res.nominal, [0.1003353, 0.5493061, np.inf])
        assert_almost_equal(res.std_dev, [0.010101, 0.2666667, np.inf])
        assert_equal(res.unit, units.Unit('rad'))

        # Invalid units
        for unit in ['m', 'm/s', 'rad', 'deg']:
            with pytest.raises(UnitsError):
                np.arctanh(QFloat(1.0, unit=unit))

        with pytest.raises(NotImplementedError):
            # out argument should fail
            np.arctanh(qf, out=[])

    def test_qfloat_np_arctan2(self):
        qf1 = QFloat(1.0, 0.01)
        qf2 = QFloat(0.0, 0.01)
        res = np.arctan2(qf1, qf2)
        assert_almost_equal(res.nominal, 1.57079633)
        assert_almost_equal(res.std_dev, 0.01)
        assert_equal(res.unit, units.Unit('rad'))

        qf1 = QFloat([0.5, 1.0, 10], [0.01, 0.2, 0.3])
        qf2 = QFloat([0.1, 0.5, 1.0], [0.01, 0.2, 0.3])
        res = np.arctan2(qf1, qf2)
        assert_almost_equal(res.nominal, [1.373401, 1.107149, 1.471128])
        assert_almost_equal(res.std_dev, [0.019612, 0.178885, 0.029851])
        assert_equal(res.unit, units.Unit('rad'))
