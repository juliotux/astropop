# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from astropop.math.physical import QFloat, UnitsError
from numpy.testing import assert_almost_equal, assert_equal

# Testing qfloat compatibility with Numpy ufuncs and array functions.


class TestQFloatNumpyArrayFuncs:
    """Test numpy array functions for numpy comatibility."""

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
                      [9.0, 10.0, 11.0, 12.0]]        )
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

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_hstack(self):
        raise NotImplementedError

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

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_nanprod(self):
        raise NotImplementedError

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_nansum(self):
        raise NotImplementedError

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
        tgt = np.array([])
        res = np.resize(qf, shp)
        assert_almost_equal(res.nominal, tgt)
        assert_almost_equal(res.std_dev, tgt * 0.1)
        assert_equal(res.unit, qf.unit)
        assert_equal(res.shape, shp)

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

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_square(self):
        raise NotImplementedError

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

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_sum(self):
        raise NotImplementedError

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_qfloat_np_stack(self):
        raise NotImplementedError

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


class TestQFloatNumpyUfuncs:
    """Test numpy array functions for numpy comatibility."""

    def test_qfloat_np_absolute(self):
        raise NotImplementedError

    def test_qfloat_np_add(self):
        raise NotImplementedError

    def test_qfloat_np_cbrt(self):
        raise NotImplementedError

    def test_qfloat_np_ceil(self):
        raise NotImplementedError

    def test_qfloat_np_copysign(self):
        raise NotImplementedError

    def test_qfloat_np_divide(self):
        raise NotImplementedError

    def test_qfloat_np_divmod(self):
        raise NotImplementedError

    def test_qfloat_np_exp(self):
        raise NotImplementedError

    def test_qfloat_np_exp2(self):
        raise NotImplementedError

    def test_qfloat_np_expm1(self):
        raise NotImplementedError

    def test_qfloat_np_fabs(self):
        raise NotImplementedError

    def test_qfloat_np_float_power(self):
        raise NotImplementedError

    def test_qfloat_np_floor(self):
        raise NotImplementedError

    def test_qfloat_np_floor_divide(self):
        raise NotImplementedError

    def test_qfloat_np_fmax(self):
        raise NotImplementedError

    def test_qfloat_np_fmin(self):
        raise NotImplementedError

    def test_qfloat_np_fmod(self):
        raise NotImplementedError

    def test_qfloat_np_hypot(self):
        raise NotImplementedError

    def test_qfloat_np_isfinit(self):
        raise NotImplementedError

    def test_qfloat_np_isinf(self):
        raise NotImplementedError

    def test_qfloat_np_isnan(self):
        raise NotImplementedError

    def test_qfloat_np_log(self):
        raise NotImplementedError

    def test_qfloat_np_log2(self):
        raise NotImplementedError

    def test_qfloat_np_log10(self):
        raise NotImplementedError

    def test_qfloat_np_log1p(self):
        raise NotImplementedError

    def test_qfloat_np_maximum(self):
        raise NotImplementedError

    def test_qfloat_np_minimum(self):
        raise NotImplementedError

    def test_qfloat_np_mod(self):
        raise NotImplementedError

    def test_qfloat_np_modf(self):
        raise NotImplementedError

    def test_qfloat_np_multiply(self):
        raise NotImplementedError

    def test_qfloat_np_negative(self):
        raise NotImplementedError

    def test_qfloat_np_positive(self):
        raise NotImplementedError

    def test_qfloat_np_power(self):
        raise NotImplementedError

    def test_qfloat_np_remainder(self):
        raise NotImplementedError

    def test_qfloat_np_rint(self):
        raise NotImplementedError

    def test_qfloat_np_sign(self):
        raise NotImplementedError

    def test_qfloat_np_signbit(self):
        raise NotImplementedError

    def test_qfloat_np_sqrt(self):
        raise NotImplementedError

    def test_qfloat_np_square(self):
        raise NotImplementedError

    def test_qfloat_np_subtract(self):
        raise NotImplementedError

    def test_qfloat_np_true_divide(self):
        raise NotImplementedError

    def test_qfloat_np_trunc(self):
        raise NotImplementedError


class TestQFloatNumpyUfuncTrigonometric:
    """Test the numpy trigonometric and inverse trigonometric functions."""

    def test_qfloat_np_radians(self):
        raise NotImplementedError

    def test_qfloat_np_degrees(self):
        raise NotImplementedError

    def test_qfloat_np_sin(self):
        raise NotImplementedError

    def test_qfloat_np_cos(self):
        raise NotImplementedError

    def test_qfloat_np_tan(self):
        raise NotImplementedError

    def test_qfloat_np_sinh(self):
        raise NotImplementedError

    def test_qfloat_np_cosh(self):
        raise NotImplementedError

    def test_qfloat_np_tanh(self):
        raise NotImplementedError

    def test_qfloat_np_arcsin(self):
        raise NotImplementedError

    def test_qfloat_np_arccos(self):
        raise NotImplementedError

    def test_qfloat_np_arctan(self):
        raise NotImplementedError

    def test_qfloat_np_arcsinh(self):
        raise NotImplementedError

    def test_qfloat_np_arccosh(self):
        raise NotImplementedError

    def test_qfloat_np_arctanh(self):
        raise NotImplementedError
