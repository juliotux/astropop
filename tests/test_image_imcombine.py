# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import os
import json
import subprocess
import sys
import textwrap
import warnings
import numpy as np
import pytest

from astropy.io import fits

from astropy.utils import NumpyRNGContext
from astropop.framedata import FrameData, PixelMaskFlags
from astropop.image.imcombine import imcombine, _sigma_clip, \
                                     _minmax_clip, ImCombiner
from astropop.testing import *


class TestCombinerMemory:
    @pytest.mark.parametrize('dtype', ['uint16', 'float32', 'float64'])
    def test_median_peak_allocations(self, dtype):
        # Isolate tracing from pytest/plugins and any already-running tracer.
        # Imports and caller-owned frames are prepared before measurement;
        # loading internal copies, combining, errors and cleanup are measured.
        script = textwrap.dedent('''
            import json
            import logging
            import sys
            import tracemalloc
            import numpy as np
            from astropop.framedata import FrameData
            from astropop.image import imcombine
            from astropop.logger import logger

            logger.setLevel(logging.ERROR)
            dtype = np.dtype(sys.argv[1])
            shape = (128, 128)
            frames = [FrameData(np.full(shape, i, dtype=dtype), dtype=dtype,
                                uncertainty=np.ones(shape, dtype=dtype))
                      for i in range(128)]
            # Exercise mask handling while retaining the same known median
            # at every output pixel (the two removed values are symmetric).
            frames[0].mask_pixels((0, 0))
            frames[-1].mask_pixels((0, 0))
            cube_bytes = sum(frame.data.nbytes for frame in frames)
            tracemalloc.start()
            # Fail explicitly if NumPy allocations are invisible to tracing.
            probe = np.empty(2**20, dtype=np.uint8)
            assert tracemalloc.get_traced_memory()[1] >= probe.nbytes
            del probe
            tracemalloc.reset_peak()
            baseline = tracemalloc.get_traced_memory()[0]
            result = imcombine(frames, method='median', dtype=dtype,
                               memory_limit=64 * 2**20)
            peak = tracemalloc.get_traced_memory()[1] - baseline
            tracemalloc.stop()
            np.testing.assert_array_equal(result.data, 63 if dtype.kind in 'iu' else 63.5)
            assert result.uncertainty is not None
            assert not result.mask.any()
            print(json.dumps(dict(peak=peak, cube_bytes=cube_bytes)))
        ''')
        process = subprocess.run([sys.executable, '-c', script, dtype],
                                 capture_output=True, text=True, timeout=120)
        assert process.returncode == 0, process.stdout + process.stderr
        measured = json.loads(process.stdout)
        # Allow four nominal cubes for owned input copies/flags, the chunk,
        # validity masks, outputs and bounded statistics scratch, with margin
        # for Python objects and NumPy/platform variation. A full-stack median
        # workspace exceeded this bound in the former implementation. This is
        # an allocation regression bound, not an RSS or memory_limit promise.
        # Two-byte pixels need more relative room for flags and promoted
        # statistics blocks; a full float64 sample cube still exceeds it.
        allowance = 6 if dtype == 'uint16' else 4
        assert measured['peak'] < allowance * measured['cube_bytes'], measured


class Test_MinMaxClip():
    def test_1D_simple(self):
        arr = np.arange(10)
        low, high = (2, 6)
        expect = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

        mask = _minmax_clip(arr, low, high)
        assert_equal(mask, expect)

    def test_2D_simple(self):
        arr = np.arange(10).reshape((2, 5))
        low, high = (2, 6)
        expect = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                          dtype=bool).reshape((2, 5))

        mask = _minmax_clip(arr, low, high)
        assert_equal(mask, expect)

    def test_3D_simple(self):
        arr = np.array([[[0, 1, 1], [2, 3, 3], [1, 2, 3]],
                        [[2, 3, 4], [2, 5, 6], [1, 0, 0]],
                        [[0, 1, 1], [2, 3, 7], [7, 0, 1]]])
        low, high = (2, 6)
        exp = np.array([[[1, 1, 1], [0, 0, 0], [1, 0, 0]],
                        [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
                        [[1, 1, 1], [0, 0, 1], [1, 1, 1]]], dtype=bool)

        mask = _minmax_clip(arr, low, high)
        assert_equal(mask, exp)

    def test_minimum_disabled(self):
        arr = np.array([0, 1, 0, 2, 3, 0])
        low, high = (None, 1)
        expect = np.array([0, 0, 0, 1, 1, 0])
        mask = _minmax_clip(arr, low, high)
        assert_equal(mask, expect)

    def test_maximum_disabled(self):
        arr = np.array([0, 1, 0, 2, 3, 0])
        low, high = (1, None)
        expect = np.array([1, 0, 1, 0, 0, 1])
        mask = _minmax_clip(arr, low, high)
        assert_equal(mask, expect)

    def test_invalid(self):
        # must mask invalid numbers
        arr = np.array([0, 1, 2, np.inf, np.nan, 5, 1])
        low, high = (1, 3)
        expect = np.array([1, 0, 0, 1, 1, 1, 0])
        mask = _minmax_clip(arr, low, high)
        assert_equal(mask, expect)


class Test_SigmaClip():
    # TODO: test 3D
    # TODO: test axis in 3D
    def test_unkown_sigmaclip(self):
        arr = np.arange(10)
        with pytest.raises(TypeError):
            _sigma_clip(arr, 'testing')

        with pytest.raises(TypeError):
            _sigma_clip(arr, '1,2')

        # too many values must raise
        with pytest.raises(ValueError):
            _sigma_clip(arr, [1, 2, 3])

    def test_sigmclip_types(self):
        arr = np.arange(10)

        # must work with numbers
        _sigma_clip(arr, 1)

        # must work with 2-elements array
        _sigma_clip(arr, np.array([1, 2]))
        _sigma_clip(arr, [1, 2])
        _sigma_clip(arr, (1, 2))

    def test_invalid(self):
        arr = np.ones((5, 5))
        exp = np.zeros((5, 5), dtype=bool)

        indx = [(1, 1), (2, 1), (2, 3)]

        arr[indx[0]] = 1000
        arr[indx[1]] = np.inf
        arr[indx[2]] = np.nan

        for i in indx:
            exp[i] = True

        mask = _sigma_clip(arr)
        assert_equal(mask, exp)

    def test_functions_names(self):
        arr = np.ones((5, 5))

        # all this should run
        _sigma_clip(arr, cen_func='median')
        _sigma_clip(arr, cen_func='mean')
        _sigma_clip(arr, dev_func='std')
        _sigma_clip(arr, cen_func='mean', dev_func='std')
        _sigma_clip(arr, dev_func='mad_std')
        _sigma_clip(arr, cen_func='median', dev_func='mad_std')

    def test_functions_callable(self):
        arr = np.ones((5, 5))

        mask = _sigma_clip(arr, cen_func=np.median)
        mask = _sigma_clip(arr, dev_func=np.std)

        # testing forced 1.0pm0.5
        def _test_cen(*args, **kwargs):
            return 1.0

        def _test_dev(*args, **kwargs):
            return 0.5

        # 1.2 should not be masked with 1.0pm0.5, 2.0 and 1000 yes
        arr[0, 0] = 1.2
        arr[1, 1] = 2
        arr[3, 2] = 1000

        mask = _sigma_clip(arr, 1, cen_func=_test_cen, dev_func=_test_dev)
        for i in range(5):
            for j in range(5):
                if (i, j) in [(1, 1), (3, 2)]:
                    assert_true(mask[i, j])
                else:
                    assert_false(mask[i, j])

    def test_functions_invalid(self):
        arr = np.ones((5, 5))

        with pytest.raises(KeyError, match='invalid'):
            _sigma_clip(arr, cen_func='invalid')
        with pytest.raises(KeyError, match='invalid'):
            _sigma_clip(arr, dev_func='invalid')
        with pytest.raises(KeyError, match='1'):
            _sigma_clip(arr, cen_func=1)
        with pytest.raises(KeyError, match='1'):
            _sigma_clip(arr, dev_func=1)
        with pytest.raises(TypeError):
            _sigma_clip(arr, cen_func=[])
        with pytest.raises(TypeError):
            _sigma_clip(arr, dev_func=[])

    def test_1D_simple(self):
        with NumpyRNGContext(123):
            arr = np.random.normal(5, 2, 100)
        arr[32] = 1000
        arr[25] = 1000
        arr[12] = -1000
        exp = np.zeros(100, dtype=bool)
        exp[32] = True
        exp[25] = True
        exp[12] = True

        mask = _sigma_clip(arr, 3)
        assert_equal(mask, exp)

    def test_2D_defaults(self):
        # mean:4.532 median:4.847 std:282 mad_std: 2.02
        arr = np.array([[6.037, 5.972, 5.841, 2.775, 0.711],
                        [6.539, 4.677, -1000, 5.633, 3.478],
                        [4.847, 7.563, 3.323, 7.952, 6.646],
                        [6.136, 2.690, 4.193, 1000., 4.483],
                        [5.673, 7.479, 3.874, 4.756, 2.021]])
        # using defaults median and mad_std
        expect_1 = [[0, 0, 0, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 0, 1]]
        expect_1 = np.array(expect_1, dtype=bool)

        expect_3 = [[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0]]
        expect_3 = np.array(expect_3, dtype=bool)

        mask = _sigma_clip(arr, 3)
        assert_equal(mask, expect_3)


class Test_ImCombineConformance():
    @pytest.mark.parametrize('dtype', ['float32', np.dtype('float32'), np.float32])
    def test_dtype_specification(self, dtype):
        combiner = ImCombiner(dtype=dtype)
        assert combiner._dtype == np.dtype('float32')

    @pytest.mark.parametrize('dtype', [np.dtype('complex64'), bool, 'invalid'])
    def test_invalid_dtype_specification(self, dtype):
        with pytest.raises(ValueError, match='Only integer or float dtypes'):
            ImCombiner(dtype=dtype)

    def test_class_creation(self):
        # defaults
        c = ImCombiner()
        assert_is_instance(c, ImCombiner)
        assert_equal(c._max_memory, 1e9)
        assert_equal(c._dtype, np.float64)

        d = ImCombiner(max_memory=2e10)
        assert_equal(d._max_memory, 2e10)

        e = ImCombiner(dtype=np.float16)
        assert_equal(e._dtype, np.float16)

        f = ImCombiner(dtype=np.float32)
        assert_equal(f._dtype, np.float32)

    def test_creation_error_dtype(self):
        msg = "Only integer or float dtypes are allowed in ImCombiner."
        with pytest.raises(ValueError, match=msg):
            ImCombiner(dtype=np.complex64)

        with pytest.raises(ValueError, match=msg):
            ImCombiner(dtype=np.complex128)

    def test_creation_default_dtypes(self):
        c = ImCombiner(dtype=float)
        assert_equal(c._dtype, float)

        msg = "Only integer or float dtypes are allowed in ImCombiner."
        assert_equal(ImCombiner(dtype=int)._dtype, np.dtype(int))

        with pytest.raises(ValueError, match=msg):
            ImCombiner(dtype=complex)

    def test_class_set_sigmaclip(self):
        # empty must disable clipping
        c = ImCombiner()
        c.set_sigma_clip()
        assert_equal(c._sigma_clip, None)
        assert_equal(c._sigma_cen_func, None)
        assert_equal(c._sigma_dev_func, None)

        # set sigmaclip one value
        c = ImCombiner()
        c.set_sigma_clip(1)
        assert_equal(c._sigma_clip, 1)
        assert_equal(c._sigma_cen_func, 'median')
        assert_equal(c._sigma_dev_func, 'mad_std')

        # set sigmaclip two values
        c = ImCombiner()
        c.set_sigma_clip((1, 2))
        assert_equal(c._sigma_clip, (1, 2))
        assert_equal(c._sigma_cen_func, 'median')
        assert_equal(c._sigma_dev_func, 'mad_std')

        # Enable and disable
        c = ImCombiner()
        c.set_sigma_clip((1, 2))
        c.set_sigma_clip()
        assert_equal(c._sigma_clip, None)
        assert_equal(c._sigma_cen_func, None)
        assert_equal(c._sigma_dev_func, None)

        # set functions
        c = ImCombiner()
        c.set_sigma_clip((1, 2), 'mean', 'mad_std')
        assert_equal(c._sigma_clip, (1, 2))
        assert_equal(c._sigma_cen_func, 'mean')
        assert_equal(c._sigma_dev_func, 'mad_std')

    def test_class_set_sigmaclip_errors(self):
        # more then 2 elements must fail
        with pytest.raises(ValueError):
            c = ImCombiner()
            c.set_sigma_clip((1, 2, 3))

        # problematic functions
        with pytest.raises(ValueError):
            c = ImCombiner()
            c.set_sigma_clip((1, 2), 'no-existing')

        with pytest.raises(ValueError):
            c = ImCombiner()
            c.set_sigma_clip((1, 2), 'mean', 'no-existing')

        # None should fail
        with pytest.raises(ValueError):
            c = ImCombiner()
            c.set_sigma_clip((1, 2), None)

        with pytest.raises(ValueError):
            c = ImCombiner()
            c.set_sigma_clip((1, 2), 'mean', None)

    def test_class_set_minmax(self):
        c = ImCombiner()
        c.set_minmax_clip(0, 1)
        assert_equal(c._minmax, (0, 1))

        # test flipping
        c = ImCombiner()
        c.set_minmax_clip(1, 0)
        assert_equal(c._minmax, (0, 1))

        # test nones
        c = ImCombiner()
        c.set_minmax_clip(None, 0)
        assert_equal(c._minmax, (None, 0))
        c = ImCombiner()
        c.set_minmax_clip(0, None)
        assert_equal(c._minmax, (0, None))

        # test one element setting
        c = ImCombiner()
        c.set_minmax_clip(0)
        assert_equal(c._minmax, (0, None))
        c = ImCombiner()
        c.set_minmax_clip(min_value=0)
        assert_equal(c._minmax, (0, None))
        c = ImCombiner()
        c.set_minmax_clip(max_value=0)
        assert_equal(c._minmax, (None, 0))

        # disable
        c = ImCombiner()
        c.set_minmax_clip()
        assert_equal(c._minmax, None)
        c.set_minmax_clip(0, 1)
        assert_equal(c._minmax, (0, 1))
        c.set_minmax_clip()
        assert_equal(c._minmax, None)

    def test_class_set_minmax_errors(self):
        c = ImCombiner()

        # not numbers should fail
        with pytest.raises(ValueError):
            c.set_minmax_clip('a')

        # not numbers should fail
        with pytest.raises(ValueError):
            c.set_minmax_clip(0, 'a')

        # not numbers should fail
        with pytest.raises(ValueError):
            c.set_minmax_clip([1, 2])

        # not numbers should fail
        with pytest.raises(ValueError):
            c.set_minmax_clip(0, [1, 2])

    def test_check_consistency(self):
        n = 10
        d = np.ones((10, 10))
        li = [FrameData(d, unit='adu') for i in range(n)]
        comb = ImCombiner()
        # empty should raise
        with pytest.raises(ValueError, match='Combiner have no images.'):
            comb._check_consistency()

        comb._load_images(li)
        # nothing should raise
        comb._check_consistency()

        # incompatible unit should raise
        comb._images[3].unit = 'm'
        with pytest.raises(ValueError, match='.* unit incompatible .*'):
            comb._check_consistency()
        comb._images[3].unit = 'adu'

        # incompatible shape should raise
        comb._images[4].data = np.ones((2, 2))
        with pytest.raises(ValueError, match='.* shape incompatible .*'):
            comb._check_consistency()

    def test_invalid_method(self):
        n = 10
        d = np.ones((10, 10))
        li = [FrameData(d, unit='adu') for i in range(n)]
        comb = ImCombiner()
        with pytest.raises(ValueError, match='hulk-smash is not a valid '
                           'combining method.'):
            comb.combine(li, method='hulk-smash')


class Test_ImCombiner_ChunkYielder:
    @pytest.mark.parametrize('dtype', ['int16', 'float32', 'float64'])
    @pytest.mark.parametrize('method', ['sum', 'mean', 'median'])
    @pytest.mark.parametrize('with_errors', [False, True])
    def test_chunks_cover_image_with_bounded_buffers(self, dtype, method, with_errors):
        # Seven 13x17 images provide 221 output pixels. Half of the 4096-byte
        # budget (2048 bytes) is available for the stacked tile and results;
        # the other half is reserved for statistics work blocks.
        #
        # On 64-bit NumPy, the estimate per output pixel is:
        #   7 * (itemsize * (1 + propagated_errors) + 1 validity byte)
        #   + 4 * (8-byte index + itemsize).
        # Only sum propagates individual input errors; mean/median use scatter.
        #
        # dtype    errors  bytes/pixel  pixel limit  target chunks  actual
        # int16    no           61          33              7          13
        # int16    yes          75          27              9          13
        # float32  no           83          24             10          13
        # float32  yes         111          18             13          13
        # float64  no          127          16             14          39
        # float64  yes         183          11             21          39
        #
        # Targets are ceil(221 / pixel_limit). Slicing uses whole rows first:
        # int16/float32 yield 13 tiles of 1x17 pixels. float64 also divides
        # columns into widths 8, 8, 1, yielding 13 * 3 = 39 tiles.
        data = np.arange(13 * 17).reshape(13, 17)
        mask = np.zeros(data.shape, dtype=bool)
        mask[3:5, 4:7] = True
        frames = [FrameData(data, mask=mask, uncertainty=2 if with_errors else None)
                  for _ in range(7)]
        comb = ImCombiner(dtype=dtype, max_memory=4096)
        comb._load_images(frames)
        visits = np.zeros(data.shape, dtype=int)
        chunks = 0
        for chunk, errors, slc in comb._chunk_yielder(method):
            chunks += 1
            visits[slc] += 1
            assert chunk.dtype == np.dtype(dtype)
            np.testing.assert_array_equal(chunk, np.broadcast_to(data[slc], chunk.shape))
            np.testing.assert_array_equal(comb._valid, np.broadcast_to(~mask[slc], chunk.shape))
            allocated = chunk.nbytes + comb._valid.nbytes
            if method == 'sum' and with_errors:
                assert errors.dtype == chunk.dtype
                np.testing.assert_array_equal(errors, 2)
                allocated += errors.nbytes
            else:
                assert errors is None
            assert allocated <= 4096 // 2
        np.testing.assert_array_equal(visits, 1)
        assert chunks > 1
        if np.dtype(np.intp).itemsize == 8:
            assert chunks == (39 if dtype == 'float64' else 13)
        comb._clear()

    def test_large_budget_yields_one_chunk(self):
        # 7 * (8 data bytes + 1 validity byte) + 4 * (8 + 8) = 127
        # estimated bytes/pixel. All 221 pixels need only 28,067 bytes,
        # comfortably below the 50,000,000-byte tile allowance: one chunk.
        comb = ImCombiner(max_memory=1e8)
        comb._load_images([FrameData(np.ones((13, 17))) for _ in range(7)])
        chunks = 0
        for data, uncertainty, slc in comb._chunk_yielder('median'):
            chunks += 1
            assert data.shape == (7, 13, 17)
            assert uncertainty is None
            np.testing.assert_array_equal(comb._valid, True)
        assert chunks == 1
        comb._clear()

    def test_read_chunk_does_not_build_full_frame_masks(self, monkeypatch):
        # One float64 image costs an estimated 73 bytes/pixel. The 500-byte
        # allowance fits 6 pixels: ceil(323/6) = 54 target chunks. Row-first
        # slicing produces 17 rows * 5 column tiles (4,4,4,4,3) = 85 chunks.
        data = np.arange(17 * 19).reshape(17, 19).astype(float)
        data[2, 3] = np.nan
        frame = FrameData(data, mask=np.indices(data.shape)[0] > 12)
        # Status bits other than MASKED must remain accepted.
        frame.add_flags(PixelMaskFlags.SATURATED, (0, 0))
        comb = ImCombiner(max_memory=1000)
        comb._load_images([frame])
        def full_mask(frame):
            pytest.fail('Constructed a full-frame mask while reading a tile')
        monkeypatch.setattr(FrameData, 'mask', property(full_mask))
        for chunk, errors, slc in comb._chunk_yielder('mean'):
            expected = ~np.isnan(data[slc]) & (np.indices(data.shape)[0][slc] <= 12)
            np.testing.assert_array_equal(comb._valid[0], expected)
            np.testing.assert_array_equal(chunk[0], data[slc])
        comb._clear()

    def test_previous_chunk_is_released_before_next_allocation(self):
        import weakref
        # 500 / 127 fits 3 pixels. For 17x19 images this gives 108 target
        # chunks, then 17 rows * 10 column tiles (nine widths of 2, then 1)
        # = 170 actual chunks. Each previous tile must die before the next.
        comb = ImCombiner(max_memory=1000)
        comb._load_images([FrameData(np.ones((17, 19))) for _ in range(7)])
        previous = None
        for chunk, errors, slc in comb._chunk_yielder('median'):
            if previous is not None:
                assert previous() is None
            previous = weakref.ref(chunk)
            del chunk
        comb._clear()


class Test_ImCombiner_LoadImages():
    @pytest.mark.parametrize('disk_cache', [True, False])
    def test_image_loading_framedata(self, tmpdir, disk_cache):
        tmp = tmpdir.strpath
        n = 10
        d = np.ones((10, 10))
        li = [FrameData(d, unit='adu', uncertainty=d, cache_folder=tmp,
                        cache_filename=f'test{i}') for i in range(n)]

        comb = ImCombiner(use_disk_cache=disk_cache)
        # must start empty
        assert_equal(len(comb._images), 0)
        assert_is_none(comb._buffer)
        comb._load_images(li)
        assert_equal(len(comb._images), n)
        assert_is_none(comb._buffer)
        for i, v in enumerate(comb._images):
            fil = os.path.join(tmp, f'test{i}')
            assert_is_instance(v, FrameData)
            if disk_cache:
                assert_true(v._memmapping)
                # assert_path_exists(fil+'_copy_copy.data')
                # assert_path_exists(fil+'_copy_copy.unct')
                # assert_path_exists(fil+'_copy_copy.mask')

        comb._clear()
        # must start empty
        assert_equal(len(comb._images), 0)
        assert_is_none(comb._buffer)

        # ensure tmp files cleaned
        if disk_cache:
            for i in range(n):
                fil = os.path.join(tmp, f'test{i}')
                assert_path_not_exists(fil+'_copy_copy.data')
                assert_path_not_exists(fil+'_copy_copy.unct')
                assert_path_not_exists(fil+'_copy_copy.mask')

    @pytest.mark.parametrize('disk_cache', [True, False])
    def test_image_loading_fitsfile(self, tmpdir, disk_cache):
        tmp = tmpdir.strpath
        n = 10
        d = np.ones((10, 10))
        li = [os.path.join(tmp, f'fits_test{i}') for i in range(n)]
        for f in li:
            fits.PrimaryHDU(d).writeto(f)

        comb = ImCombiner(use_disk_cache=disk_cache)
        comb._load_images(li)

        assert_equal(len(comb._images), n)
        assert_is_none(comb._buffer)
        for i, v in enumerate(comb._images):
            assert_is_instance(v, FrameData)
            if disk_cache:
                assert_true(v._memmapping)

        comb._clear()
        # must start empty
        assert_equal(len(comb._images), 0)
        assert_is_none(comb._buffer)

    @pytest.mark.parametrize('disk_cache', [True, False])
    def test_image_loading_fitshdu(self, disk_cache):
        n = 10
        d = np.ones((10, 10))
        li = [fits.PrimaryHDU(d) for i in range(n)]

        comb = ImCombiner(use_disk_cache=disk_cache)
        comb._load_images(li)

        assert_equal(len(comb._images), n)
        assert_is_none(comb._buffer)
        for i, v in enumerate(comb._images):
            assert_is_instance(v, FrameData)
            if disk_cache:
                assert_true(v._memmapping)

        comb._clear()
        # must start empty
        assert_equal(len(comb._images), 0)
        assert_is_none(comb._buffer)

    def test_error_image_loading_empty(self):
        comb = ImCombiner()
        with pytest.raises(ValueError, match='Image list is empty.'):
            comb._load_images([])


class Test_ImCombiner_Rejection():
    @pytest.mark.parametrize('clip', [(0, 20), (0, None), (None, 20)])
    def test_apply_minmax_clip(self, clip):
        _min, _max = clip
        outliers = ((4, 2), (6, 2), (2, 9), (5, 2))
        outvalues = [1e6, -2e3, 5e3, -7e2]
        expect = np.zeros((10, 10))
        with NumpyRNGContext(123):
            data = np.random.normal(loc=10, scale=1, size=[10, 10])

        for p, v in zip(outliers, outvalues):
            data[p] = v
            if _min is not None:
                if _min > v:
                    expect[p] = 1
            if _max is not None:
                if _max < v:
                    expect[p] = 1

        data[0:2, 0:3] = np.nan
        expect[0:2, 0:3] = 1
        expect = expect.astype(bool)
        _s = np.sum(expect)

        # force assign the buffer
        comb = ImCombiner()
        comb._buffer = data
        # with these limits, only the outliers must be masked
        comb.set_minmax_clip(_min, _max)
        comb._valid = None
        comb._apply_rejection()
        # original mask must be kept
        _sr = np.sum(~comb._valid)
        assert_equal(~comb._valid, expect)

    def test_apply_sigmaclip(self):
        data = np.array([1, -1, 1, -1, 65000], dtype=np.float32)
        comb = ImCombiner()

        # if threshold=1, only 65000 must be masked.
        comb._buffer = data.copy()
        comb.set_sigma_clip(1)
        expect = [0, 0, 0, 0, 1]
        comb._valid = None
        comb._apply_rejection()
        assert_equal(~comb._valid, expect)

        # if threshold=3, 65000 must be masked too using mad_std.
        comb._buffer = data.copy()
        comb.set_sigma_clip(3)
        expect = [0, 0, 0, 0, 1]
        comb._valid = None
        comb._apply_rejection()
        assert_equal(~comb._valid, expect)

        # if threshold=3 and a mask, the mask must be preserved.
        comb._buffer = data.copy()
        comb._buffer[1] = np.nan
        comb.set_sigma_clip(3)
        expect = [0, 1, 0, 0, 1]
        comb._valid = None
        comb._apply_rejection()
        assert_equal(~comb._valid, expect)

    def test_apply_sigmaclip_only_lower(self):
        data = np.array([1, -1, 1, -1, 65000], dtype=np.float32)
        comb = ImCombiner()

        # 65000 must not be masked
        comb._buffer = data.copy()
        comb.set_sigma_clip((1, None))
        expect = [0, 0, 0, 0, 0]
        comb._valid = None
        comb._apply_rejection()
        assert_equal(~comb._valid, expect)

    def test_apply_sigmaclip_only_higher(self):
        data = np.array([1, -1, 1, -1, -65000], dtype=np.float32)
        comb = ImCombiner()

        # -65000 must not be masked
        comb._buffer = data.copy()
        comb.set_sigma_clip((None, 1))
        expect = [0, 0, 0, 0, 0]
        comb._valid = None
        comb._apply_rejection()
        assert_equal(~comb._valid, expect)


class Test_ImCombiner_Combine():
    @pytest.mark.parametrize('entry', ['class', 'function'])
    @pytest.mark.parametrize('disk_cache', [False, True])
    @pytest.mark.parametrize('method', ['sum', 'median', 'mean'])
    @pytest.mark.parametrize('dtype', ['int16', np.int32, np.dtype('uint16'), '>i2'])
    def test_integer_computation(self, tmp_path, entry, disk_cache, method, dtype):
        images = [FrameData([[value, 50], [value, value]], uncertainty=error,
                            unit='adu', mask=[[False, True], [False, False]])
                  for value, error in [(1.9, 3.9), (4.9, 4.9)]]
        kwargs = dict(dtype=dtype, use_disk_cache=disk_cache,
                      tmp_dir=str(tmp_path / 'cache'))
        if entry == 'class':
            result = ImCombiner(**kwargs).combine(images, method)
        else:
            result = imcombine(images, method=method, **kwargs)
        # Pixels and errors are converted before any reduction: [1, 4],
        # [3, 4]. Integer mean is 2; integer population std is 1.
        assert result.dtype == np.dtype(dtype)
        # FrameData uncertainty setters normalize byte order for memmaps.
        assert result.uncertainty.dtype == np.dtype(dtype).newbyteorder('=')
        expected = 5 if method == 'sum' else 2
        error = 5 if method == 'sum' else 1
        np.testing.assert_array_equal(result.data, [[expected, 0], [expected, expected]])
        np.testing.assert_array_equal(result.uncertainty, [[error, 0], [error, error]])
        np.testing.assert_array_equal(result.mask, [[False, True], [False, False]])
        assert result.unit == images[0].unit
        for frame, value in zip(images, (1.9, 4.9)):
            assert frame.data[0, 0] == value
        assert not list((tmp_path / 'cache').glob('*.npy'))

    @pytest.mark.parametrize('method', ['mean', 'median'])
    def test_integer_accumulators_promote(self, method):
        images = [np.full((2, 2), 30000, dtype='int16') for _ in range(2)]
        result = imcombine(images, method=method, dtype='int16')
        # The intermediate 60000 must not wrap before division by two.
        np.testing.assert_array_equal(result.data, 30000)
        assert result.dtype == np.dtype('int16')

    @pytest.mark.parametrize('method', ['mean', 'median'])
    def test_integer_negative_fraction_truncates_towards_zero(self, method):
        result = imcombine([np.full((2, 2), n) for n in (-1, -4)],
                           method=method, dtype='int16')
        np.testing.assert_array_equal(result.data, -2)

    def test_integer_uncertainty_promotes_before_squaring(self):
        images = [FrameData([[1]], uncertainty=200) for _ in range(2)]
        result = imcombine(images, method='sum', dtype='int16')
        np.testing.assert_array_equal(result.uncertainty, [[282]])

    @pytest.mark.parametrize('dtype, error', [('int16', 181), ('uint16', 255),
                                             ('int32', 46340), ('uint32', 65535),
                                             ('int64', 3037000499), ('uint64', 4294967295)])
    def test_integer_root_at_large_exact_squares(self, dtype, error):
        result = imcombine([FrameData([[1]], uncertainty=error)],
                           method='sum', dtype=dtype)
        np.testing.assert_array_equal(result.uncertainty, [[error]])
        assert result.uncertainty.dtype == np.dtype(dtype)

    def test_integer_output_cast_follows_numpy(self):
        values = np.full((2, 2, 2), 30000, dtype='int16')
        result = imcombine(list(values), method='sum', dtype='int16', sum_normalize=False)
        np.testing.assert_array_equal(result.data, np.sum(values, axis=0).astype('int16'))

    @pytest.mark.parametrize('invalid', [np.nan, np.inf])
    def test_integer_nonfinite_input_raises(self, invalid):
        with pytest.raises(ValueError, match='finite input pixels'):
            imcombine([np.full((2, 2), invalid)], dtype='int16')

    @pytest.mark.parametrize('method', ['mean', 'median', 'sum'])
    @pytest.mark.parametrize('skip', [False, True])
    def test_integer_rejection(self, monkeypatch, method, skip):
        from astropop.config import AstropopConfig
        images = [FrameData(np.full((3, 4), v), uncertainty=e)
                  for v, e in [(1, 3), (4, 4), (100, 100)]]
        monkeypatch.setattr(AstropopConfig, 'FRAMEDATA_DISABLE_UNCERTAINTY', skip)
        combiner = ImCombiner(dtype='int16', max_memory=64)
        combiner.set_minmax_clip(0, 10)
        result = combiner.combine(images, method)
        # Normalize in promoted arithmetic, then truncate only the output.
        np.testing.assert_array_equal(result.data, 7 if method == 'sum' else 2)
        if skip:
            assert result.uncertainty is None
        else:
            np.testing.assert_array_equal(result.uncertainty, 7 if method == 'sum' else 1)
        assert images[-1]._unct[0, 0] == 100

    def test_integer_sigma_clipping(self):
        images = [np.full((2, 2), value) for value in (1, 2, 3, 100)]
        result = imcombine(images, method='mean', dtype='int16', sigma_clip=3,
                           sigma_dev_func='mad_std')
        np.testing.assert_array_equal(result.data, 2)

    @pytest.mark.parametrize('dtype', ['int16', 'float32', 'float64'])
    def test_clipping_callback_cannot_modify_pixels(self, dtype):
        def center(data, axis):
            data[:] = 999
            return np.full(data.shape[1:], 2, dtype=data.dtype)

        images = [np.full((2, 2), value) for value in (1, 2, 3)]
        result = imcombine(images, method='mean', dtype=dtype, sigma_clip=3,
                           sigma_cen_func=center, sigma_dev_func='mad_std')
        np.testing.assert_array_equal(result.data, 2)

    @pytest.mark.parametrize('dtype, value', [('int64', 2**53 + 1),
                                             ('uint64', 2**63)])
    def test_integer_reductions_follow_numpy_precision(self, dtype, value):
        data = np.full((1, 2, 2), value, dtype=dtype)
        result = imcombine(list(data), dtype=dtype, method='mean')
        expected = np.mean(data, axis=0).astype(dtype)
        np.testing.assert_array_equal(result.data, expected)

    def test_integer_image_count_uses_bookkeeping_dtype(self):
        result = imcombine([np.ones((1, 1))] * 128, dtype='int8', method='mean')
        np.testing.assert_array_equal(result.data, 1)

    @pytest.mark.parametrize('entry', ['class', 'function'])
    @pytest.mark.parametrize('disk_cache', [False, True])
    @pytest.mark.parametrize('method', ['sum', 'median', 'mean'])
    @pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64,
                                      'float32', np.dtype('float32')])
    def test_combination_preserves_dtype(self, tmp_path, entry, disk_cache, method, dtype):
        images = [FrameData(np.full((3, 4), value), unit='adu', uncertainty=error)
                  for value, error in [(1, 3), (3, 4)]]
        if entry == 'class':
            result = ImCombiner(dtype=dtype, use_disk_cache=disk_cache,
                                tmp_dir=str(tmp_path)).combine(images, method)
        else:
            # Keep the existing positional method and memory-limit arguments.
            result = imcombine(images, method, 1e6, dtype=dtype,
                               use_disk_cache=disk_cache, tmp_dir=str(tmp_path))
        expected_dtype = np.dtype(dtype)
        assert result.data.dtype == result.uncertainty.dtype == expected_dtype
        assert result.data.nbytes == 12 * expected_dtype.itemsize
        np.testing.assert_array_equal(result.data, 4 if method == 'sum' else 2)
        if method == 'sum':
            np.testing.assert_array_equal(result.uncertainty, 5)
        else:
            # Population scatter of [1, 3], divided by sqrt(2).
            expected = np.asarray(1 / np.sqrt(2), dtype=expected_dtype)
            np.testing.assert_allclose(result.uncertainty, expected,
                                       rtol=np.finfo(expected_dtype).eps)
        assert result.unit == images[0].unit
        np.testing.assert_array_equal(result.mask, False)
        for image, value, error in zip(images, (1, 3), (3, 4)):
            np.testing.assert_array_equal(image.data, value)
            np.testing.assert_array_equal(image.uncertainty, error)
            assert image.dtype == np.dtype('float64')
        assert not tmp_path.exists() or not list(tmp_path.iterdir())

    def test_default_dtype_remains_float64(self):
        result = imcombine([np.ones((2, 3), dtype=np.float32)], method='sum')
        assert result.data.dtype == result.uncertainty.dtype == np.dtype('float64')

    @pytest.mark.parametrize('option', ['FRAMEDATA_DISABLE_UNCERTAINTY'])
    def test_float32_with_disabled_errors(self, monkeypatch, option):
        from astropop.config import AstropopConfig
        images = [FrameData(np.full((3, 4), value), uncertainty=1) for value in (1, 3)]
        monkeypatch.setattr(AstropopConfig, option, True)
        result = imcombine(images, method='mean', dtype='float32')
        assert result.dtype == np.dtype('float32')
        assert result.uncertainty is None
        np.testing.assert_array_equal(result.data, 2)

    @pytest.mark.parametrize('method', ['sum', 'median', 'mean'])
    def test_combine_mask_median(self, method):
        # Just check if mask combining works.
        images = [None]*10
        shape = (10, 10)
        for i in range(10):
            mask = np.zeros(shape)
            # all (2, 5) elements are masked, so the result must be
            mask[2, 5] = 1
            images[i] = FrameData(np.ones(shape), unit='adu', mask=mask)

        # these points in result must no be masked
        images[5].mask[7, 7] = 1
        images[2].mask[7, 7] = 1
        images[8].mask[8, 8] = 1
        expect = np.zeros(shape, dtype=bool)
        expect[2, 5] = 1

        # with ImCombiner
        comb = ImCombiner()
        res = comb.combine(images, method)
        assert_equal(res.mask, expect)

        # with imcombine function
        assert_equal(imcombine(images, method).mask, expect)

    @pytest.mark.parametrize('method', ['sum', 'median', 'mean'])
    def test_combine_simple(self, method):
        values = [0.8, 1.0, 0.9, 1.1, 1.2, 1.1, 1.0, 0.8, 0.9]
        n = len(values)
        shape = (10, 10)
        images = [FrameData(np.ones(shape)*i, unit='adu') for i in values]
        if method == 'median':
            med = np.median(values)
            std = np.std(values)/np.sqrt(n)
        elif method == 'mean':
            med = np.mean(values)
            std = np.std(values)/np.sqrt(n)
        elif method == 'sum':
            med = np.sum(values)
            std = np.std(values)*np.sqrt(n)

        comb = ImCombiner()
        res1 = comb.combine(images, method=method)
        res2 = imcombine(images, method=method)
        for res in (res1, res2):
            assert_equal(res.data, np.ones(shape)*med)
            assert_almost_equal(res.uncertainty, np.ones(shape)*std)
            assert_equal(res.meta['astropop imcombine nimages'], n)
            assert_equal(res.meta['astropop imcombine method'], method)

    @pytest.mark.parametrize('method', ['sum', 'median', 'mean'])
    def test_combine_masked(self, method):
        values = [0.8, 1.0, 0.9, 1.1, 1.2, 1.1, 1.0, 0.8, 0.9]
        n = len(values)
        shape = (10, 10)
        images = [FrameData(np.ones(shape)*i, unit='adu') for i in values]

        images[0].mask_pixels((1, 0))
        images[0].data[1, 0] = 100000

        if method == 'median':
            med = np.ones(shape)*np.median(values)
            std = np.ones(shape)*np.std(values)/np.sqrt(n)
            med[1, 0] = np.median(values[1:])
            std[1, 0] = np.std(values[1:])/np.sqrt(n-1)
        elif method == 'mean':
            med = np.ones(shape)*np.mean(values)
            std = np.ones(shape)*np.std(values)/np.sqrt(n)
            med[1, 0] = np.mean(values[1:])
            std[1, 0] = np.std(values[1:])/np.sqrt(n-1)
        elif method == 'sum':
            med = np.ones(shape)*np.sum(values)
            std = np.ones(shape)*np.std(values)*np.sqrt(n)
            med[1, 0] = np.sum(values[1:])*float(n)/(n-1)
            std[1, 0] = np.std(values[1:])*np.sqrt(n-1)*float(n)/(n-1)

        comb = ImCombiner()
        res1 = comb.combine(images, method=method)
        res2 = imcombine(images, method=method)
        for res in (res1, res2):
            assert_almost_equal(res.data, med)
            assert_almost_equal(res.uncertainty, std)
            assert_equal(res.meta['astropop imcombine nimages'], n)
            assert_equal(res.meta['astropop imcombine method'], method)


class Test_ImCombiner_HeaderMerging():
    def create_images(self):
        images = []
        for i in range(30):
            meta = {'first_equal': 1,
                    'second_equal': 2,
                    'first_differ': i,
                    'second_differ': i // 2,
                    'third_differ': i % 3}
            images.append(FrameData(np.ones((10, 10)), unit='adu', meta=meta))

        return images

    def test_mergeheaders_conformance(self):
        comb = ImCombiner()
        comb.set_merge_header('no_merge')
        comb.set_merge_header('first')
        comb.set_merge_header('only_equal')
        comb.set_merge_header('selected_keys', ['key 1'])
        with pytest.raises(ValueError, match='not known.'):
            comb.set_merge_header('unkown method')
        with pytest.raises(ValueError, match='No key assigned'):
            comb.set_merge_header('selected_keys')

    def test_mergeheaders_default(self):
        # default is 'no_merge'
        images = self.create_images()
        comb = ImCombiner()
        res = comb.combine(images, method='sum')
        for i, v in zip(['nimages', 'method'], [30, 'sum']):
            assert_equal(res.meta[f'astropop imcombine {i}'], v)

    def test_mergeheaders_no_merge(self):
        images = self.create_images()
        expect = {'astropop imcombine nimages': 30,
                  'astropop imcombine method': 'sum'}
        comb = ImCombiner(merge_header='no_merge')
        res = comb.combine(images, method='sum')
        for k, v in expect.items():
            assert_equal(res.meta[k], v)

        # explicit setting
        comb = ImCombiner()
        comb.set_merge_header('no_merge')
        res = comb.combine(images, method='sum')
        for k, v in expect.items():
            assert_equal(res.meta[k], v)

        # func
        res = imcombine(images, method='sum', merge_header='no_merge')
        for k, v in expect.items():
            assert_equal(res.meta[k], v)

    def test_mergeheaders_first(self):
        images = self.create_images()
        expect = images[0].meta
        expect.update({'astropop imcombine nimages': 30,
                       'astropop imcombine method': 'sum'})

        comb = ImCombiner(merge_header='first')
        res = comb.combine(images, method='sum')
        for k, v in expect.items():
            assert_equal(res.meta[k], v)

        # explicit setting
        comb = ImCombiner()
        comb.set_merge_header('first')
        res = comb.combine(images, method='sum')
        for k, v in expect.items():
            assert_equal(res.meta[k], v)

        # func
        res = imcombine(images, method='sum', merge_header='first')
        for k, v in expect.items():
            assert_equal(res.meta[k], v)

    def test_mergeheaders_only_equal(self):
        images = self.create_images()
        expect = {'astropop imcombine nimages': 30,
                  'astropop imcombine method': 'sum',
                  'first_equal': 1,
                  'second_equal': 2}

        comb = ImCombiner(merge_header='only_equal')
        res = comb.combine(images, method='sum')
        for k, v in expect.items():
            assert_equal(res.meta[k], v)

        # explicit setting
        comb = ImCombiner()
        comb.set_merge_header('only_equal')
        res = comb.combine(images, method='sum')
        for k, v in expect.items():
            assert_equal(res.meta[k], v)

        # func
        res = imcombine(images, method='sum', merge_header='only_equal')
        for k, v in expect.items():
            assert_equal(res.meta[k], v)

    def test_mergeheaders_selected_keys(self):
        images = self.create_images()
        keys = ['first_equal', 'third_differ', 'first_differ']
        expect = {'astropop imcombine nimages': 30,
                  'astropop imcombine method': 'sum',
                  'first_equal': 1,
                  'third_differ': 0,
                  'first_differ': 0}

        comb = ImCombiner(merge_header='selected_keys', merge_header_keys=keys)
        res = comb.combine(images, method='sum')
        for k, v in expect.items():
            assert_equal(res.meta[k], v)

        # explicit setting
        comb = ImCombiner()
        comb.set_merge_header('selected_keys', keys=keys)
        res = comb.combine(images, method='sum')
        for k, v in expect.items():
            assert_equal(res.meta[k], v)

        # func
        res = imcombine(images, method='sum', merge_header='selected_keys',
                        merge_header_keys=keys)
        for k, v in expect.items():
            assert_equal(res.meta[k], v)


class TestRejectedUncertainties:
    @pytest.mark.parametrize('skip', [False, True])
    @pytest.mark.parametrize('normalize', [False, True])
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_rejected_samples_do_not_contribute_uncertainty(self, monkeypatch,
                                                          skip, normalize, dtype):
        from astropop.config import AstropopConfig

        frames = [FrameData(np.full((3, 4), value), uncertainty=error)
                  for value, error in [(1, 3), (2, 4), (100, 1000)]]
        monkeypatch.setattr(AstropopConfig, 'FRAMEDATA_DISABLE_UNCERTAINTY', skip)
        combiner = ImCombiner(dtype=dtype)
        combiner.set_minmax_clip(0, 10)
        result = combiner.combine(frames, 'sum', sum_normalize=normalize)
        factor = 1.5 if normalize else 1
        assert result.dtype == np.dtype(dtype)
        np.testing.assert_allclose(result.data, 3*factor)
        if skip:
            assert result.uncertainty is None
        else:
            np.testing.assert_allclose(result.uncertainty, 5*factor)
        np.testing.assert_array_equal(frames[-1]._unct, 1000)


class TestMaskedReductions:
    @staticmethod
    def _median_oracle(data, valid):
        result = np.empty(data.shape[1:], dtype=np.result_type(data.dtype, float))
        for pixel in np.ndindex(result.shape):
            values = sorted(v.item() for v, keep in
                            zip(data[(slice(None),) + pixel], valid[(slice(None),) + pixel]) if keep)
            if not values:
                result[pixel] = np.nan if data.dtype.kind == 'f' else 0
            elif len(values) % 2:
                result[pixel] = np.median(np.array(values, dtype=data.dtype))
            else:
                lower, upper = values[len(values) // 2 - 1:len(values) // 2 + 1]
                result[pixel] = np.mean(np.array([lower, upper], dtype=data.dtype))
        return result

    @pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int8',
                                      'uint8', 'int16', 'uint16', 'int32',
                                      'int64', 'uint64', '>i2', '>f4'])
    @pytest.mark.parametrize('samples', [1, 2, 7, 8])
    @pytest.mark.parametrize('block_pixels', [1, 4, 1024])
    def test_median_matches_scalar_oracle(self, dtype, samples, block_pixels):
        from astropop.image._combine import _masked_median
        rng = np.random.default_rng(301018)
        dt = np.dtype(dtype)
        data = rng.integers(0, 30, (samples, 3, 5)).astype(dt)
        if dt.kind in 'iu':
            data[:, 0, 1] = np.iinfo(dt).max
        else:
            data[0, 0, 1] = np.nan
        valid = rng.random(data.shape) > 0.3
        valid[:, 0, 0] = False
        valid[:, 0, 1] = True
        valid &= ~np.isnan(data)
        original, original_valid = data.copy(), valid.copy()
        result = _masked_median(data, valid, block_pixels=block_pixels)
        np.testing.assert_array_equal(result, self._median_oracle(data, valid))
        np.testing.assert_array_equal(data, original)
        np.testing.assert_array_equal(valid, original_valid)

    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    @pytest.mark.parametrize('method', ['mean', 'sum', 'std', 'mad_std'])
    def test_masked_statistics_match_independent_functions(self, dtype, method):
        from astropy.stats import mad_std
        from astropop.image._combine import _reduce
        rng = np.random.default_rng(123)
        data = rng.normal(100, 4, (31, 4, 5)).astype(dtype)
        valid = rng.random(data.shape) > 0.2
        valid[0] = True
        # Compare promoted calculations with an independent float64 reference.
        filled = data.astype(np.float64)
        filled[~valid] = np.nan
        if method == 'mad_std':
            expected = mad_std(filled, axis=0, ignore_nan=True)
        else:
            expected = getattr(np, 'nan' + method)(filled, axis=0)
        result = _reduce(data, valid, method, block_pixels=3)
        np.testing.assert_allclose(result, expected, rtol=5 * np.finfo(dtype).eps)

    def test_median_work_blocks_are_bounded(self, monkeypatch):
        import astropop.image._combine as module
        middle = module._middle_values
        sizes = []
        def record(work, accepted, count):
            sizes.append(len(work))
            assert work.flags.c_contiguous
            return middle(work, accepted, count)
        monkeypatch.setattr(module, '_middle_values', record)
        data = np.arange(300 * 17 * 19, dtype='float32').reshape(300, 17, 19)
        result = module._masked_median(data, np.ones(data.shape, dtype=bool), block_pixels=13)
        np.testing.assert_array_equal(result, np.median(data, axis=0))
        assert len(sizes) > 1 and max(sizes) <= 13


class TestCombinerSharedMask:
    @pytest.mark.parametrize('dtype', ['int16', 'float32', 'float64'])
    @pytest.mark.parametrize('method', ['sum', 'mean', 'median'])
    @pytest.mark.parametrize('normalize', [True, False])
    def test_fully_rejected_pixels_are_masked(self, dtype, method, normalize):
        frames = [FrameData([[1, 2]], uncertainty=3, mask=[[True, False]]) for _ in range(3)]
        result = imcombine(frames, dtype=dtype, method=method, sum_normalize=normalize)
        np.testing.assert_array_equal(result.mask, [[True, False]])
        if np.dtype(dtype).kind == 'f':
            assert np.isnan(result.data[0, 0])
            assert np.isnan(result.uncertainty[0, 0])
        else:
            assert result.data[0, 0] == result.uncertainty[0, 0] == 0

    @pytest.mark.parametrize('method', ['mean', 'median'])
    def test_scatter_methods_do_not_read_source_uncertainty(self, monkeypatch, method):
        frames = [FrameData([[value]], uncertainty=1000) for value in (1, 3)]
        prop = FrameData.uncertainty
        def read(frame):
            if any(frame is source for source in frames):
                pytest.fail('Read source uncertainties for a scatter estimator')
            return prop.fget(frame)
        monkeypatch.setattr(FrameData, 'uncertainty', property(read, prop.fset))
        result = imcombine(frames, method=method)
        np.testing.assert_array_equal(result.data, [[2]])
        np.testing.assert_allclose(result.uncertainty, [[1 / np.sqrt(2)]])

    @pytest.mark.parametrize('dtype', ['int16', 'float32', 'float64'])
    def test_public_minmax_uses_shared_data_and_error_mask(self, dtype):
        frames = [FrameData([[v]], uncertainty=e) for v, e in [(1, 3), (4, 4), (100, 1000)]]
        result = imcombine(frames, method='sum', dtype=dtype,
                           minmax_clip=(0, 10), sum_normalize=False)
        np.testing.assert_array_equal(result.data, [[5]])
        np.testing.assert_array_equal(result.uncertainty, [[5]])
        assert frames[-1].uncertainty[0, 0] == 1000

    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    def test_missing_error_samples_keep_existing_quadrature_behavior(self, dtype):
        frames = [FrameData([[1]], uncertainty=[[np.nan]]),
                  FrameData([[2]], uncertainty=3)]
        result = imcombine(frames, method='sum', dtype=dtype)
        np.testing.assert_array_equal(result.data, [[3]])
        np.testing.assert_array_equal(result.uncertainty, [[3]])

    @pytest.mark.parametrize('budget', [0, -1, np.nan, np.inf])
    def test_invalid_memory_budget(self, budget):
        with pytest.raises(ValueError, match='finite and positive'):
            ImCombiner(max_memory=budget)


class TestMaskedScatterEdges:
    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    def test_rejected_extremes_do_not_overflow_scatter(self, dtype):
        from astropop.image._combine import _reduce
        data = np.array([[[1]], [[3]], [[np.finfo(dtype).max]]], dtype=dtype)
        valid = np.array([[[True]], [[True]], [[False]]])
        with np.errstate(over='raise', invalid='raise'):
            result = _reduce(data, valid, 'std', block_pixels=1)
        np.testing.assert_array_equal(result, [[1]])

    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    def test_mad_ignores_nan_residuals_from_infinities(self, dtype):
        from astropy.stats import mad_std
        from astropop.image._combine import _reduce
        data = np.array([[[1, np.inf, np.inf]], [[2, np.inf, 3]]], dtype=dtype)
        with np.errstate(invalid='ignore'):
            # Astropy's NumPy backend warns on all-NaN residuals, whereas its
            # optional Bottleneck backend can return NaN silently. Compare the
            # numerical result without requiring either backend's warning.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='All-NaN slice',
                                        category=RuntimeWarning)
                expected = mad_std(data, axis=0, ignore_nan=True)
            result = _reduce(data, np.ones(data.shape, dtype=bool), 'mad_std', block_pixels=1)
        np.testing.assert_allclose(result, expected, equal_nan=True)


class TestPromotedIntegerReductions:
    @pytest.mark.parametrize('disk_cache', [False, True])
    @pytest.mark.parametrize('budget', [256, 65536])
    def test_uint16_bias_mean(self, tmp_path, disk_cache, budget):
        frames = [FrameData(np.full((3, 4), 1000, dtype=np.uint16), dtype='uint16')
                  for _ in range(150)]
        result = imcombine(frames, method='mean', dtype='uint16',
                           memory_limit=budget, use_disk_cache=disk_cache,
                           tmp_dir=str(tmp_path/'cache'))
        np.testing.assert_array_equal(result.data, 1000)
        np.testing.assert_array_equal(result.uncertainty, 0)
        assert result.dtype == result.uncertainty.dtype == np.dtype('uint16')
        assert all(frame.dtype == np.dtype('uint16') for frame in frames)
        assert not list(tmp_path.glob('cache/*'))

    @pytest.mark.parametrize('method', ['mean', 'median'])
    def test_mean_and_scatter_agree_with_float_reference(self, method):
        rng = np.random.default_rng(20261019)
        data = rng.integers(900, 1100, (150, 3, 4), dtype=np.uint16)
        frames = [FrameData(image, dtype='uint16') for image in data]
        result = imcombine(frames, method=method, dtype='uint16')
        expected = getattr(np, method)(data.astype(np.float64), axis=0)
        expected_error = np.std(data.astype(np.float64), axis=0)/np.sqrt(len(data))
        np.testing.assert_array_equal(result.data, expected.astype('uint16'))
        np.testing.assert_array_equal(result.uncertainty, expected_error.astype('uint16'))
        for frame, original in zip(frames, data):
            np.testing.assert_array_equal(frame.data, original)

    @pytest.mark.parametrize('method', ['mean', 'median'])
    def test_uint16_high_median_values(self, method):
        result = imcombine([np.full((2, 2), 60000, dtype='uint16')]*2,
                           method=method, dtype='uint16')
        np.testing.assert_array_equal(result.data, 60000)
        np.testing.assert_array_equal(result.uncertainty, 0)

    @pytest.mark.parametrize('deviation', ['std', 'mad_std'])
    def test_unsigned_clipping_matches_astropy(self, deviation):
        from astropy.stats import sigma_clip
        data = np.array([0, 1, 1, 2, 1000], dtype='uint16')[:, None, None]
        expected = sigma_clip(data.astype(float), sigma=2.5, cenfunc='median',
                              stdfunc=deviation, axis=0, maxiters=1)
        result = imcombine(list(data), method='mean', dtype='uint16',
                           sigma_clip=2.5, sigma_dev_func=deviation)
        np.testing.assert_array_equal(result.data, expected.mean(axis=0).astype('uint16'))

    @pytest.mark.parametrize('dtype, value', [('int64', 2**62),
                                             ('uint64', 2**63)])
    @pytest.mark.parametrize('method', ['mean', 'median'])
    def test_representable_large_integer_values(self, dtype, value, method):
        images = [np.full((2, 2), value, dtype=dtype)]*2
        result = imcombine(images, method=method, dtype=dtype)
        np.testing.assert_array_equal(result.data, value)
        np.testing.assert_array_equal(result.uncertainty, 0)


class TestRejectionFraction:
    @pytest.mark.parametrize('filters', [dict(minmax_clip=(None, 10)),
                                         dict(sigma_clip=3, sigma_dev_func='mad_std'),
                                         dict(minmax_clip=(None, 10), sigma_clip=3,
                                              sigma_dev_func='mad_std')])
    @pytest.mark.parametrize('method', ['mean', 'median', 'sum'])
    @pytest.mark.parametrize('budget', [128, 10**9])
    @pytest.mark.parametrize('disk_cache', [False, True])
    @pytest.mark.parametrize('disabled', [False, True])
    def test_fraction_pools_only_new_rejections(self, monkeypatch, tmp_path,
                                               filters, method, budget,
                                               disk_cache, disabled, caplog):
        from astropop.config import AstropopConfig as conf
        frames = [FrameData(np.full((2, 3), value, dtype=float), uncertainty=1)
                  for value in (1, 2, 3, 100)]
        frames[0].mask_pixels((0, 0))
        frames[1].data[0, 2] = np.nan
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_FLAGS', disabled)
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_UNCERTAINTY', disabled)
        with caplog.at_level('INFO', logger='astropop'):
            result = imcombine(frames, method=method, memory_limit=budget,
                               use_disk_cache=disk_cache, tmp_dir=str(tmp_path/'cache'),
                               **filters)
        # Six 100s are rejected. Of 24 inputs, one is NaN and one is masked
        # only when flags are enabled. Neither input exclusion is a rejection.
        eligible = 23 if disabled else 22
        summaries = [r.getMessage() for r in caplog.records
                     if r.getMessage().startswith('Combination filters rejected ')]
        assert summaries == [f'Combination filters rejected 6 of {eligible} '
                             f'initially valid samples ({100*6/eligible:.2f}%).']
        if disabled:
            assert result.flags is result.uncertainty is None
        assert not list(tmp_path.glob('cache/*'))

    def test_no_filter_has_no_rejection_summary(self, caplog):
        with caplog.at_level('INFO', logger='astropop'):
            imcombine([np.ones((2, 3))])
        assert 'Combination filters rejected' not in caplog.text

    @pytest.mark.parametrize('value, fraction', [(1, 0.0), (100, 1.0), (np.nan, 0.0)])
    def test_zero_all_and_empty(self, value, fraction, caplog):
        with caplog.at_level('INFO', logger='astropop'):
            imcombine([np.full((2, 3), value)], minmax_clip=(0, 10))
        eligible = 0 if np.isnan(value) else 6
        rejected = eligible if value == 100 else 0
        assert (f'Combination filters rejected {rejected} of {eligible} '
                f'initially valid samples ({100*fraction:.2f}%).') in caplog.text

    def test_weighted_across_unequal_chunks(self, monkeypatch, caplog):
        from astropop.image.imcombine import ImCombiner
        frames = [FrameData(np.ones((2, 3))) for _ in range(2)]
        frames[0].data[:, 0] = 100
        combiner = ImCombiner()
        # The first tile has 2/4 rejections; the second has 0/8. Their simple
        # average (1/4) is wrong: count samples globally to obtain 2/12.
        monkeypatch.setattr(combiner, '_chunk_slices', lambda errors: iter([
            (slice(0, 2), slice(0, 1)), (slice(0, 2), slice(1, 3))]))
        combiner.set_minmax_clip(None, 10)
        with caplog.at_level('INFO', logger='astropop'):
            combiner.combine(frames, 'mean')
        assert ('Combination filters rejected 2 of 12 initially valid '
                'samples (16.67%).') in caplog.text
        # Counts are local to each call, including reuse after a previous run.
        caplog.clear()
        combiner.set_minmax_clip(None, 1000)
        with caplog.at_level('INFO', logger='astropop'):
            combiner.combine(frames, 'mean')
        assert ('Combination filters rejected 0 of 12 initially valid '
                'samples (0.00%).') in caplog.text
