"""Optional storage and propagation preserve image values and units."""

import copy
import importlib
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.nddata import CCDData, StdDevUncertainty

from astropop.config import AstropopConfig as conf
from astropop.framedata import FrameData, PixelMaskFlags, read_framedata
from astropop.image.imarith import imarith
from astropop.image.imcombine import ImCombiner, imcombine
from astropop.image.processing import trim_image
from astropop.image.register import CrossCorrelationRegister
from astropop.math.physical import QFloat


class TestOptionalArrays:
    class UnreadableArray:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Disabled data must not be converted')

    @pytest.mark.parametrize('memmap', [False, True])
    def test_uncertainty_assignment_keeps_existing_dtype(self, tmp_path, memmap):
        frame = FrameData(np.ones((2, 3)), dtype='f4',
                         cache_folder=str(tmp_path / 'frame'),
                         use_memmap_backend=memmap)
        # Existing setters expand Python scalars with their own dtype, while
        # explicitly supplied arrays are converted to the image dtype.
        frame.uncertainty = 0.123456789012345
        assert frame.uncertainty.dtype == np.dtype('f8')
        frame.uncertainty = np.full(frame.shape, 0.25)
        assert frame.uncertainty.dtype == np.dtype('f4')
        np.testing.assert_array_equal(frame.uncertainty, 0.25)

    @pytest.mark.parametrize('operation, expected, unit', [
        ('+', 2.5, u.m), ('-', 1.5, u.m),
        ('*', 100, u.m*u.cm), ('/', 0.04, u.m/u.cm),
        ('//', 0, u.m/u.cm), ('%', 2, u.m),
    ])
    def test_skipping_keeps_existing_unit_conventions(self, monkeypatch,
                                                      operation, expected, unit):
        # Floor division and remainder intentionally retain QFloat's existing
        # raw-nominal conventions; changing their unit policy is separate work.
        frame = FrameData(np.full((2, 3), 2), unit='m', uncertainty=0.1)
        monkeypatch.setattr(conf, 'IMARITH_SKIP_UNCERTAINTY', True)
        result = imarith(frame, 50*u.cm, operation)
        np.testing.assert_allclose(result.data, expected)
        assert result.unit == unit

    @pytest.mark.parametrize('exponent', [np.array([2, 2]), np.array(2),
                                         QFloat(np.array(2)), 2*u.m])
    def test_skipped_power_keeps_exponent_validation(self, monkeypatch, exponent):
        monkeypatch.setattr(conf, 'IMARITH_SKIP_UNCERTAINTY', True)
        with pytest.raises(ValueError):
            imarith(FrameData(np.ones((2, 2))), exponent, '**')

    def test_skipped_integer_inputs_do_not_overflow(self, monkeypatch):
        monkeypatch.setattr(conf, 'IMARITH_SKIP_UNCERTAINTY', True)
        result = imarith(np.array([[50_000]], dtype=np.int32), 2, '**')
        np.testing.assert_allclose(result.data, 2_500_000_000)

    def test_skipped_arithmetic_rejects_none(self, monkeypatch):
        monkeypatch.setattr(conf, 'IMARITH_SKIP_UNCERTAINTY', True)
        with pytest.raises(TypeError):
            imarith(FrameData(np.ones((2, 2))), None, '+')

    @pytest.mark.parametrize('operation, expected', [('//', 2), ('%', 0)])
    def test_skipped_scalar_arithmetic_without_optional_arrays(self, monkeypatch,
                                                               operation, expected):
        monkeypatch.setattr(conf, 'IMARITH_SKIP_UNCERTAINTY', True)
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_FLAGS', True)
        result = imarith(4, 2, operation)
        np.testing.assert_allclose(result.data, expected)
        assert result.uncertainty is result.flags is None

    @pytest.mark.parametrize('memmap', [False, True])
    @pytest.mark.parametrize('inplace', [False, True])
    @pytest.mark.parametrize('disabled', [False, True])
    def test_registration_preserves_flags_and_optional_errors(self, monkeypatch,
                                                              tmp_path, memmap,
                                                              inplace, disabled):
        reference = np.zeros((24, 24))
        reference[8, 8] = 100
        moving = FrameData(np.roll(reference, (2, 3), axis=(0, 1)),
                           uncertainty=1, cache_folder=str(tmp_path / 'moving'),
                           use_memmap_backend=memmap)
        moving.add_flags(PixelMaskFlags.BAD, (12, 13))
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_UNCERTAINTY', disabled)
        result = CrossCorrelationRegister(window=(slice(3, 21),)*2).register_framedata(
            FrameData(reference), moving, inplace=inplace)
        assert (result is moving) is inplace
        np.testing.assert_allclose(result.data, reference, atol=1e-10)
        assert result.mask_flags(PixelMaskFlags.BAD)[10, 10]
        assert (result.uncertainty is None) is disabled
        if not inplace:
            assert moving.mask_flags(PixelMaskFlags.BAD)[12, 13]
            np.testing.assert_array_equal(moving.uncertainty, 1)

    @pytest.mark.parametrize('kind', ['frame', 'qfloat', 'ccd'])
    def test_arithmetic_never_reads_skipped_uncertainty(self, monkeypatch,
                                                       caplog, kind):
        data = np.full((2, 3), 4.0)
        if kind == 'frame':
            operand = FrameData(data, unit='m', uncertainty=1)
        elif kind == 'qfloat':
            operand = QFloat(data, data, u.m)
        else:
            operand = CCDData(data, unit='m', uncertainty=StdDevUncertainty(data))

        def unreadable(*args, **kwargs):
            pytest.fail('Read an input uncertainty while propagation was skipped')

        monkeypatch.setattr(conf, 'IMARITH_SKIP_UNCERTAINTY', True)
        original = type(operand).uncertainty
        monkeypatch.setattr(type(operand), 'uncertainty',
                            property(unreadable, original.fset))
        from astropop.logger import logger
        with caplog.at_level('DEBUG', logger=logger.name):
            result = imarith(operand, 100*u.cm, '+')
        np.testing.assert_allclose(result.data, 5)
        assert result.unit == u.m

    def test_arithmetic_failure_removes_log_handler(self, monkeypatch):
        from astropop.logger import logger
        handlers = list(logger.handlers)
        monkeypatch.setattr(conf, 'IMARITH_SKIP_UNCERTAINTY', True)
        with pytest.raises(u.UnitsError):
            imarith(FrameData(np.ones((2, 3)), unit='m'), 1*u.s, '+')
        assert logger.handlers == handlers

    def test_combine_does_not_read_disabled_input_arrays(self, monkeypatch):
        frames = [FrameData(np.full((3, 4), value), uncertainty=1)
                  for value in (1, 2, 3)]

        def unreadable(*args):
            pytest.fail('Read disabled input storage while combining')

        monkeypatch.setattr(conf, 'IMARITH_SKIP_UNCERTAINTY', True)
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_FLAGS', True)
        for name in ('uncertainty', 'flags'):
            descriptor = getattr(FrameData, name)
            monkeypatch.setattr(FrameData, name,
                                property(unreadable, descriptor.fset))
        result = imcombine(frames, method='sum')
        np.testing.assert_array_equal(result.data, 6)
        assert result._unct is result._flags is None

    def test_combine_partial_load_releases_mapped_arrays(self, monkeypatch,
                                                        tmp_path):
        frames = [FrameData(np.ones((3, 4)), uncertainty=1) for _ in range(2)]
        enable = FrameData.enable_memmap

        def fail_second_image(frame, *args, **kwargs):
            enable(frame, *args, **kwargs)
            if kwargs.get('filename') == 'image_1':
                raise RuntimeError('Partial load failure')

        monkeypatch.setattr(FrameData, 'enable_memmap', fail_second_image)
        combiner = ImCombiner(use_disk_cache=True, tmp_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match='Partial load failure'):
            combiner.combine(frames, 'sum')
        assert combiner._images == []
        assert combiner._buffer is combiner._unct_bf is None
        assert not list(tmp_path.glob('*'))

    @pytest.mark.parametrize('name', ['flags', 'uncertainty'])
    def test_clear_mapped_array_without_reading(self, monkeypatch, tmp_path,
                                               name):
        frame = FrameData(np.arange(12).reshape(3, 4), uncertainty=2,
                         cache_folder=str(tmp_path / 'frame'),
                         use_memmap_backend=True)
        filename = getattr(frame, name).filename
        import astropop.framedata.framedata as module
        delete = module.delete_array_memmap

        def delete_without_reading(array, read=True, remove=False):
            assert read is False
            return delete(array, read=read, remove=remove)

        monkeypatch.setattr(module, 'delete_array_memmap', delete_without_reading)
        setattr(frame, name, None)
        assert getattr(frame, name) is None
        assert not Path(filename).exists()
        np.testing.assert_array_equal(frame.data, np.arange(12).reshape(3, 4))

    def test_missing_flags_allocate_on_demand(self):
        frame = FrameData(np.arange(6).reshape(2, 3))
        frame.flags = None
        assert frame.mask is None
        data = frame.get_masked_data()
        np.testing.assert_array_equal(data, frame.data)
        assert not np.shares_memory(data, frame.data)
        frame.add_flags(PixelMaskFlags.BAD, (0, 0))
        assert frame.flags[0, 0] == PixelMaskFlags.BAD.value
        frame.mask_pixels((1, 2))
        assert frame.mask[1, 2]

    @pytest.mark.parametrize('kind', ['ccd', 'qfloat', 'fits'])
    def test_disabled_input_arrays_are_not_read(self, monkeypatch, kind):
        def unreadable(*args):
            pytest.fail('Read an array whose storage is disabled')

        data = np.ones((2, 3))
        if kind == 'ccd':
            source = CCDData(data, unit='adu',
                             uncertainty=StdDevUncertainty(data))
            monkeypatch.setattr(CCDData, 'uncertainty', property(unreadable))
            monkeypatch.setattr(CCDData, 'mask', property(unreadable))
        elif kind == 'qfloat':
            source = QFloat(data, data, u.adu)
            monkeypatch.setattr(QFloat, 'uncertainty', property(unreadable))
        else:
            from astropy.io import fits
            source = FrameData(data, uncertainty=1).to_hdu()
            monkeypatch.setattr(fits.ImageHDU, 'data', property(unreadable))
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_UNCERTAINTY', True)
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_FLAGS', True)
        result = read_framedata(source)
        assert result.uncertainty is result.flags is None
        np.testing.assert_array_equal(result.data, data)

    @pytest.mark.parametrize('disable_uncertainty', [False, True])
    @pytest.mark.parametrize('disable_flags', [False, True])
    @pytest.mark.parametrize('memmap', [False, True])
    def test_storage_switches(self, monkeypatch, tmp_path, disable_uncertainty,
                              disable_flags, memmap):
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_UNCERTAINTY', disable_uncertainty)
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_FLAGS', disable_flags)
        data = np.ones((4, 5))
        frame = FrameData(data, uncertainty=self.UnreadableArray() if disable_uncertainty else 2,
                         flags=self.UnreadableArray() if disable_flags else np.zeros(data.shape),
                         mask=self.UnreadableArray() if disable_flags else np.ones(data.shape),
                         cache_folder=str(tmp_path) if memmap else None,
                         use_memmap_backend=memmap)
        assert (frame.uncertainty is None) == disable_uncertainty
        assert (frame.flags is None) == disable_flags
        assert (frame.mask is None) == disable_flags
        frame.uncertainty = self.UnreadableArray() if disable_uncertainty else 3
        frame.flags = self.UnreadableArray() if disable_flags else np.zeros(data.shape)
        frame.mask_pixels((0, 0))
        frame.add_flags(PixelMaskFlags.BAD, (1, 1))
        assert (frame.mask is None) == disable_flags
        if memmap:
            assert len(list(tmp_path.iterdir())) == 1 + (not disable_flags) + (not disable_uncertainty)
        for cloned in [frame.copy(), copy.deepcopy(frame), frame.astype('f4')]:
            np.testing.assert_array_equal(cloned.data, data)
            assert (cloned.uncertainty is None) == disable_uncertainty
            assert (cloned.flags is None) == disable_flags
        np.testing.assert_array_equal(frame.get_masked_data()[1:], data[1:])


    def test_options_are_not_retroactive(self, monkeypatch):
        frame = FrameData(np.ones((2, 2)), uncertainty=3)
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_UNCERTAINTY', True)
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_FLAGS', True)
        assert frame.uncertainty is not None
        assert frame.flags is not None
        clone = frame.copy()
        assert clone.uncertainty is clone.flags is None
        frame.uncertainty = 4
        frame.mask_pixels((0, 0))
        assert frame.uncertainty is frame.flags is None


    @pytest.mark.parametrize('kind', ['fits', 'ccd'])
    def test_optional_io(self, monkeypatch, tmp_path, kind):
        original = FrameData(np.ones((3, 4)), unit='adu', uncertainty=2,
                             mask=np.ones((3, 4), dtype=bool))
        source = original.to_hdu() if kind == 'fits' else original.to_ccddata()
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_FLAGS', True)
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_UNCERTAINTY', True)
        result = read_framedata(source)
        assert result.flags is result.mask is result.uncertainty is None
        ccd = result.to_ccddata()
        assert ccd.mask is ccd.uncertainty is None
        hdul = result.to_hdu()
        assert len(hdul) == 1
        filename = tmp_path/'optional.fits'
        result.write(filename)
        reread = read_framedata(filename)
        assert reread.flags is reread.uncertainty is None
        np.testing.assert_array_equal(reread.data, original.data)

    @pytest.mark.parametrize('option', ['IMARITH_SKIP_UNCERTAINTY', 'FRAMEDATA_DISABLE_UNCERTAINTY'])
    @pytest.mark.parametrize('operation', ['+', '-', '*', '/', '//', '%', '**'])
    @pytest.mark.parametrize('inplace', [False, True])
    def test_arithmetic_without_error_calculations(self, monkeypatch, option, operation, inplace):
        first = FrameData(np.full((3, 4), 4.0), uncertainty=0.3)
        second = QFloat(2, 0.1)
        expected = imarith(first, second, operation)
        module = importlib.import_module('astropop.image.imarith')
        monkeypatch.setattr(conf, option, True)
        monkeypatch.setattr(module, '_arith', lambda *args: pytest.fail('QFloat propagation called'))
        actual = imarith(first, second, operation, inplace=inplace)
        assert (actual is first) == inplace
        assert actual.uncertainty is None
        np.testing.assert_allclose(actual.data, expected.data)
        assert actual.unit == expected.unit
        if not inplace:
            assert first.uncertainty is not None


    @pytest.mark.parametrize('kind', ['quantity', 'frame', 'ccd', 'qfloat'])
    def test_skip_arithmetic_units(self, monkeypatch, kind):
        monkeypatch.setattr(conf, 'IMARITH_SKIP_UNCERTAINTY', True)
        first = FrameData(np.full((2, 3), 2), unit='m', uncertainty=0.2)
        values = np.full((2, 3), 100)
        second = {'quantity': values*u.cm,
                  'frame': FrameData(values, unit='cm', uncertainty=1),
                  'ccd': CCDData(values, unit='cm', uncertainty=StdDevUncertainty(values)),
                  'qfloat': QFloat(values, np.ones(values.shape), 'cm')}[kind]
        result = imarith(first, second, '+')
        np.testing.assert_allclose(result.data, 3)
        assert result.unit == u.m
        assert result.uncertainty is None
        with pytest.raises(u.UnitsError):
            imarith(first, np.ones((2, 3))*u.s, '+')

    @pytest.mark.parametrize('disable_flags', [False, True])
    @pytest.mark.parametrize('method', ['sum', 'mean', 'median'])
    @pytest.mark.parametrize('option', ['IMARITH_SKIP_UNCERTAINTY', 'FRAMEDATA_DISABLE_UNCERTAINTY'])
    @pytest.mark.parametrize('disk_cache', [False, True])
    def test_combine_without_uncertainty(self, monkeypatch, tmp_path, method, option,
                                         disable_flags, disk_cache):
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_FLAGS', disable_flags)
        frames = [FrameData(np.full((4, 5), i), unit='adu', uncertainty=0.5)
                  for i in [1, 2, 6]]
        frames[0].mask_pixels((0, 0))
        expected = imcombine(frames, method=method)
        monkeypatch.setattr(conf, option, True)
        module = importlib.import_module('astropop.image.imcombine')
        monkeypatch.setitem(module._funcs, 'std', lambda *a, **k: pytest.fail('Uncertainty std calculated'))
        comb = ImCombiner(max_memory=200, use_disk_cache=disk_cache, tmp_dir=str(tmp_path))
        comb._load_images(frames)
        assert all(f.uncertainty is None for f in comb._images)
        for _, uncertainty_buffer, _ in comb._chunk_yielder(method):
            assert uncertainty_buffer is None
        actual = comb.combine(frames, method)
        np.testing.assert_allclose(actual.data, expected.data)
        assert actual.unit == expected.unit
        assert actual.uncertainty is None
        assert (actual.mask is None) == disable_flags
        assert not list(tmp_path.glob('*'))
        assert all(f.uncertainty is not None for f in frames)


    @pytest.mark.parametrize('skip', [False, True])
    def test_combine_failure_releases_buffers(self, monkeypatch, tmp_path, skip):
        monkeypatch.setattr(conf, 'IMARITH_SKIP_UNCERTAINTY', skip)
        comb = ImCombiner(use_disk_cache=True, tmp_dir=str(tmp_path))
        with pytest.raises(ValueError, match='shape'):
            comb.combine([FrameData(np.ones((3, 4))),
                          FrameData(np.ones((4, 4)))], 'sum')
        assert not list(tmp_path.glob('*'))
        assert comb._images == []
        assert comb._buffer is comb._unct_bf is None
        result = comb.combine([FrameData(np.ones((3, 4)))], 'mean')
        np.testing.assert_array_equal(result.data, 1)


    def test_skip_uncertainty_keeps_rejection(self, monkeypatch):
        frames = [FrameData(np.full((3, 4), value), uncertainty=1)
                  for value in [1, 1, 1, 1, 100]]
        monkeypatch.setattr(conf, 'IMARITH_SKIP_UNCERTAINTY', True)
        combiner = ImCombiner()
        combiner.set_minmax_clip(0, 5)
        result = combiner.combine(frames, method='mean')
        np.testing.assert_array_equal(result.data, 1)
        assert result.uncertainty is None

    def test_disabled_flags_arithmetic_and_registration(self, monkeypatch):
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_FLAGS', True)
        first = FrameData(np.arange(100).reshape(10, 10), uncertainty=1)
        first.mask_pixels((1, 1))
        second = imarith(first, 2, '*')
        assert second.flags is second.mask is None
        registered = CrossCorrelationRegister().register_framedata(first, second)
        assert registered.flags is registered.mask is None
        trimmed = trim_image(registered, slice(2, 8), slice(3, 9))
        assert trimmed.flags is trimmed.mask is None
        assert trimmed.shape == (6, 6)


    @pytest.mark.parametrize('inplace', [False, True])
    @pytest.mark.parametrize('disabled', [False, True])
    def test_trim_preserves_absent_uncertainty(self, monkeypatch, inplace, disabled):
        frame = FrameData(np.arange(30).reshape(5, 6))
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_UNCERTAINTY', disabled)
        monkeypatch.setattr(FrameData, 'get_uncertainty',
                            lambda *a, **k: pytest.fail('Allocated missing uncertainty'))
        result = trim_image(frame, slice(1, 4), slice(2, 5), inplace=inplace)
        assert result.uncertainty is None
        np.testing.assert_array_equal(result.data, np.arange(30).reshape(5, 6)[2:5, 1:4])


    @pytest.mark.parametrize('inplace', [False, True])
    @pytest.mark.parametrize('operation', ['trim', 'register'])
    def test_processing_does_not_read_disabled_uncertainty(self, monkeypatch, inplace, operation):
        frame = FrameData(np.arange(100).reshape(10, 10), uncertainty=2)
        monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_UNCERTAINTY', True)
        monkeypatch.setattr(FrameData, 'get_uncertainty',
                            lambda *a, **k: pytest.fail('Read disabled uncertainty'))
        if operation == 'trim':
            result = trim_image(frame, slice(2, 8), slice(1, 7), inplace=inplace)
        else:
            register = CrossCorrelationRegister()
            result = register.register_framedata(frame, frame, inplace=inplace)
        assert result.uncertainty is None
        if not inplace:
            np.testing.assert_array_equal(frame.uncertainty, 2)
