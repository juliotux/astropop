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
from astropop.math.physical import QFloat


class TestOptionalArrays:
    class UnreadableArray:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Disabled data must not be converted')

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
