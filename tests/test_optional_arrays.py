"""Optional storage and propagation preserve image values and units."""

import copy
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.nddata import CCDData, StdDevUncertainty

from astropop.config import AstropopConfig as conf
from astropop.framedata import FrameData, PixelMaskFlags, read_framedata
from astropop.math.physical import QFloat


class TestOptionalArrays:
    class UnreadableArray:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Disabled data must not be converted')

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

