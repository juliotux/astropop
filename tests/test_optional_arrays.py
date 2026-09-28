"""Optional storage and propagation must preserve data and unit semantics."""

import copy
import importlib

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty

from astropop.config import AstropopConfig as conf
from astropop.framedata import FrameData, PixelMaskFlags, read_framedata
from astropop.image.imarith import imarith
from astropop.image.imcombine import ImCombiner, imcombine
from astropop.image.processing import trim_image
from astropop.image.register import CrossCorrelationRegister
from astropop.math.physical import QFloat


class UnreadableArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Disabled data must not be converted or allocated')


@pytest.mark.parametrize('disable_uncertainty', [False, True])
@pytest.mark.parametrize('disable_flags', [False, True])
@pytest.mark.parametrize('memmap', [False, True])
def test_storage_switches(monkeypatch, tmp_path, disable_uncertainty,
                          disable_flags, memmap):
    monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_UNCERTAINTY', disable_uncertainty)
    monkeypatch.setattr(conf, 'FRAMEDATA_DISABLE_FLAGS', disable_flags)
    data = np.ones((4, 5))
    frame = FrameData(data, uncertainty=UnreadableArray() if disable_uncertainty else 2,
                     flags=UnreadableArray() if disable_flags else np.zeros(data.shape),
                     mask=UnreadableArray() if disable_flags else np.ones(data.shape),
                     cache_folder=tmp_path if memmap else None,
                     use_memmap_backend=memmap)
    assert (frame.uncertainty is None) == disable_uncertainty
    assert (frame.flags is None) == disable_flags
    assert (frame.mask is None) == disable_flags
    frame.uncertainty = UnreadableArray() if disable_uncertainty else 3
    frame.flags = UnreadableArray() if disable_flags else np.zeros(data.shape)
    frame.mask_pixels((0, 0))
    frame.add_flags(PixelMaskFlags.BAD, (1, 1))
    assert (frame.mask is None) == disable_flags
    if memmap:
        assert len(list(tmp_path.iterdir())) == 1 + (not disable_flags) + (not disable_uncertainty)
    else:
        assert frame.cache is None
    for cloned in [frame.copy(), copy.deepcopy(frame), frame.astype('f4')]:
        np.testing.assert_array_equal(cloned.data, data)
        assert (cloned.uncertainty is None) == disable_uncertainty
        assert (cloned.flags is None) == disable_flags
    np.testing.assert_array_equal(frame.get_masked_data()[1:], data[1:])


def test_options_are_not_retroactive(monkeypatch):
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
def test_optional_io(monkeypatch, tmp_path, kind):
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




