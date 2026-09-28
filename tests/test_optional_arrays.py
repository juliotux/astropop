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


@pytest.mark.parametrize('option', ['IMARITH_SKIP_UNCERTAINTY', 'FRAMEDATA_DISABLE_UNCERTAINTY'])
@pytest.mark.parametrize('operation', ['+', '-', '*', '/', '//', '%', '**'])
@pytest.mark.parametrize('inplace', [False, True])
def test_arithmetic_without_error_calculations(monkeypatch, option, operation, inplace):
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
def test_skip_arithmetic_units(monkeypatch, kind):
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




