# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Regression tests for registration restricted to an image window."""

import numpy as np
import pytest
from skimage.transform import AffineTransform

from astropop.framedata import FrameData
from astropop.image.register import (AsterismRegister, CrossCorrelationRegister,
                                    compute_shift_list,
                                    register_framedata_list)


@pytest.fixture
def window_images():
    rng = np.random.default_rng(123)
    reference = np.zeros((100, 120))
    reference[35:55, 45:65] = rng.uniform(1, 10, (20, 20))
    moving = np.roll(reference, (3, -4), axis=(0, 1))
    # A much brighter, stationary region would dominate full-image correlation.
    reference[:20, :20] = moving[:20, :20] = rng.uniform(100, 1000, (20, 20))
    return reference, moving, (slice(25, 70), slice(30, 85))


def test_window_excludes_conflicting_pixels(window_images):
    reference, moving, window = window_images
    full = CrossCorrelationRegister().compute_transform(reference, moving)
    np.testing.assert_allclose(full.translation, [0, 0])
    reg = CrossCorrelationRegister(window=window)
    aligned, mask, tform = reg.register_image(reference, moving, cval=0)
    np.testing.assert_allclose(tform.translation, [-4, 3])
    assert aligned.shape == mask.shape == reference.shape
    np.testing.assert_allclose(aligned[window], reference[window], atol=1e-10)
    # Pixels outside the estimation window are also shifted.
    np.testing.assert_allclose(aligned[5:15, 5:15], moving[8:18, 1:11])


@pytest.mark.parametrize('inplace', [False, True])
def test_window_frame_lists(window_images, inplace):
    reference, moving, window = window_images
    frames = [FrameData(reference), FrameData(moving)]
    shifts = compute_shift_list(frames, ref_image=1, window=window)
    np.testing.assert_allclose(shifts, [[4, -3], [0, 0]])
    aligned = register_framedata_list(frames, window=window, inplace=inplace)
    assert aligned[1].shape == reference.shape
    assert (aligned[1] is frames[1]) == inplace
    np.testing.assert_allclose(aligned[1].data[window], reference[window],
                               atol=1e-10)
    assert aligned[1].meta['astropop registration_shift_x'] == -4
    assert aligned[1].meta['astropop registration_shift_y'] == 3


def test_window_crops_masks_and_restores_coordinates(monkeypatch):
    reg = AsterismRegister(window=(slice(-60, -10), slice(30, None)))
    image = np.zeros((100, 120))
    mask = np.zeros_like(image, dtype=bool)
    mask[45, 35] = True
    local = AffineTransform(rotation=0.2, translation=(3, -2))

    def compute(image1, image2, mask1, mask2):
        assert image1.shape == image2.shape == (50, 90)
        np.testing.assert_array_equal(mask1, mask[40:90, 30:])
        assert mask2 is None
        return local

    monkeypatch.setattr(reg, '_compute_transform', compute)
    actual = reg.compute_transform(image, image, mask1=mask)
    points = np.array([[32, 42], [60, 70], [100, 80]])
    np.testing.assert_allclose(actual(points), local(points - [30, 40]) + [30, 40])


def test_asterism_window_rotation():
    rng = np.random.default_rng(45)
    points = rng.uniform([90, 80], [300, 250], (25, 2))
    expected = AffineTransform(rotation=0.04, translation=(5, -3))
    yy, xx = np.indices((320, 380))

    def render(positions):
        image = rng.normal(100.0, 0.5, xx.shape)
        for index, (x, y) in enumerate(positions):
            image += (1000 + 100*index) * np.exp(
                -((xx-x)**2 + (yy-y)**2) / (2*1.5**2))
        # Identical bright sources outside the window must not enter matching.
        image[5:15, 5:15] = 1e7
        return image

    reg = AsterismRegister(window=(slice(35, 291), slice(45, 365)))
    actual = reg.compute_transform(render(points), render(expected(points)))
    np.testing.assert_allclose(actual(points), expected(points), atol=0.2)


@pytest.mark.parametrize('window', [1, (slice(1),), (1, 2),
                                   (slice(None, None, 2), slice(None)),
                                   (slice(None), slice(None, None, -1))])
@pytest.mark.parametrize('cls', [CrossCorrelationRegister, AsterismRegister])
def test_invalid_window(window, cls):
    with pytest.raises(ValueError, match='window'):
        cls(window=window)


@pytest.mark.parametrize('window', [(slice(5, 5), slice(None)),
                                   (slice(20, 30), slice(None))])
@pytest.mark.parametrize('method', ['compute_transform', 'register_image'])
def test_empty_window(window, method):
    reg = CrossCorrelationRegister(window=window)
    with pytest.raises(ValueError, match='non-empty'):
        getattr(reg, method)(np.ones((10, 10)), np.ones((10, 10)))


def test_window_incompatible_shapes():
    reg = CrossCorrelationRegister(window=(slice(None), slice(None)))
    with pytest.raises(ValueError, match='matching 2D shapes'):
        reg.compute_transform(np.ones((10, 10)), np.ones((11, 10)))


def test_window_fourier_rejected():
    with pytest.raises(ValueError, match='real-space'):
        CrossCorrelationRegister(window=(slice(None), slice(None)),
                                 space='fourier')


@pytest.mark.parametrize('window', [None, (slice(None), slice(None))])
def test_full_window_matches_default(window_images, window):
    reference, moving, _ = window_images
    actual = CrossCorrelationRegister(window=window).compute_transform(
        reference, moving)
    expected = CrossCorrelationRegister().compute_transform(reference, moving)
    np.testing.assert_allclose(actual.params, expected.params)


def test_window_preserves_full_uncertainty(window_images):
    reference, moving, window = window_images
    uncertainty = np.arange(moving.size, dtype=float).reshape(moving.shape)
    frame = FrameData(moving, uncertainty=uncertainty)
    reg = CrossCorrelationRegister(window=window)
    aligned = reg.register_framedata(FrameData(reference), frame, cval=0)
    assert aligned.uncertainty.shape == moving.shape
    np.testing.assert_allclose(aligned.uncertainty[5:15, 5:15],
                               uncertainty[8:18, 1:11])
    assert np.all(aligned.mask[:, :4])
    assert np.all(aligned.mask[-3:, :])
