# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from skimage import transform
from astropop.image.register import AsterismRegister, \
                                    CrossCorrelationRegister, \
                                    register_framedata_list
from astropop.framedata import FrameData
from astropop.testing import assert_almost_equal, assert_equal, \
                             assert_is, assert_is_not

from .test_detection import gen_position_flux, gen_image


def gen_positions_transformed(x, y, flux, dx, dy, limits,
                              rotation=None, rotation_center=None):
    """Generate translated positions."""
    x, y = x+dx,  y+dy

    if rotation is not None:
        rotation_center = rotation_center or np.array(limits)/2
        cx, cy = rotation_center
        theta = np.deg2rad(rotation)
        nx = cx + (x-cx)*np.cos(theta) - (y-cy)*np.sin(theta)
        ny = cy + (x-cx)*np.sin(theta) + (y-cy)*np.cos(theta)
        x, y = nx, ny

    # ensure all positions are inside the image
    mask = x>=0
    mask &= x<=limits[0]
    mask &= y>=0
    mask &= y<=limits[1]
    where = np.where(mask)

    return x[where], y[where], flux[where]


class Test_AsterismRegister:
    @pytest.mark.parametrize('shift', [(25, 32), (-12, 5), (23.42, 12.43)])
    def test_compute_transform_translation(self, shift):
        size = (1024, 1024)
        sky = 800
        n = 100
        rdnoise = 5
        x, y, f = gen_position_flux(np.array(size)+80, n, 1e4, 4e6)
        x -= 40
        y -= 40
        sx, sy = shift

        x1, y1, flux1 = gen_positions_transformed(x, y, f, 0, 0, size)
        im1 = gen_image(size, x1, y1, flux1,
                        sky, rdnoise, sigma=2)

        x2, y2, flux2 = gen_positions_transformed(x, y, f, sx, sy, size)
        im2 = gen_image(size, x2, y2, flux2,
                        sky, rdnoise, sigma=2)

        ar = AsterismRegister()
        tform = ar._compute_transform(im1, im2)

        assert_almost_equal(tform.translation, shift, decimal=1)
        assert_almost_equal(tform.rotation, 0, decimal=3)
        assert_almost_equal(tform.scale, 1, decimal=4)

    def test_compute_transform_rotation(self):
        size = (1024, 1024)
        sky = 800
        n = 300
        rdnoise = 5
        x, y, f = gen_position_flux(2*np.array(size), n, 1e4, 4e6)
        x -= 500
        y -= 500

        x1, y1, flux1 = gen_positions_transformed(x, y, f, 0, 0, size)
        im1 = gen_image(size, x1, y1, flux1,
                        sky, rdnoise, sigma=2)

        x2, y2, flux2 = gen_positions_transformed(x, y, f, 0, 0, size,
                                                  rotation=35.2)
        im2 = gen_image(size, x2, y2, flux2,
                        sky, rdnoise, sigma=2)

        ar = AsterismRegister()
        tform = ar._compute_transform(im1, im2)

        assert_almost_equal(tform.rotation, np.deg2rad(35.2), decimal=3)
        # the translation is needed due to the form skimage handles rotation
        assert_almost_equal(tform.translation, [388.7, -201.5], decimal=0)
        assert_almost_equal(tform.scale, 1, decimal=4)


class Test_CrossCorrelationRegister:
    @pytest.mark.parametrize('shift', [(25, 32), (-12, 5), (23.42, 12.43)])
    def test_compute_transform(self, shift):
        size = (1024, 1024)
        sky = 800
        n = 60
        rdnoise = 5
        x, y, f = gen_position_flux(np.array(size)+80, n, 1e4, 4e6)
        x -= 40
        y -= 40
        sx, sy = shift

        x1, y1, flux1 = gen_positions_transformed(x, y, f, 0, 0, size)
        im1 = gen_image(size, x1, y1, flux1,
                        sky, rdnoise, sigma=2)

        x2, y2, flux2 = gen_positions_transformed(x, y, f, sx, sy, size)
        im2 = gen_image(size, x2, y2, flux2,
                        sky, rdnoise, sigma=2)

        ccr = CrossCorrelationRegister(upsample_factor=10)
        tform = ccr._compute_transform(im1, im2)

        assert_almost_equal(tform.translation, shift, decimal=1)
        assert_almost_equal(tform.rotation, 0, decimal=3)
        assert_almost_equal(tform.scale, 1, decimal=4)


class Test_Registration:
    @pytest.mark.parametrize('cval,fill', [(0, 0),
                                           ('median', 1.0),
                                           ('mean', 1.51)])
    def test_register_image(self, cval, fill):
        im1 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 2, 2, 2, 1, 1, 1, 1, 1],
               [1, 2, 4, 6, 4, 2, 1, 1, 1, 1],
               [1, 2, 6, 8, 6, 2, 1, 1, 1, 1],
               [1, 2, 4, 6, 4, 2, 1, 1, 1, 1],
               [1, 1, 2, 2, 2, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

        im2 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 2, 2, 2, 1, 1, 1],
               [1, 1, 1, 2, 4, 6, 4, 2, 1, 1],
               [1, 1, 1, 2, 6, 8, 6, 2, 1, 1],
               [1, 1, 1, 2, 4, 6, 4, 2, 1, 1],
               [1, 1, 1, 1, 2, 2, 2, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

        expect = np.array(im1, dtype='f8')
        expect[0, :] = fill
        expect[:, -2:] = fill

        mask = np.zeros_like(im2, dtype=bool)
        mask[0, :] = 1
        mask[:, -2:] = 1

        ar = CrossCorrelationRegister()
        im_reg, mask_reg, tform = ar.register_image(np.array(im1), np.array(im2),
                                                    cval=cval)

        assert_equal(im_reg, expect)
        assert_equal(mask_reg, mask)
        assert_equal(tform.translation, [2, -1])

    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('cval,fill', [(0, 0),
                                           ('median', 1.0),
                                           ('mean', 1.51)])
    def test_register_framedata(self, inplace, cval, fill):
        im1 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 2, 2, 2, 1, 1, 1, 1, 1],
               [1, 2, 4, 6, 4, 2, 1, 1, 1, 1],
               [1, 2, 6, 8, 6, 2, 1, 1, 1, 1],
               [1, 2, 4, 6, 4, 2, 1, 1, 1, 1],
               [1, 1, 2, 2, 2, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

        im2 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 2, 2, 2, 1, 1, 1],
               [1, 1, 1, 2, 4, 6, 4, 2, 1, 1],
               [1, 1, 1, 2, 6, 8, 6, 2, 1, 1],
               [1, 1, 1, 2, 4, 6, 4, 2, 1, 1],
               [1, 1, 1, 1, 2, 2, 2, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

        expect = np.array(im1, dtype='f8')
        expect[0, :] = fill
        expect[:, -2:] = fill

        mask = np.zeros_like(im2, dtype=bool)
        mask[0, :] = 1
        mask[:, -2:] = 1

        expect_unct = np.ones_like(im2, dtype='f8')
        expect_unct[0, :] = np.nan
        expect_unct[:, -2:] = np.nan

        frame1 = FrameData(im1, dtype='f8')
        frame1.meta['moving'] = False
        frame1.uncertainty = np.ones_like(im1)
        frame2 = FrameData(im2, dtype='f8')
        frame2.meta['moving'] = True
        frame2.uncertainty = np.ones_like(im2)

        ar = CrossCorrelationRegister()
        frame_reg= ar.register_framedata(frame1, frame2,
                                         cval=cval, inplace=inplace)

        assert_equal(frame_reg.data, expect)
        assert_equal(frame_reg.mask, mask)
        assert_equal(frame_reg.uncertainty, expect_unct)
        assert_equal(frame_reg.meta['astropop registration'],
                                    'cross-correlation')
        assert_equal(frame_reg.meta['astropop registration_shift'], [2, -1])
        assert_equal(frame_reg.meta['astropop registration_rot'], 0)
        assert_equal(frame_reg.meta['moving'], True)
        if inplace:
            assert_is(frame_reg, frame2)
        else:
            assert_is_not(frame_reg, frame2)

    def test_register_image_equal(self):
        im = gen_image((50, 50), [25], [25], [10000], 10, 0, sigma=3)
        ar = CrossCorrelationRegister()
        im_reg, mask_reg, tform = ar.register_image(im, im)
        assert_is(im_reg, im)
        assert_equal(im_reg, im)
        assert_equal(mask_reg, np.zeros_like(im))
        assert_equal(tform.translation, [0, 0])

    @pytest.mark.parametrize('inplace', [True, False])
    def test_register_frame_equal(self, inplace):
        im = gen_image((50, 50), [25], [25], [10000], 10, 0, sigma=3)
        im = FrameData(im)
        ar = CrossCorrelationRegister()
        im_reg= ar.register_framedata(im, im, inplace=inplace)
        if inplace:
            assert_is(im_reg, im)
        else:
            assert_is_not(im_reg, im)
        assert_equal(im_reg.data, im.data)
        assert_equal(im_reg.mask, np.zeros_like(im))
        assert_equal(im_reg.meta['astropop registration_shift'], [0, 0])


class Test_Register_FrameData_List:
    _shifts = [(0, 0), (-1, 2.4), (1.5, 3.2), (-2.2, 1.75), (-0.5, 0.5)]

    def gen_frame_list(self, size):
        sky = 800
        rdnoise = 10
        n = 100
        x, y, f = gen_position_flux(np.array(size)+80, n, 1e4, 4e6)
        x -= 40
        y -= 40

        frame_list = []
        for shift in self._shifts:
            x1, y1, flux1 = gen_positions_transformed(x, y, f, *shift, size)
            im1 = gen_image(size, x1, y1, flux1,
                            sky, rdnoise, sigma=2)
            frame = FrameData(im1, meta={'test expect_shift': list(shift)})
            frame_list.append(frame)

        return frame_list

    def test_error_unkown_algorithm(self):
        with pytest.raises(ValueError, match='Algorithm noexisting unknown.'):
            register_framedata_list([FrameData(None) for i in range(10)],
                                    algorithm='noexisting')

    def test_error_non_framedata(self):
        with pytest.raises(TypeError, match='Only a list of FrameData'):
            register_framedata_list([np.zeros((10, 10)) for i in range(10)])

    def test_error_non_iterable_list(self):
        with pytest.raises(TypeError):
            register_framedata_list(10)

    def test_error_incompatible_shapes(self):
        frame_list = [FrameData(np.zeros((i+1, i+1))) for i in range(10)]
        with pytest.raises(ValueError, match='incompatible shapes'):
            register_framedata_list(frame_list)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_register_framedata_crosscorr(self, inplace):
        frame_list = self.gen_frame_list((1024, 1024))
        reg_list = register_framedata_list(frame_list,
                                           algorithm='cross-correlation',
                                           inplace=inplace,
                                           upsample_factor=10, space='real')
        assert_equal(len(frame_list), len(reg_list))
        for org, reg in zip(frame_list, reg_list):
            if inplace:
                assert_is(org, reg)
            else:
                assert_is_not(org, reg)
            assert_almost_equal(reg.meta['astropop registration_shift'],
                                org.meta['test expect_shift'], decimal=0)

    @pytest.mark.parametrize('inplace', [True, False])
    def test_register_framedata_asterism(self, inplace):
        frame_list = self.gen_frame_list((1024, 1024))
        reg_list = register_framedata_list(frame_list,
                                           algorithm='asterism-matching',
                                           inplace=inplace,
                                           max_control_points=30,
                                           detection_threshold=5)
        assert_equal(len(frame_list), len(reg_list))
        for org, reg in zip(frame_list, reg_list):
            if inplace:
                assert_is(org, reg)
            else:
                assert_is_not(org, reg)
            assert_almost_equal(reg.meta['astropop registration_shift'],
                                org.meta['test expect_shift'], decimal=0)
