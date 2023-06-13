# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from astropop.image.register import AsterismRegister, \
                                    CrossCorrelationRegister, \
                                    register_framedata_list, \
                                    compute_shift_list
from astropop.framedata import FrameData, PixelMaskFlags
from astropop.testing import *

from .test_detection import gen_position_flux, gen_image


def gen_positions_transformed(x, y, flux, dx, dy, limits,
                              rotation=None, rotation_center=None):
    """Generate translated positions."""
    x, y = x+dx, y+dy

    if rotation is not None:
        rotation_center = rotation_center or np.array(limits)/2
        cx, cy = rotation_center
        theta = np.deg2rad(rotation)
        nx = cx + (x-cx)*np.cos(theta) - (y-cy)*np.sin(theta)
        ny = cy + (x-cx)*np.sin(theta) + (y-cy)*np.cos(theta)
        x, y = nx, ny

    # ensure all positions are inside the image
    mask = x >= 0
    mask &= x <= limits[0]
    mask &= y >= 0
    mask &= y <= limits[1]
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

        flags = np.zeros_like(im2, dtype='u1')
        flags[5, 5] = 5
        exp_flags = np.zeros_like(im2, dtype='u1')
        exp_flags[3, 6] = 5
        exp_flags[0, :] = (PixelMaskFlags.REMOVED |
                           PixelMaskFlags.OUT_OF_BOUNDS).value
        exp_flags[:, -2:] = (PixelMaskFlags.REMOVED |
                             PixelMaskFlags.OUT_OF_BOUNDS).value

        expect_unct = np.ones_like(im2, dtype='f8')
        expect_unct[0, :] = np.nan
        expect_unct[:, -2:] = np.nan

        frame1 = FrameData(im1, dtype='f8', flags=flags)
        frame1.meta['moving'] = False
        frame1.uncertainty = np.ones_like(im1)
        frame2 = FrameData(im2, dtype='f8', flags=flags)
        frame2.meta['moving'] = True
        frame2.uncertainty = np.ones_like(im2)

        ar = CrossCorrelationRegister()
        frame_reg = ar.register_framedata(frame1, frame2,
                                          cval=cval, inplace=inplace)

        assert_equal(frame_reg.data, expect)
        assert_equal(frame_reg.mask, mask)
        assert_equal(frame_reg.uncertainty, expect_unct)
        assert_equal(frame_reg.meta['astropop registration'],
                     'cross-correlation')
        assert_equal(frame_reg.meta['astropop registration_shift_x'], 2)
        assert_equal(frame_reg.meta['astropop registration_shift_y'], -1)
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
        flags = np.zeros((50, 50), dtype=np.uint8)
        flags[25, 25] = 5
        im.flags = flags
        ar = CrossCorrelationRegister()
        im_reg = ar.register_framedata(im, im, inplace=inplace)
        if inplace:
            assert_is(im_reg, im)
        else:
            assert_is_not(im_reg, im)
        assert_equal(im_reg.data, im.data)
        assert_equal(im_reg.mask, np.zeros_like(im))
        assert_equal(im_reg.flags, flags)
        assert_equal(im_reg.meta['astropop registration_shift_x'], 0)
        assert_equal(im_reg.meta['astropop registration_shift_y'], 0)


class Test_Register_FrameData_List:
    _shifts = [(0, 0), (-1, 22.4), (15.5, 3.2), (2.2, -1.75), (-5.4, 0.5)]

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
            frame = FrameData(im1, meta={'test expect_shift_x': shift[0],
                                         'test expect_shift_y': shift[1]})
            frame_list.append(frame)

        return frame_list

    def test_error_unkown_algorithm(self):
        with pytest.raises(ValueError, match='Algorithm noexisting unknown.'):
            register_framedata_list([FrameData([[1]]) for i in range(10)],
                                    algorithm='noexisting')
        with pytest.raises(ValueError, match='Algorithm noexisting unknown.'):
            compute_shift_list([FrameData([[1]]) for i in range(10)],
                               algorithm='noexisting')

    def test_error_non_framedata(self):
        with pytest.raises(TypeError, match='Only a list of FrameData'):
            register_framedata_list([np.zeros((10, 10)) for i in range(10)])
        with pytest.raises(TypeError, match='Only a list of FrameData'):
            compute_shift_list([np.zeros((10, 10)) for i in range(10)])

    def test_error_non_iterable_list(self):
        with pytest.raises(TypeError):
            register_framedata_list(10)
        with pytest.raises(TypeError):
            compute_shift_list(10)

    def test_error_incompatible_shapes(self):
        frame_list = [FrameData(np.zeros((i+1, i+1))) for i in range(10)]
        with pytest.raises(ValueError, match='incompatible shapes'):
            register_framedata_list(frame_list)
        with pytest.raises(ValueError, match='incompatible shapes'):
            compute_shift_list(frame_list)

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
            for i in ['x', 'y']:
                ap_reg_shift = reg.meta[f'astropop registration_shift_{i}']
                ex_reg_shift = org.meta[f'test expect_shift_{i}']
                assert_almost_equal(ap_reg_shift, ex_reg_shift, decimal=0)

        if not inplace:
            shift_list = compute_shift_list(frame_list,
                                            algorithm='cross-correlation',
                                            upsample_factor=10, space='real')
            assert_equal(len(frame_list), len(shift_list))
            assert_almost_equal(shift_list, self._shifts, decimal=0)

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
            for i in ['x', 'y']:
                ap_reg_shift = reg.meta[f'astropop registration_shift_{i}']
                ex_reg_shift = org.meta[f'test expect_shift_{i}']
                assert_almost_equal(ap_reg_shift, ex_reg_shift, decimal=0)

        if not inplace:
            shift_list = compute_shift_list(frame_list,
                                            algorithm='asterism-matching',
                                            max_control_points=30,
                                            detection_threshold=5)
            assert_equal(len(frame_list), len(shift_list))
            assert_almost_equal(shift_list, self._shifts, decimal=0)

    def test_register_framedata_list_ref_image(self):
        frame_list = self.gen_frame_list((1024, 1024))
        reg_list = register_framedata_list(frame_list,
                                           algorithm='asterism-matching',
                                           ref_image=4,
                                           max_control_points=30,
                                           detection_threshold=5)
        assert_equal(len(frame_list), len(reg_list))
        ref_shift = np.array(self._shifts[4])

        for org, reg in zip(frame_list, reg_list):
            for i, ref in zip(['x', 'y'], ref_shift):
                ap_reg_shift = reg.meta[f'astropop registration_shift_{i}']
                ex_reg_shift = org.meta[f'test expect_shift_{i}'] - ref
                assert_almost_equal(ap_reg_shift, ex_reg_shift, decimal=0)

        shift_list = compute_shift_list(frame_list,
                                        ref_image=4,
                                        algorithm='asterism-matching',
                                        max_control_points=30,
                                        detection_threshold=5)
        assert_equal(len(frame_list), len(shift_list))
        assert_almost_equal(shift_list, self._shifts - ref_shift, decimal=0)

    def test_register_framedata_list_clip(self):
        frame_list = self.gen_frame_list((512, 1024))
        reg_list = register_framedata_list(frame_list, clip_output=True,
                                           inplace=True,
                                           algorithm='asterism-matching',
                                           max_control_points=30,
                                           detection_threshold=5)

        assert_equal(len(frame_list), len(reg_list))
        for org, reg in zip(frame_list, reg_list):
            assert_is(org, reg)
            for i in ['x', 'y']:
                ap_reg_shift = reg.meta[f'astropop registration_shift_{i}']
                ex_reg_shift = org.meta[f'test expect_shift_{i}']
                assert_almost_equal(ap_reg_shift, ex_reg_shift, decimal=0)
            assert_equal(reg.meta['astropop trimmed_section'], '6:-16,2:-23')
            # x: 6:-16, y: 2:-23
            assert_equal(reg.shape, (1024-23-2, 512-6-16))
            # no masked pixel should remain
            assert_false(np.any(reg.mask))

    def test_register_framedata_list_skip_failure_default(self):
        # defult behavior is raise error
        frame_list = self.gen_frame_list((512, 1024))
        frame_list[2].data = np.ones((1024, 512))

        with pytest.raises(ValueError):
            register_framedata_list(frame_list, algorithm='asterism-matching')

        with pytest.raises(ValueError):
            compute_shift_list(frame_list, algorithm='asterism-matching')

    def test_register_framedata_list_skip_failure_false(self):
        frame_list = self.gen_frame_list((512, 1024))
        frame_list[2].data = np.ones((1024, 512))

        with pytest.raises(ValueError):
            register_framedata_list(frame_list, algorithm='asterism-matching',
                                    skip_failure=False)

        with pytest.raises(ValueError):
            compute_shift_list(frame_list, algorithm='asterism-matching',
                               skip_failure=False)

    @pytest.mark.parametrize('cval,expct_cval', [(np.nan, np.nan),
                                                 ('median', 1),
                                                 ('mean', 1),
                                                 (0, 0)])
    def test_register_framedata_list_skip_failure_true(self, cval, expct_cval):
        frame_list = self.gen_frame_list((512, 1024))
        frame_list[2].data = np.ones((1024, 512))

        reg_list = register_framedata_list(frame_list, clip_output=True,
                                           inplace=False,
                                           algorithm='asterism-matching',
                                           max_control_points=30,
                                           detection_threshold=5,
                                           cval=cval,
                                           skip_failure=True)

        assert_equal(len(frame_list), len(reg_list))
        assert_is_none(reg_list[2].meta['astropop registration_shift_x'])
        assert_is_none(reg_list[2].meta['astropop registration_shift_y'])
        assert_is_none(reg_list[2].meta['astropop registration_rot'])
        assert_equal(reg_list[2].meta['astropop registration'], 'failed')
        assert_equal(reg_list[2].data, np.full(reg_list[2].shape, expct_cval))
        assert_true(np.all(reg_list[2].mask))

        for org, reg in zip(frame_list, reg_list):
            assert_equal(reg.meta['astropop trimmed_section'], '6:-3,2:-23')
            # x: 6:-3, y: 2:-23, since frame[2] is not available
            assert_equal(reg.shape, (1024-23-2, 512-6-3))

        shift_list = compute_shift_list(frame_list,
                                        algorithm='asterism-matching',
                                        max_control_points=30,
                                        detection_threshold=5,
                                        skip_failure=True)
        shift_list_expt = np.array(self._shifts)
        shift_list_expt[2][:] = np.nan
        assert_almost_equal(shift_list, shift_list_expt, decimal=1)
