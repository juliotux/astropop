# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest

import numpy as np
import os
from astroquery.skyview import SkyView
from astropy.coordinates import Angle, SkyCoord
from astropy.config import get_cache_dir
from astropy.io import fits
from astropy.table import Table
from astropy.nddata.ccddata import _generate_wcs_and_update_header
from astropy.wcs import WCS
from astropy import units
from astropy.utils.data import download_file

from astropop.astrometry.astrometrynet import _solve_field, \
                                              solve_astrometry_image, \
                                              solve_astrometry_xy, \
                                              solve_astrometry_hdu, \
                                              solve_astrometry_framedata, \
                                              AstrometrySolver
from astropop.astrometry.astrometrynet import _parse_angle, \
                                              _parse_coordinates, \
                                              _parse_crpix, \
                                              _parse_pltscl
from astropop.astrometry.manual_wcs import wcs_from_coords
from astropop.astrometry.coords_utils import guess_coordinates
from astropop.framedata import FrameData
from astropop.photometry.detection import starfind

from astropop.testing import *


def compare_wcs(wcs, nwcs):
    for i in [(100, 100), (1000, 1500), (357.5, 948.2), (2015.1, 403.7)]:
        res1 = np.array(wcs.all_pix2world(*i, 0))
        res2 = np.array(nwcs.all_pix2world(*i, 0))
        assert_almost_equal(res1, res2, decimal=1)


skip_astrometry = pytest.mark.skipif("_solve_field is None or "
                                     "os.getenv('SKIP_TEST_ASTROMETRY', "
                                     "False)")


@pytest.mark.remote_data
class Test_AstrometrySolver:
    def get_image(self):
        # return image name and index name
        cache_dir = get_cache_dir()
        cache_dir = os.path.join(cache_dir, 'astropop')

        # download the image
        os.makedirs(cache_dir, exist_ok=True)
        image = os.path.join(cache_dir, 'M67.fits')
        # make sure that this is the M67 image
        if os.path.isfile(image):
            f = fits.open(image)
            if not f[0].header['ASTROPOP'] == 'Test Astrometry M67':
                os.remove(image)
        if not os.path.isfile(image):
            dss_image = SkyView.get_images(position='M67',
                                           survey='DSS',
                                           radius=Angle('0.2 deg'))[0][0]
            dss_image.header['ASTROPOP'] = 'Test Astrometry M67'
            dss_image.writeto(image)

        # download the index file
        index = os.path.join(cache_dir, 'indexes',
                             '5200', 'index-5203-04.fits')
        os.makedirs(os.path.dirname(index), exist_ok=True)
        if not os.path.isfile(index):
            astr_url = 'https://portal.nersc.gov/project/cosmo/temp/dstn/'
            index_url = 'index-5200/LITE/index-5203-04.fits'
            f = download_file(astr_url + index_url, cache=True,
                              allow_insecure=True)
            os.rename(f, index)

        options = {'ra': '08:51:18.0', 'dec': '11:48:00.0',
                   'radius': '1.0 deg', 'add_path': os.path.dirname(index)}

        return image, index, options

    @pytest.mark.parametrize('angle,unit,fail', [(Angle(1.0, 'degree'), None, False),
                                                 (1.0, None, False),
                                                 ('1 degree', None, False),
                                                 ('1 deg', None, False),
                                                 (np.radians(1.0), 'radian', False),
                                                 ('not angle', None, True),
                                                 ('01:00:00', 'deg', False),
                                                 ('00:04:00', 'hourangle', False),
                                                 ('1 yr', None, True),
                                                 ('60 min', None, False)])
    def test_parse_angle(self, angle, unit, fail):
        if not fail:
            assert_almost_equal(_parse_angle(angle, unit), 1.0)
        else:
            with pytest.raises((units.UnitsError, ValueError)):
                _parse_angle(angle)

    @pytest.mark.parametrize('options', [{'center': SkyCoord(1, 1, unit='deg')},
                                         {'ra': 1.0, 'dec': 1.0},
                                         {'ra': '00:04:00', 'dec': '01:00:00'},
                                         {'ra': '00h04m00s', 'dec': '01d00m00s'},
                                         {'center': (1.0, 1.0)}])
    def test_parse_center(self, options):
        args = _parse_coordinates(options)
        # this options must be popped
        assert_not_in('center', options)
        assert_not_in('ra', options)
        assert_not_in('dec', options)
        assert_equal(args[0], '--ra')
        assert_equal(args[2], '--dec')
        assert_almost_equal(float(args[1]), 1.0)
        assert_almost_equal(float(args[3]), 1.0)

    def test_parse_center_fails(self):
        assert_equal(_parse_coordinates({}), [])

        with pytest.raises(ValueError, match='conflicts with'):
            _parse_coordinates({'center': (1.0, 1.0),
                                'ra': 1.0, 'dec': 1.0})

    def test_parse_pltscl(self):
        def _assert(options, expect):
            args = _parse_pltscl(option)
            assert_equal(args[0], expect[0])
            assert_almost_equal(float(args[1]), float(expect[1]))
            assert_equal(args[2], expect[2])
            assert_almost_equal(float(args[3]), float(expect[3]))
            assert_equal(args[4], expect[4])
            assert_equal(args[5], expect[5])

        option = {'plate-scale': 0.2, 'scale-tolerance': 0.2}
        expect = ['--scale-low', '0.16', '--scale-high', '0.24',
                  '--scale-units', 'arcsecperpix']
        _assert(option, expect)

        # default tolerance is 0.2
        option = {'plate-scale': 0.2}
        expect = ['--scale-low', '0.16', '--scale-high', '0.24',
                  '--scale-units', 'arcsecperpix']
        _assert(option, expect)

        # another unit
        option = {'plate-scale': 0.2, 'scale-units': 'degreeperpix'}
        expect = ['--scale-low', '0.16', '--scale-high', '0.24',
                  '--scale-units', 'degreeperpix']
        _assert(option, expect)

        # low and high
        option = {'scale-low': 1, 'scale-high': 2, 'scale-units': 'arcsecperpix'}
        expect = ['--scale-low', '1', '--scale-high', '2',
                  '--scale-units', 'arcsecperpix']
        _assert(option, expect)

    def test_parse_scale_errors(self):
        assert_equal(_parse_pltscl({}), [])
        with pytest.raises(ValueError, match='is in conflict with'):
            _parse_pltscl({'plate-scale': 1, 'scale-low': 1, 'scale-high': 1})
        with pytest.raises(ValueError, match='must specify'):
            _parse_pltscl({'scale-low': 1, 'scale-high': 1})

    def test_parse_crpix_center(self):
        opt = {'crpix-center': None}
        arg = _parse_crpix(opt)
        assert_equal(arg, ['--crpix-center'])
        assert_not_in('crpix-center', opt)

    @pytest.mark.parametrize('x,y', [(1.0, 1.0), (1, 1), ('1', '1')])
    def test_parse_crpix_xy(self, x, y):
        opt = {'crpix-x': x, 'crpix-y': y}
        arg = _parse_crpix(opt)
        assert_not_in('crpix-x', opt)
        assert_not_in('crpix-y', opt)
        assert_equal(arg[0], '--crpix-x')
        assert_equal(arg[2], '--crpix-y')
        assert_almost_equal(float(arg[1]), 1.0)
        assert_almost_equal(float(arg[3]), 1.0)

    def test_parse_crpix_fails(self):
        with pytest.raises(ValueError, match='conflicts with'):
            _parse_crpix({'crpix-center': None, 'crpix-x': 1, 'crpix-y': 1})
        assert_equal(_parse_crpix({}), [])

    @skip_astrometry
    def test_read_cfg(self):
        a = AstrometrySolver()  # read the default configuration
        cfg = a._read_config()
        assert_not_in('index', cfg)
        assert_false(cfg['inparallel'])
        assert_not_in('', cfg)
        assert_not_in('minwidth', cfg)
        assert_not_in('maxwidth', cfg)
        assert_equal(cfg['cpulimit'], '300')
        assert_true(cfg['autoindex'])
        assert_equal(len(cfg['add_path']), 1)

    @skip_astrometry
    def test_read_cfg_fname(self, tmpdir):
        fname = tmpdir / 'test.cfg'
        f = open(fname, 'w')
        f.write("inparallel\n")
        f.write(" minwidth 0.1 \n")
        f.write(" maxwidth 180\n")
        f.write("depths 10 20 30 40   50 60\n")
        f.write("cpulimit   300\n")
        f.write("\n\n")
        f.write("# comment\n")
        f.write("add_path /data  # commented path\n")
        f.write("add_path /data1\n")
        f.write("#add_path /data2\n")  # data2 will not be present
        f.write("autoindex\n")
        f.write("index index-219\n")
        f.write("index index-220\n")
        f.write("index index-221\n")
        f.write("# index index-222\n")  # 222 will not be present
        f.close()

        a = AstrometrySolver()
        cfg = a._read_config(fname)
        assert_not_equal("", cfg)
        assert_true(cfg['inparallel'])
        assert_equal(cfg['minwidth'], '0.1')
        assert_equal(cfg['maxwidth'], '180')
        assert_equal(cfg['depths'], [10, 20, 30, 40, 50, 60])
        assert_equal(cfg['cpulimit'], '300')
        assert_equal(cfg['add_path'], ['/data', '/data1'])
        assert_equal(cfg['index'], ['index-219', 'index-220', 'index-221'])
        assert_true(cfg['autoindex'])

    @skip_astrometry
    def test_read_cfg_with_options(self, tmpdir):
        fname = tmpdir / 'test.cfg'
        f = open(fname, 'w')
        f.write("inparallel\n")
        f.write(" minwidth 0.1 \n")
        f.write(" maxwidth 180\n")
        f.write("depths 10 20 30 40   50 60\n")
        f.write("cpulimit   300\n")
        f.write("\n\n")
        f.write("# comment\n")
        f.write("add_path /data  # commented path\n")
        f.write("add_path /data1\n")
        f.write("#add_path /data2\n")  # data2 will not be present
        f.write("autoindex\n")
        f.write("index index-219\n")
        f.write("index index-220\n")
        f.write("index index-221\n")
        f.write("# index index-222\n")  # 222 will not be present
        f.close()

        a = AstrometrySolver()
        cfg = a._read_config(fname, {'depths': [10, 30, 50],
                                     'add_path': '/data3',
                                     'index': ['indx4', 'indx5']})
        assert_not_equal("", cfg)
        assert_true(cfg['inparallel'])
        assert_equal(cfg['minwidth'], '0.1')
        assert_equal(cfg['maxwidth'], '180')
        assert_equal(cfg['depths'], [10, 30, 50])
        assert_equal(cfg['cpulimit'], '300')
        assert_equal(cfg['add_path'], ['/data', '/data1', '/data3'])
        assert_equal(cfg['index'], ['index-219', 'index-220', 'index-221',
                                    'indx4', 'indx5'])
        assert_true(cfg['autoindex'])

    @skip_astrometry
    def test_write_config(self, tmpdir):
        fname = tmpdir / 'test.cfg'

        a = AstrometrySolver()
        a.config = {'inparallel': False,
                    'autoindex': True,
                    'cpulimit': 300,
                    'minwidth': 0.1,
                    'maxwidth': 180,
                    'depths': [20, 40, 60],
                    'index': ['011', '012'],
                    'add_path': ['/path1', '/path2']}
        a._write_config(fname)

        with open(fname, 'r') as f:
            for line in f.readlines():
                assert_in(line.strip('\n'), ['autoindex', 'inparallel',
                                             'cpulimit 300', 'minwidth 0.1',
                                             'maxwidth 180', 'depths 20 40 60',
                                             'index 011', 'index 012',
                                             'add_path /path1',
                                             'add_path /path2'])

    @skip_astrometry
    def test_pop_config(self):
        a = AstrometrySolver()
        options1 = {'inparallel': False,
                    'autoindex': True,
                    'cpulimit': 300,
                    'minwidth': 0.1,
                    'maxwidth': 180,
                    'depths': [20, 40, 60],
                    'index': ['011', '012'],
                    'add_path': ['/path1', '/path2'],
                    'ra': 0.0, 'dec': 0.0, 'radius': 1.0}
        options, cfg = a._pop_config(options1)
        assert_is_not(options1, options)
        assert_equal(options['ra'], 0.0)
        assert_equal(options['dec'], 0.0)
        assert_equal(options['radius'], 1.0)
        assert_equal(options['cpulimit'], 300)
        assert_equal(cfg['inparallel'], False)
        assert_equal(cfg['autoindex'], True)
        assert_equal(cfg['minwidth'], 0.1)
        assert_equal(cfg['maxwidth'], 180)
        assert_equal(cfg['depths'], [20, 40, 60])
        assert_equal(cfg['index'], ['011', '012'])
        assert_equal(cfg['add_path'], ['/path1', '/path2'])

    @skip_astrometry
    def test_only_write_config_when_needed(self, tmpdir):
        a = AstrometrySolver()
        args = a._get_args(tmpdir/'1/', tmpdir/'1/fitsfile.fits',
                           {'ra': 0.0, 'dec': 0.0, 'radius': 1.0},
                           output_dir=tmpdir, correspond='test.correspond')
        assert_not_in('--config', args)

        args = a._get_args(tmpdir/'2/', tmpdir/'2/fitsfile.fits',
                           {'ra': 0.0, 'dec': 0.0, 'radius': 1.0,
                            'inparallel': False},
                           output_dir=tmpdir, correspond='test.correspond')
        assert_in('--config', args)

    @skip_astrometry
    def test_solve_astrometry_hdu(self, tmpdir):
        data, index, options = self.get_image()
        index_dir = os.path.dirname(index)
        hdu = fits.open(data)[0]
        header, wcs = _generate_wcs_and_update_header(hdu.header)
        hdu.header = header
        result = solve_astrometry_hdu(hdu, options=options)
        assert_true(isinstance(result.wcs, WCS))
        assert_equal(result.wcs.naxis, 2)
        compare_wcs(wcs, result.wcs)
        assert_is_instance(result.header, fits.Header)
        assert_is_instance(result.correspondences, Table)
        for k in ['field_x', 'field_y', 'index_x', 'index_y',
                  'field_ra', 'field_dec', 'index_ra', 'index_dec']:
            assert_in(k, result.correspondences.colnames)

    @skip_astrometry
    def test_solve_astrometry_xyl(self, tmpdir):
        data, index, options = self.get_image()
        index_dir = os.path.dirname(index)
        hdu = fits.open(data)[0]
        header, wcs = _generate_wcs_and_update_header(hdu.header)
        hdu.header = header
        phot = starfind(data=hdu.data, threshold=5,
                        background=np.median(hdu.data),
                        noise=np.std(hdu.data))
        imw, imh = hdu.data.shape
        result = solve_astrometry_xy(phot['x'], phot['y'], phot['flux'],
                                     width=imw, height=imh,
                                     options=options)
        assert_true(isinstance(result.wcs, WCS))
        assert_equal(result.wcs.naxis, 2)
        compare_wcs(wcs, result.wcs)
        assert_is_instance(result.header, fits.Header)
        assert_is_instance(result.correspondences, Table)
        for k in ['field_x', 'field_y', 'index_x', 'index_y',
                  'field_ra', 'field_dec', 'index_ra', 'index_dec']:
            assert_in(k, result.correspondences.colnames)

    @skip_astrometry
    def test_solve_astrometry_image(self, tmpdir):
        data, index, options = self.get_image()
        index_dir = os.path.dirname(index)
        hdu = fits.open(data)[0]
        header, wcs = _generate_wcs_and_update_header(hdu.header)
        hdu.header = header
        name = tmpdir.join('testimage.fits').strpath
        hdu.writeto(name)
        result = solve_astrometry_image(name, options=options)
        compare_wcs(wcs, result.wcs)
        assert_is_instance(result.header, fits.Header)
        assert_is_instance(result.correspondences, Table)
        for k in ['field_x', 'field_y', 'index_x', 'index_y',
                  'field_ra', 'field_dec', 'index_ra', 'index_dec']:
            assert_in(k, result.correspondences.colnames)

    @skip_astrometry
    def test_solve_astrometry_framedata(self, tmpdir):
        data, index, options = self.get_image()
        index_dir = os.path.dirname(index)
        hdu = fits.open(data)[0]
        header, wcs = _generate_wcs_and_update_header(hdu.header)
        hdu.header = header
        f = FrameData(hdu.data, header=hdu.header)
        result = solve_astrometry_framedata(f, options=options)
        compare_wcs(wcs, result.wcs)
        assert_is_instance(result.header, fits.Header)


class Test_ManualWCS:
    def test_manual_wcs_top(self):
        # Checked with DS9
        x, y = (11, 11)
        ra, dec = (10.0, 0.0)
        ps = 36  # arcsec/px
        north = 'top'  # north to right
        wcs = wcs_from_coords(x, y, ra, dec, ps, north)
        assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
        assert_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, 0.1))
        assert_almost_equal(wcs.all_pix2world(21, 11, 1), (9.9, 0.0))
        assert_almost_equal(wcs.all_pix2world(1, 11, 1), (10.1, 0.0))
        assert_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, -0.1))
        assert_almost_equal(wcs.all_pix2world(21, 21, 1), (9.9, 0.1))
        assert_almost_equal(wcs.all_pix2world(1, 21, 1), (10.1, 0.1))
        assert_almost_equal(wcs.all_pix2world(1, 1, 1), (10.1, -0.1))

    def test_manual_wcs_left(self):
        # Checked with DS9
        x, y = (11, 11)
        ra, dec = (10.0, 0.0)
        ps = 36  # arcsec/px
        north = 'left'  # north to right
        wcs = wcs_from_coords(x, y, ra, dec, ps, north)
        assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
        assert_almost_equal(wcs.all_pix2world(11, 21, 1), (9.9, 0.0))
        assert_almost_equal(wcs.all_pix2world(21, 11, 1), (10.0, -0.1))
        assert_almost_equal(wcs.all_pix2world(1, 11, 1), (10.0, 0.1))
        assert_almost_equal(wcs.all_pix2world(11, 1, 1), (10.1, 0.0))
        assert_almost_equal(wcs.all_pix2world(21, 21, 1), (9.9, -0.1))
        assert_almost_equal(wcs.all_pix2world(1, 21, 1), (9.9, 0.1))
        assert_almost_equal(wcs.all_pix2world(1, 1, 1), (10.1, 0.1))

    def test_manual_wcs_bottom(self):
        # Checked with DS9
        x, y = (11, 11)
        ra, dec = (10.0, 0.0)
        ps = 36  # arcsec/px
        north = 'bottom'  # north to right
        wcs = wcs_from_coords(x, y, ra, dec, ps, north)
        assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
        assert_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, -0.1))
        assert_almost_equal(wcs.all_pix2world(21, 11, 1), (10.1, 0.0))
        assert_almost_equal(wcs.all_pix2world(1, 11, 1), (9.9, 0.0))
        assert_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, 0.1))
        assert_almost_equal(wcs.all_pix2world(21, 21, 1), (10.1, -0.1))
        assert_almost_equal(wcs.all_pix2world(1, 21, 1), (9.9, -0.1))
        assert_almost_equal(wcs.all_pix2world(1, 1, 1), (9.9, 0.1))

    def test_manual_wcs_right(self):
        # Checked with DS9
        x, y = (11, 11)
        ra, dec = (10.0, 0.0)
        ps = 36  # arcsec/px
        north = 'right'
        wcs = wcs_from_coords(x, y, ra, dec, ps, north)
        assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
        assert_almost_equal(wcs.all_pix2world(11, 21, 1), (10.1, 0.0))
        assert_almost_equal(wcs.all_pix2world(21, 11, 1), (10.0, 0.1))
        assert_almost_equal(wcs.all_pix2world(1, 11, 1), (10.0, -0.1))
        assert_almost_equal(wcs.all_pix2world(11, 1, 1), (9.9, 0.))
        assert_almost_equal(wcs.all_pix2world(21, 21, 1), (10.1, 0.1))
        assert_almost_equal(wcs.all_pix2world(1, 21, 1), (10.1, -0.1))
        assert_almost_equal(wcs.all_pix2world(1, 1, 1), (9.9, -0.1))

    def test_manual_wcs_angle(self):
        # Checked with DS9
        x, y = (11, 11)
        ra, dec = (10.0, 0.0)
        ps = 36  # arcsec/px
        north = 45
        wcs = wcs_from_coords(x, y, ra, dec, ps, north)
        assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
        assert_almost_equal(wcs.all_pix2world(21, 21, 1), (10.0, 0.14142))
        assert_almost_equal(wcs.all_pix2world(1, 1, 1), (10.0, -0.14142))
        assert_almost_equal(wcs.all_pix2world(1, 21, 1), (10.14142, 0.0))
        assert_almost_equal(wcs.all_pix2world(21, 1, 1), (9.858579, 0.0))

    def test_manual_wcs_top_flip_ra(self):
        # Checked with DS9
        x, y = (11, 11)
        ra, dec = (10.0, 0.0)
        ps = 36  # arcsec/px
        north = 'top'
        flip = 'ra'
        wcs = wcs_from_coords(x, y, ra, dec, ps, north, flip=flip)
        assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
        assert_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, 0.1))
        assert_almost_equal(wcs.all_pix2world(21, 11, 1), (10.1, 0.0))
        assert_almost_equal(wcs.all_pix2world(1, 11, 1), (9.9, 0.0))
        assert_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, -0.1))
        assert_almost_equal(wcs.all_pix2world(21, 21, 1), (10.1, 0.1))
        assert_almost_equal(wcs.all_pix2world(1, 21, 1), (9.9, 0.1))
        assert_almost_equal(wcs.all_pix2world(1, 1, 1), (9.9, -0.1))

    def test_manual_wcs_top_flip_dec(self):
        # Checked with DS9
        x, y = (11, 11)
        ra, dec = (10.0, 0.0)
        ps = 36  # arcsec/px
        north = 'top'
        flip = 'dec'
        wcs = wcs_from_coords(x, y, ra, dec, ps, north, flip=flip)
        assert_almost_equal(wcs.all_pix2world(11, 11, 1), (ra, dec))
        assert_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, -0.1))
        assert_almost_equal(wcs.all_pix2world(21, 11, 1), (9.9, 0.0))
        assert_almost_equal(wcs.all_pix2world(1, 11, 1), (10.1, 0.0))
        assert_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, 0.1))
        assert_almost_equal(wcs.all_pix2world(21, 21, 1), (9.9, -0.1))
        assert_almost_equal(wcs.all_pix2world(1, 21, 1), (10.1, -0.1))
        assert_almost_equal(wcs.all_pix2world(1, 1, 1), (10.1, 0.1))

    def test_manual_wcs_top_flip_all(self):
        # Checked with DS9
        x, y = (11, 11)
        ra, dec = (10.0, 0.0)
        ps = 36  # arcsec/px
        north = 'top'
        flip = 'all'
        wcs = wcs_from_coords(x, y, ra, dec, ps, north, flip=flip)
        assert_almost_equal(wcs.all_pix2world(11, 21, 1), (10.0, -0.1))
        assert_almost_equal(wcs.all_pix2world(21, 11, 1), (10.1, 0.0))
        assert_almost_equal(wcs.all_pix2world(1, 11, 1), (9.9, 0.0))
        assert_almost_equal(wcs.all_pix2world(11, 1, 1), (10.0, 0.1))
        assert_almost_equal(wcs.all_pix2world(21, 21, 1), (10.1, -0.1))
        assert_almost_equal(wcs.all_pix2world(1, 21, 1), (9.9, -0.1))
        assert_almost_equal(wcs.all_pix2world(1, 1, 1), (9.9, 0.1))

    def test_raise_north_angle(self):
        with pytest.raises(ValueError) as exc:
            wcs_from_coords(0, 0, 0, 0, 0, 'not a direction')
            assert_in('invalid value for north', str(exc.value))


class Test_GuessCoords:
    def test_guess_coords_float(self):
        ra = 10.0
        dec = 0.0
        assert_equal(guess_coordinates(ra, dec, skycoord=False), (ra, dec))

    def test_guess_coords_strfloat(self):
        ra = "10.0"
        dec = "-27.0"
        assert_equal(guess_coordinates(ra, dec, skycoord=False), (10, -27))

    def test_guess_coords_hexa_space(self):
        ra = "1 00 00"
        dec = "-43 30 00"
        assert_almost_equal(guess_coordinates(ra, dec, skycoord=False),
                            (15.0, -43.5))

    def test_guess_coords_hexa_dots(self):
        ra = "1:00:00"
        dec = "-43:30:00"
        assert_almost_equal(guess_coordinates(ra, dec, skycoord=False),
                            (15.0, -43.5))

    def test_guess_coords_skycord_float(self):
        ra = 10.0
        dec = 0.0
        sk = guess_coordinates(ra, dec, skycoord=True)
        assert_is_instance(sk, SkyCoord)
        assert_equal(sk.ra.degree, ra)
        assert_equal(sk.dec.degree, dec)

    def test_guess_coords_skycord_hexa(self):
        ra = "1:00:00"
        dec = "00:00:00"
        sk = guess_coordinates(ra, dec, skycoord=True)
        assert_is_instance(sk, SkyCoord)
        assert_true(sk.ra.degree - 15 < 1e-8)
        assert_true(sk.dec.degree - 0 < 1e-8)

    def test_guess_coords_list_hexa(self):
        ra = ["1:00:00", "2:30:00"]
        dec = ["00:00:00", "1:00:00"]
        sra, sdec = guess_coordinates(ra, dec, skycoord=False)
        assert_almost_equal(sra, [15, 37.5])
        assert_almost_equal(sdec, [0, 1])

    def test_guess_coords_list_float(self):
        ra = [10.0, 15, 20]
        dec = [0.0, 1.0, -1.0]
        sra, sdec = guess_coordinates(ra, dec, skycoord=False)
        assert_equal(sra, ra)
        assert_equal(sdec, dec)

    def test_guess_coords_list_diff(self):
        ra = np.arange(10)
        dec = np.arange(15)
        with pytest.raises(ValueError):
            guess_coordinates(ra, dec)

    def test_guess_coords_list_nolist(self):
        ra = np.arange(10)
        dec = 1
        with pytest.raises(ValueError):
            guess_coordinates(ra, dec)
