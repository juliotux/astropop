# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import os
import pytest
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.time import Time
from astropop.catalogs.simbad import SimbadSourcesCatalog, simbad_query_id
from astropop.catalogs import vizier
from astropop.catalogs.gaia import GaiaDR3SourcesCatalog
from astropop.catalogs._online_tools import _timeout_retry, \
                                            _fix_query_table, \
                                            get_center_radius, \
                                            astroquery_radius, \
                                            astroquery_skycoord, \
                                            astroquery_query
from astropop.catalogs._sources_catalog import _OnlineSourcesCatalog, \
                                               SourcesCatalog
from astropop.math import QFloat

from astropop.testing import *


sources = Table({'id': ['id1', 'id2', 'id3', 'id4'],
                 'ra': [2.44644404, 0.52522258, 0.64638169, 4.16520547],
                 'dec': [4.92305031, 3.65404807, 4.50588171, 3.80703142],
                 'pm_ra': [278.6, 114.3, 8.6, 270.1],
                 'pm_dec': [25.7, 202.6, 122.3, 256.3],
                 'mag': [2.16, 3.00, 3.55, 4.81],
                 'mag_error': [0.01, 0.02, 0.01, 0.03]})


class DummySourcesCatalog(_OnlineSourcesCatalog):
    _available_filters = ['A', 'B']

    def _setup_catalog(self):
        return

    def _do_query(self):
        self._query = Table(sources)
        mag = {'A': QFloat(sources['mag'], uncertainty=sources['mag_error'],
                           unit='mag'),
               'B': QFloat(sources['mag'], uncertainty=sources['mag_error'],
                           unit='mag')}
        obj_mag = None
        for i in self.filters:
            if obj_mag is None:
                obj_mag = {}
            obj_mag[i] = mag[i]
        SourcesCatalog.__init__(self, ra=sources['ra'],
                                dec=sources['dec'], unit='degree',
                                pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/year'),
                                pm_dec=sources['pm_dec']*u.Unit('mas/year'),
                                frame='icrs', obstime='J2005.0',
                                ids=sources['id'], mag=mag)


sirius_coords = ["Sirius", "06h45m09s -16d42m58s", [101.28715, -16.7161158],
                 np.array([101.28715, -16.7161158]), (101.28715, -16.7161158),
                 SkyCoord(101.28715, -16.7161158, unit=('degree', 'degree'))]
hd674_coords = ["HD 674", "00h10m52s -54d17m26s", [2.716748, -54.290647],
                np.array([2.716748, -54.290647]), (2.716748, -54.290647),
                SkyCoord(2.716748, -54.290647, unit='degree')]
search_radius = ['0.1d', 0.1, Angle('0.1d')]


@pytest.mark.flaky(reruns=5)
@pytest.mark.remote_data
class Test_OnlineTools:
    def test_timeout_retry_error(self):
        def _only_fail(*args, **kwargs):
            assert_equal(len(args), 1)
            assert_equal(args[0], 1)
            assert_equal(len(kwargs), 1)
            assert_equal(kwargs['test'], 2)
            raise TimeoutError

        with pytest.raises(TimeoutError, match='TimeOut obtained in'):
            _timeout_retry(_only_fail, 1, test=2)

    def test_timeout_retry_pass(self):
        i = 0

        def _only_fail(*args, **kwargs):
            nonlocal i
            assert_equal(len(args), 1)
            assert_equal(args[0], 1)
            assert_equal(len(kwargs), 1)
            assert_equal(kwargs['test'], 2)
            if i < 5:
                i += 1
                raise TimeoutError
            return i

        res = _timeout_retry(_only_fail, 1, test=2)
        assert_equal(res, 5)

    def test_wrap_table(self):
        class StrObj__:
            def __init__(self, s):
                self._s = s

            def __str__(self):
                return str(self._s)

        tab = Table()
        tab['a'] = ['A3#Â'.encode('utf-8') for i in range(10)]
        tab['b'] = ['B3#Ê' for i in range(10)]
        tab['c'] = [StrObj__(i) for i in range(10)]

        _fix_query_table(tab)

        assert_equal(len(tab), 10)
        assert_equal(tab['a'], ['A3#Â' for i in range(10)])
        assert_equal(tab['b'], ['B3#Ê' for i in range(10)])
        assert_equal(tab['c'], [f'{i}' for i in range(10)])
        assert_equal(tab['a'].dtype.char, 'U')
        assert_equal(tab['a'].dtype.char, 'U')
        assert_equal(tab['a'].dtype.char, 'U')

    def test_get_center_radius(self):
        ra = np.arange(11)
        dec = np.arange(11)
        c_ra, c_dec, rad = get_center_radius(ra, dec)
        assert_equal(c_ra, 5)
        assert_equal(c_dec, 5)
        assert_equal(rad, 10)

    @pytest.mark.parametrize('value', sirius_coords)
    def test_astroquery_skycoord_string_obj(self, value):
        skcord = astroquery_skycoord(value)
        assert_is_instance(skcord, SkyCoord)
        assert_almost_equal(skcord.ra.degree, 101.28715, decimal=3)
        assert_almost_equal(skcord.dec.degree, -16.7161158, decimal=3)

    def test_astroquery_skycoord_error(self):
        value = 'this should raise error'
        with pytest.raises(ValueError, match='could not be resolved'):
            astroquery_skycoord(value)

        with pytest.raises(ValueError, match='Center coordinates 1.0'):
            astroquery_skycoord(1.0)

    def test_astroquery_radius(self):
        ang = Angle(1.0, unit='degree')
        stra = '1d'
        strb = '3600 arcsec'
        inta = 1

        assert_equal(astroquery_radius(ang), Angle("1.0d"))
        assert_equal(astroquery_radius(stra), Angle("1.0d"))
        assert_equal(astroquery_radius(strb), Angle("1.0d"))
        assert_equal(astroquery_radius(inta), Angle("1.0d"))

    def test_astroquery_radius_error(self):
        not_angle = '10 not angle'
        with pytest.raises(ValueError, match='Invalid character at col'):
            astroquery_radius(not_angle)
        with pytest.raises(TypeError, match='not supported.'):
            astroquery_radius([])

    def test_astroquery_query_error(self):
        def return_none(a, b=1):
            return None
        with pytest.raises(RuntimeError, match='No online catalog result'):
            astroquery_query(return_none, 1, b=1)


class Test_DummySourcesCatalog:
    # Test things handled by the base catalog
    def test_catalog_creation(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0])
        assert_is_instance(c, SourcesCatalog)
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0],
                                band='A')
        assert_is_instance(c, SourcesCatalog)

        with pytest.raises(TypeError):
            DummySourcesCatalog()

    @pytest.mark.parametrize('radius', search_radius)
    @pytest.mark.parametrize('center', sirius_coords)
    def test_catalog_center_radius(self, center, radius):
        c = DummySourcesCatalog(center, radius)

        assert_is_instance(c.center, SkyCoord)
        assert_almost_equal(c.center.ra.degree, 101.287155, decimal=3)
        assert_almost_equal(c.center.dec.degree, -16.7161158, decimal=3)

        assert_is_instance(c.radius, Angle)
        assert_almost_equal(c.radius.degree, 0.1)

    def test_catalog_properties(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0])
        assert_is_instance(c.sources_id(), np.ndarray)
        assert_equal(c.sources_id().shape, (4))
        assert_is_instance(c.skycoord(), SkyCoord)
        assert_is_instance(c.ra_dec_list(), np.ndarray)
        assert_equal(c.ra_dec_list().shape, (4, 2))
        assert_is_instance(c.mag_list('A'), np.ndarray)
        assert_equal(c.mag_list('A').shape, (4, 2))

        assert_equal(c.sources_id(), sources['id'])
        assert_almost_equal(c.ra_dec_list()[:, 0], sources['ra'])
        assert_almost_equal(c.ra_dec_list()[:, 1], sources['dec'])
        assert_almost_equal(c.mag_list('A')[:, 0], sources['mag'])
        assert_almost_equal(c.mag_list('A')[:, 1], sources['mag_error'])

    def test_catalog_table(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0])
        t = c.table()
        assert_is_instance(t, Table)
        assert_equal(t.colnames, ['id', 'ra', 'dec', 'pm_ra_cosdec',
                                  'pm_dec', 'A', 'A_error',
                                  'B', 'B_error'])
        assert_equal(t['id'], sources['id'])
        assert_almost_equal(t['ra'], sources['ra'])
        assert_almost_equal(t['dec'], sources['dec'])
        assert_almost_equal(t['pm_ra_cosdec'], sources['pm_ra'])
        assert_almost_equal(t['pm_dec'], sources['pm_dec'])
        assert_almost_equal(t['A'], sources['mag'])
        assert_almost_equal(t['B_error'], sources['mag_error'])

    def test_catalog_getitem_number(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0])
        nc = c[0]
        assert_equal(nc.sources_id(), sources['id'][0])
        assert_almost_equal(nc.ra_dec_list(),
                            [[sources['ra'][0], sources['dec'][0]]])
        assert_almost_equal(nc.mag_list('A'),
                            [[sources['mag'][0], sources['mag_error'][0]]])

    def test_catalog_getitem_array(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0])
        for items in [[2, 3], slice(2, None), np.array([2, 3])]:
            nc = c[items]
            assert_equal(len(nc), 2)
            assert_equal(nc.sources_id(), sources['id'][2:])
            assert_almost_equal(nc.ra_dec_list()[:, 0], sources['ra'][2:])
            assert_almost_equal(nc.ra_dec_list()[:, 1], sources['dec'][2:])
            assert_almost_equal(nc.table()['pm_ra_cosdec'], sources['pm_ra'][2:])
            assert_almost_equal(nc.table()['pm_dec'], sources['pm_dec'][2:])
            assert_almost_equal(nc.mag_list('A')[:, 0], sources['mag'][2:])
            assert_almost_equal(nc.mag_list('A')[:, 1], sources['mag_error'][2:])

        # Ensure error is raised when a list of strings
        with pytest.raises(KeyError):
            c['id', 'ra']

        with pytest.raises(KeyError):
            c[(2, 3)]

    def test_catalog_getitem_columns(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0])
        for i in sources.colnames:
            assert_equal(c[i], sources[i])

        with pytest.raises(KeyError):
            c['no column']

    def test_catalog_len(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0])
        assert_equal(len(c), 4)
        assert_equal(len(c[0]), 1)

    def test_catalog_match_objects(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0])
        m = c.match_objects([0.52525258, 0.87265989, 4.16526547],
                            [3.65404807, 5.50588171, 3.80703142],
                            limit_angle='1 arcsec')
        assert_is_instance(m, SourcesCatalog)
        assert_equal(m.sources_id(), ['id2', '', 'id4'])
        assert_almost_equal(m.ra_dec_list()[:, 0], [0.52522258, np.nan, 4.16520547])
        assert_almost_equal(m.ra_dec_list()[:, 1], [3.65404807, np.nan, 3.80703142])
        assert_almost_equal(m.mag_list('A')[:, 0], [3.00, np.nan, 4.81], decimal=2)
        assert_almost_equal(m.mag_list('A')[:, 1], [0.02, np.nan, 0.03], decimal=2)
        assert_almost_equal(m.mag_list('B')[:, 0], [3.00, np.nan, 4.81], decimal=2)
        assert_almost_equal(m.mag_list('B')[:, 1], [0.02, np.nan, 0.03], decimal=2)

    def test_catalog_band(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0],
                                band='A')
        assert_equal(c.filters, ['A'])

        # default behavior is all
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0])
        assert_equal(c.filters, ['A', 'B'])
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0],
                                band='all')
        assert_equal(c.filters, ['A', 'B'])

        # None should have no filters
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0],
                                band=None)
        assert_equal(c.filters, [])

    def test_catalog_invalid_band(self):
        for i in ['C', ('A', 'C'), ['A', 'C']]:
            with pytest.raises(ValueError,
                               match='not available for this catalog'):
                c = DummySourcesCatalog(sirius_coords[0], search_radius[0],
                                        band=i)
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0])
        with pytest.raises(ValueError, match='not available'):
            c.mag_list('C')

        D = DummySourcesCatalog
        D._available_filters = None
        with pytest.raises(ValueError, match='No filters'):
            c = D(sirius_coords[0], search_radius[0], band='A')


@pytest.mark.flaky(reruns=5)
@pytest.mark.remote_data
class Test_Simbad():
    def test_simbad_creation_errors(self):
        # Need arguments
        with pytest.raises(TypeError):
            SimbadSourcesCatalog()
        with pytest.raises(TypeError):
            SimbadSourcesCatalog('test')

        # generic search ok
        SimbadSourcesCatalog('Sirius', '0.05d')

    @pytest.mark.parametrize('radius', search_radius)
    @pytest.mark.parametrize('center', sirius_coords)
    def test_csimbad_creation_params(self, center, radius):
        s = SimbadSourcesCatalog(center, radius, band='V')
        assert_equal(s.sources_id()[0], '* alf CMa')
        assert_almost_equal(s.ra_dec_list()[0], [101.287155, -16.7161158],
                            decimal=5)
        assert_almost_equal(s.magnitude('V')[0].nominal, -1.46)

        assert_is_instance(s.center, SkyCoord)
        assert_almost_equal(s.center.ra.degree, 101.287155, decimal=3)
        assert_almost_equal(s.center.dec.degree, -16.7161158, decimal=3)

        assert_is_instance(s.radius, Angle)
        assert_almost_equal(s.radius.degree, 0.1)

    def test_simbad_properties_types(self):
        s = SimbadSourcesCatalog(sirius_coords[0], search_radius[0], band='V')

        assert_is_instance(s.sources_id(), np.ndarray)
        assert_equal(s.sources_id().shape, (len(s)))
        assert_is_instance(s.skycoord(), SkyCoord)
        assert_is_instance(s.magnitude('V'), QFloat)
        assert_is_instance(s.ra_dec_list(), np.ndarray)
        assert_equal(s.ra_dec_list().shape, (len(s), 2))
        assert_is_instance(s.mag_list('V'), np.ndarray)
        assert_equal(s.mag_list('V').shape, (len(s), 2))
        assert_is_instance(s.coordinates_bibcode(), np.ndarray)
        assert_equal(s.coordinates_bibcode().shape, (len(s),))
        assert_is_instance(s.magnitudes_bibcode('V'), np.ndarray)
        assert_equal(s.magnitudes_bibcode('V').shape, (len(s),))

    def test_simbad_properties_table(self):
        s = SimbadSourcesCatalog(sirius_coords[0], search_radius[0], band='V')
        t = s.table()
        assert_is_instance(t, Table)
        assert_equal(len(t), len(s))
        colnames = ['id', 'ra', 'dec', 'pm_ra_cosdec', 'pm_dec',
                    'V', 'V_error']
        assert_equal(t.colnames, colnames)
        assert_equal(t['id'][0], '* alf CMa')
        assert_almost_equal(t['ra'][0], 101.287155)
        assert_almost_equal(t['dec'][0], -16.7161158)
        assert_almost_equal(t['V'][0], -1.46, decimal=2)
        assert_true(np.isnan(t['V_error'][0]))

    def test_simbad_default_band(self):
        s = SimbadSourcesCatalog(sirius_coords[0], search_radius[0])
        assert_equal(s.filters, [])
        assert_is_none(s._mags_table)
        with pytest.raises(ValueError, match='This SourcesCatalog has no'):
            s.mag_list('V')

    def test_simbad_bands_all(self):
        s = SimbadSourcesCatalog(sirius_coords[0], search_radius[0],
                                 band='all')
        assert_equal(s.filters, s._available_filters)

    def test_simbad_coords_bibcode(self):
        s = SimbadSourcesCatalog(sirius_coords[0], '10 arcsec')
        assert_is_instance(s.coordinates_bibcode(), np.ndarray)
        assert_equal(s.coordinates_bibcode()[0], '2007A&A...474..653V')

    def test_simbad_mags_bibcode(self):
        s = SimbadSourcesCatalog(sirius_coords[0], '10 arcsec',
                                 band=['V', 'B', 'R'])
        assert_equal(s.magnitudes_bibcode('V')[0], '2002yCat.2237....0D')
        assert_equal(s.magnitudes_bibcode('B')[0], '2002yCat.2237....0D')
        assert_equal(s.magnitudes_bibcode('R')[0], '2002yCat.2237....0D')

    def test_simbad_available_filters(self):
        s = SimbadSourcesCatalog(sirius_coords[0], '10 arcsec',
                                 band=['V', 'B', 'R'])
        assert_equal(s.available_filters,
                     ['B', 'V', 'R', 'I', 'J', 'H', 'K',
                      'u', 'g', 'r', 'i', 'z'])


@pytest.mark.flaky(reruns=5)
@pytest.mark.remote_data
class Test_SimbadQueryID:
    @pytest.mark.parametrize('order, expect', [(None, 'alf CMa'),
                                               (['NAME'], 'Dog Star'),
                                               (['*'], 'alf CMa'),
                                               (['HIP'], 'HIP 32349'),
                                               (['HIC', 'HD'], 'HIC 32349'),
                                               (['NONE', 'HD'], 'HD 48915'),
                                               (['UBV M', 'HD'],
                                                'UBV M 12413')])
    def test_simbad_query_id(self, order, expect):
        idn = simbad_query_id(101.28715, -16.7161158, '5s', name_order=order)
        assert_equal(idn, expect)

    @pytest.mark.parametrize('coords,name', [((16.82590917, -72.4676825),
                                              'HD 6884'),
                                             ((86.46641167, -67.24053806),
                                              'LHA 120-S 61')])
    def test_simbad_query_id_non_default(self, coords, name):
        order = ['NAME', 'HD', 'HR', 'HYP', 'AAVSO', 'LHA']
        idn = simbad_query_id(*coords, '5s', name_order=order)
        assert_equal(idn, name)

    def test_simbad_query_id_multiple(self):
        ra = [101.28715, 88.79293875, 191.93028625]
        dec = [-16.7161158, 7.40706389, -59.68877194]
        name = ['alf CMa', 'alf Ori', 'bet Cru']
        res = simbad_query_id(ra, dec, '2s')
        assert_equal(res, name)


@pytest.mark.flaky(reruns=5)
class Test_VizierGeneral:
    def test_print_available_catalogs(self):
        assert_is_instance(vizier.list_vizier_catalogs(), str)
        assert_equal(vizier.list_vizier_catalogs()[:45],
                     'Available pre-configured Vizier catalogs are:')
        assert_equal(vizier.list_vizier_catalogs.__doc__,
                     "List available vizier catalogs\n\n"
                     'Notes\n-----\n'+vizier.list_vizier_catalogs())

    def test_query_fail(self):
        with pytest.raises(RuntimeError,
                           match='An error occured during online query.'):
            vizier.vsx('HD 674', '0.5 arcsec')

    def test_create_with_file(self):
        file = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'astropop', 'catalogs', 'vizier_catalogs', 'ucac4.yml')
        file = os.path.abspath(file)
        cat = vizier.VizierSourcesCatalog(file, 'HD 674', '0.5 arcsec')
        assert_equal(cat.name, 'ucac4')
        assert_equal(cat._conf['description'], "UCAC4 Catalogue (Zacharias+, 2012)")
        assert_in('ucac4', cat.help())


@pytest.mark.flaky(reruns=5)
# @pytest.mark.remote_data
class Test_Vizier_UCAC4:
    hd674_mags = {
        'J': [10.157, 0.02],
        'H': [10.083, 0.02],
        'K': [10.029, 0.02],
        'B': [10.752, 0.01],
        'V': [10.597, 0.07],
        'i': [10.688, 0.01]
    }

    def test_ucac4_creation_errors(self):
        # Need arguments
        with pytest.raises(TypeError):
            vizier.ucac4()
        with pytest.raises(TypeError):
            vizier.ucac4('test')

        with pytest.raises(ValueError, match='Filter None not available.'):
            vizier.ucac4('Sirius', '0.05d', band='None')
        # Filter None should pass, no mag data
        vizier.ucac4('Sirius', '0.05d', None)

    def test_ucac4_creation_filters(self):
        c = vizier.ucac4(hd674_coords[0], '0.05d')

        assert_equal(c.sources_id()[0], 'UCAC4 179-000175')
        for k, v in self.hd674_mags.items():
            assert_almost_equal(c.mag_list(k)[0], v)

    @pytest.mark.parametrize('radius', search_radius)
    @pytest.mark.parametrize('center', sirius_coords)
    def test_ucac4_query_input_types(self, center, radius):
        c = vizier.ucac4(center, radius, band='V')

        assert_equal(c.sources_id()[0], 'UCAC4 367-016700')
        assert_almost_equal(c.ra_dec_list()[0], [101.28715, -16.7161158], decimal=5)
        assert_almost_equal(c.mag_list('V')[0], [-1.440, 0.0])

    def test_ucac4_properties_types(self):
        s = vizier.ucac4(sirius_coords[0], search_radius[0], band='V')

        assert_is_instance(s.sources_id(), np.ndarray)
        assert_equal(s.sources_id().shape, (len(s)))
        assert_is_instance(s.skycoord(), SkyCoord)
        assert_is_instance(s.magnitude('V'), QFloat)
        assert_is_instance(s.ra_dec_list(), np.ndarray)
        assert_equal(s.ra_dec_list().shape, (len(s), 2))
        assert_is_instance(s.mag_list('V'), np.ndarray)
        assert_equal(s.mag_list('V').shape, (len(s), 2))

    def test_ucac4_help(self):
        help = vizier.ucac4('hd 674', '10 arcsec').help()
        assert_equal(help[:5], 'ucac4')
        assert_in('Available filters are:', help)
        for i in self.hd674_mags.keys():
            assert_in(f'  - {i}:', help)

    def test_ucac4_properties(self):
        s = vizier.ucac4(sirius_coords[0], search_radius[0], band=['V'])
        assert_equal(s.available_filters,
                     ['J', 'H', 'K', 'B', 'V', 'g', 'r', 'i'])
        assert_equal(s.name, 'ucac4')
        assert_is_instance(s.center, SkyCoord)
        assert_is_instance(s.radius, Angle)
        assert_equal(s.filters, ['V'])


@pytest.mark.flaky(reruns=5)
@pytest.mark.remote_data
class Test_Vizier_APASS9:
    hd674_mags = {
        'B': [10.775, 0.031],
        'V': [10.626, 0.043],
        "g_": [10.783, 0.104],
        "r_": [10.675, 0.047],
        "i_": [10.726, 0.077]
    }

    def test_apass9_creation_errors(self):
        # Need arguments
        with pytest.raises(TypeError):
            vizier.apass9()
        with pytest.raises(TypeError):
            vizier.apass9('test')

        with pytest.raises(ValueError, match='Filter None not available.'):
            vizier.apass9('Sirius', '0.05d', band='None')
        # Filter None should pass, no mag data
        vizier.apass9('Sirius', '0.05d', None)

    @pytest.mark.parametrize('radius', search_radius)
    @pytest.mark.parametrize('center', hd674_coords)
    def test_apass9_query_input_types(self, center, radius):
        c = vizier.apass9(center, radius, band='V')

        assert_equal(c.sources_id()[0], '')
        assert_almost_equal(c.ra_dec_list()[0], [2.716748, -54.290647], decimal=5)
        assert_almost_equal(c.mag_list('V')[0], [10.626, 0.043])

    def test_apass9_creation_filters(self):
        c = vizier.apass9(hd674_coords[0], '0.05d')

        assert_equal(c.sources_id()[0], '')
        for k, v in self.hd674_mags.items():
            assert_almost_equal(c.mag_list(k)[0], v)

    def test_apass9_properties_types(self):
        s = vizier.apass9(hd674_coords[0], search_radius[0], band='V')

        assert_is_instance(s.sources_id(), np.ndarray)
        assert_equal(s.sources_id().shape, (len(s)))
        assert_is_instance(s.skycoord(), SkyCoord)
        assert_is_instance(s.magnitude('V'), QFloat)
        assert_is_instance(s.ra_dec_list(), np.ndarray)
        assert_equal(s.ra_dec_list().shape, (len(s), 2))
        assert_is_instance(s.mag_list('V'), np.ndarray)
        assert_equal(s.mag_list('V').shape, (len(s), 2))

    def test_apass9_help(self):
        help = vizier.apass9('hd 674', '10 arcsec').help()
        assert_equal(help[:6], 'apass9')
        assert_in('Available filters are:', help)
        for i in self.hd674_mags.keys():
            assert_in(f'  - {i}:', help)

    def test_apass9_properties(self):
        s = vizier.apass9(hd674_coords[0], search_radius[0], band=['V'])
        assert_equal(s.available_filters,
                     ['B', 'V', "g_", "r_", "i_"])
        assert_equal(s.name, 'apass9')
        assert_is_instance(s.center, SkyCoord)
        assert_is_instance(s.radius, Angle)
        assert_equal(s.filters, ['V'])


@pytest.mark.flaky(reruns=5)
@pytest.mark.remote_data
class Test_Vizier_GSC242:
    hd674_mags = {
        'G': [10.551350, 0.000404],
        'Bj': [np.nan, np.nan],
        'Fpg': [np.nan, np.nan],
        'Epg': [np.nan, np.nan],
        'Npg': [np.nan, np.nan],
        'U': [np.nan, np.nan],
        'B': [10.775, 0.031],
        'V': [10.626, 0.043],
        'u': [11.902, 0.014],
        'g': [10.783, 0.104],
        'r': [10.675, 0.047],
        'i': [10.726, 0.077],
        'z': [10.840, 0.007],
        'y': [np.nan, np.nan],
        'J': [11.055, 0.00],
        'H': [10.083, 0.022],
        'Ks': [10.313, 0.001],
        'Z': [np.nan, np.nan],
        'Y': [np.nan, np.nan],
        'W1': [10.026, 0.023],
        'W2': [10.044, 0.020],
        'W3': [10.043, 0.051],
        'W4': [8.522, np.nan],
        'FUV': [16.229, 0.031],
        'NUV': [13.755, 0.007]
    }

    def test_gsc242_creation_errors(self):
        # Need arguments
        with pytest.raises(TypeError):
            vizier.gsc242()
        with pytest.raises(TypeError):
            vizier.gsc242('test')

        with pytest.raises(ValueError, match='Filter None not available.'):
            vizier.gsc242('Sirius', '0.05d', band='None')
        # Filter None should pass, no mag data
        vizier.gsc242('Sirius', '0.05d', None)

    @pytest.mark.parametrize('radius', search_radius)
    @pytest.mark.parametrize('center', hd674_coords)
    def test_gsc242_query_input_types(self, center, radius):
        c = vizier.gsc242(center, radius, band='V')

        assert_equal(c.sources_id()[0], 'GSC2 S17J000168')
        assert_almost_equal(c.ra_dec_list()[0], [2.7168074, -54.2906086], decimal=5)
        assert_almost_equal(c.mag_list('V')[0], [10.626, 0.043])

    def test_gsc242_creation_filters(self):
        c = vizier.gsc242(hd674_coords[0], '0.05d')

        assert_equal(c.sources_id()[0], 'GSC2 S17J000168')
        for k, v in self.hd674_mags.items():
            assert_almost_equal(c.mag_list(k)[0], v, decimal=3)

    def test_gsc242_properties_types(self):
        s = vizier.gsc242(hd674_coords[0], search_radius[0], band='V')

        assert_is_instance(s.sources_id(), np.ndarray)
        assert_equal(s.sources_id().shape, (len(s)))
        assert_is_instance(s.skycoord(), SkyCoord)
        assert_is_instance(s.magnitude('V'), QFloat)
        assert_is_instance(s.ra_dec_list(), np.ndarray)
        assert_equal(s.ra_dec_list().shape, (len(s), 2))
        assert_is_instance(s.mag_list('V'), np.ndarray)
        assert_equal(s.mag_list('V').shape, (len(s), 2))

    def test_gsc242_help(self):
        help = vizier.gsc242('hd 674', '10 arcsec').help()
        assert_equal(help[:6], 'gsc242')
        assert_in('Available filters are:', help)
        for i in self.hd674_mags.keys():
            assert_in(f'  - {i}:', help)

    def test_gsc242_properties(self):
        s = vizier.gsc242(hd674_coords[0], search_radius[0], band=['V'])
        assert_equal(s.available_filters,
                     ['G', 'RP', 'BP', 'Bj', 'Fpg', 'Epg', 'Npg', 'U', 'B', 'V', 'u',
                      'g', 'r', 'i', 'z', 'y', 'J', 'H', 'Ks', 'Z', 'Y', 'W1', 'W2',
                      'W3', 'W4', 'FUV', 'NUV'])
        assert_equal(s.name, 'gsc242')
        assert_is_instance(s.center, SkyCoord)
        assert_is_instance(s.radius, Angle)
        assert_equal(s.filters, ['V'])


@pytest.mark.flaky(reruns=5)
@pytest.mark.remote_data
class Test_VSXVizierCatalog:
    def test_vsx_creation_errors(self):
        # Need arguments
        with pytest.raises(TypeError):
            vizier.vsx()
        with pytest.raises(TypeError):
            vizier.vsx('test')

        with pytest.raises(ValueError, match='No filters available'):
            vizier.vsx('Sirius', '0.05d', band='None')
        # Filter None should pass, no mag data
        vizier.vsx('Sirius', '0.05d', None)

    def test_vsx_creation_filters(self):
        with pytest.raises(ValueError, match='No filters available'):
            vizier.vsx('Sirius', '0.05d', band='V')

        # none and all should pass
        vizier.vsx('Sirius', '0.05d', band=None)
        vizier.vsx('Sirius', '0.05d', band='all')

    def test_vsx_properties_types(self):
        s = vizier.vsx('RMC 40', search_radius[0])

        assert_is_instance(s.sources_id(), np.ndarray)
        assert_equal(s.sources_id().shape, (len(s)))
        assert_is_instance(s.skycoord(), SkyCoord)
        assert_is_instance(s.ra_dec_list(), np.ndarray)
        assert_equal(s.ra_dec_list().shape, (len(s), 2))

        # since no filters are available
        with pytest.raises(ValueError, match=' no photometic information.'):
            s.magnitude('V')
        with pytest.raises(ValueError, match=' no photometic information.'):
            s.mag_list('V')

    def test_vsx_getting_ids(self):
        s = vizier.vsx('RMC 40', '10 arcsec')

        assert_equal(s.sources_id()[0], 'VSX 234935')
        assert_equal(s['Name'][0], 'SMC V2018')

    def test_vsx_help(self):
        help = vizier.vsx('RMC 40', '10 arcsec').help()
        assert_equal(help[:3], 'vsx')
        assert_in("This catalog has no photometric informations.", help)

    def test_vsx_properties(self):
        s = vizier.vsx('RMC 40', '10 arcsec')
        assert_equal(s.available_filters, [])
        assert_equal(s.name, 'vsx')
        assert_is_instance(s.center, SkyCoord)
        assert_is_instance(s.radius, Angle)
        assert_equal(s.filters, [])


@pytest.mark.flaky(reruns=5)
class Test_2MASSVizierSourcesCatalog:
    hd674_mags = {
        'J': [10.157, 0.021],
        'H': [10.083, 0.022],
        'K': [10.029, 0.021]
    }

    def test_twomass_creation_errors(self):
        # Need arguments
        with pytest.raises(TypeError):
            vizier.twomass()
        with pytest.raises(TypeError):
            vizier.twomass('test')

        with pytest.raises(ValueError, match='Filter None not available.'):
            vizier.twomass('Sirius', '0.05d', band='None')
        # Filter None should pass, no mag data
        vizier.twomass('Sirius', '0.05d', None)

    def test_twomass_creation_filters(self):
        c = vizier.twomass(hd674_coords[0], '0.05d')

        assert_equal(c.sources_id()[0], '2MASS 00105201-5417264')
        for k, v in self.hd674_mags.items():
            assert_almost_equal(c.mag_list(k)[0], v)

    @pytest.mark.parametrize('radius', search_radius)
    @pytest.mark.parametrize('center', sirius_coords)
    def test_twomass_query_input_types(self, center, radius):
        c = vizier.twomass(center, radius, band='J')

        assert_equal(c.sources_id()[0], '2MASS 06450887-1642566')
        assert_almost_equal(c.ra_dec_list()[0], [101.286999, -16.715742], decimal=5)
        assert_almost_equal(c.mag_list('J')[0], [-1.391, 0.109])

    def test_twomass_properties_types(self):
        s = vizier.twomass(sirius_coords[0], search_radius[0], band='J')

        assert_is_instance(s.sources_id(), np.ndarray)
        assert_equal(s.sources_id().shape, (len(s)))
        assert_is_instance(s.skycoord(), SkyCoord)
        assert_is_instance(s.magnitude('J'), QFloat)
        assert_is_instance(s.ra_dec_list(), np.ndarray)
        assert_equal(s.ra_dec_list().shape, (len(s), 2))
        assert_is_instance(s.mag_list('J'), np.ndarray)
        assert_equal(s.mag_list('J').shape, (len(s), 2))

    def test_twomass_help(self):
        help = vizier.twomass('hd 674', '10 arcsec').help()
        assert_equal(help[:7], 'twomass')
        assert_in('Available filters are:', help)
        for i in self.hd674_mags.keys():
            assert_in(f'  - {i}:', help)

    def test_twomass_properties(self):
        s = vizier.twomass('hd 674', '10 arcsec')
        assert_equal(s.available_filters, ['J', 'H', 'K'])
        assert_equal(s.name, 'twomass')
        assert_is_instance(s.center, SkyCoord)
        assert_is_instance(s.radius, Angle)
        assert_equal(s.filters, ['J', 'H', 'K'])


@pytest.mark.flaky(reruns=5)
class Test_WISEVizierSourcesCatalog:
    hd674_mags = {
        'W1': [10.013, 0.024],
        'W2': [10.038, 0.020],
        'W3': [10.064, 0.046],
        'W4': [9.086, 0.445],
        'J': [10.157, 0.021],
        'H': [10.083, 0.022],
        'K': [10.029, 0.021]
    }

    def test_wise_creation_errors(self):
        # Need arguments
        with pytest.raises(TypeError):
            vizier.wise()
        with pytest.raises(TypeError):
            vizier.wise('test')

        with pytest.raises(ValueError, match='Filter None not available.'):
            vizier.wise('Sirius', '0.05d', band='None')
        # Filter None should pass, no mag data
        vizier.wise('Sirius', '0.05d', None)

    def test_wise_creation_filters(self):
        c = vizier.wise(hd674_coords[0], '0.05d')

        assert_equal(c.sources_id()[0], 'WISE J001052.01-541726.2')
        for k, v in self.hd674_mags.items():
            assert_almost_equal(c.mag_list(k)[0], v)

    @pytest.mark.parametrize('radius', search_radius)
    @pytest.mark.parametrize('center', sirius_coords)
    def test_wise_query_input_types(self, center, radius):
        c = vizier.wise(center, radius, band='W1')

        assert_equal(c.sources_id()[0], 'WISE J064508.27-164302.3')
        assert_almost_equal(c.ra_dec_list()[0], [101.284486, -16.717310], decimal=5)
        assert_almost_equal(c.mag_list('W1')[0], [2.387, 0.059])

    def test_wise_properties_types(self):
        s = vizier.wise(sirius_coords[0], search_radius[0], band='W1')

        assert_is_instance(s.sources_id(), np.ndarray)
        assert_equal(s.sources_id().shape, (len(s)))
        assert_is_instance(s.skycoord(), SkyCoord)
        assert_is_instance(s.magnitude('W1'), QFloat)
        assert_is_instance(s.ra_dec_list(), np.ndarray)
        assert_equal(s.ra_dec_list().shape, (len(s), 2))
        assert_is_instance(s.mag_list('W1'), np.ndarray)
        assert_equal(s.mag_list('W1').shape, (len(s), 2))

    def test_wise_help(self):
        help = vizier.wise('hd 674', '10 arcsec').help()
        assert_equal(help[:4], 'wise')
        assert_in('Available filters are:', help)
        for i in self.hd674_mags.keys():
            assert_in(f'  - {i}:', help)

    def test_wise_properties(self):
        s = vizier.wise('hd 674', '10 arcsec', band=['W1', 'W2', 'W3', 'W4'])
        assert_equal(s.available_filters, ['W1', 'W2', 'W3', 'W4', 'J', 'H', 'K'])
        assert_equal(s.name, 'wise')
        assert_is_instance(s.center, SkyCoord)
        assert_is_instance(s.radius, Angle)
        assert_equal(s.filters, ['W1', 'W2', 'W3', 'W4'])


@pytest.mark.flaky(reruns=5)
class Test_AllWISEVizierSourcesCatalog:
    hd674_mags = {
        'W1': [10.026, 0.023],
        'W2': [10.044, 0.020],
        'W3': [10.043, 0.051],
        'W4': [8.522, np.nan],
        'J': [10.157, 0.021],
        'H': [10.083, 0.022],
        'K': [10.029, 0.021]
    }

    def test_allwise_creation_errors(self):
        # Need arguments
        with pytest.raises(TypeError):
            vizier.allwise()
        with pytest.raises(TypeError):
            vizier.allwise('test')

        with pytest.raises(ValueError, match='Filter None not available.'):
            vizier.allwise('Sirius', '0.05d', band='None')
        # Filter None should pass, no mag data
        vizier.allwise('Sirius', '0.05d', None)

    def test_allwise_creation_filters(self):
        c = vizier.allwise(hd674_coords[0], '0.05d')

        assert_equal(c.sources_id()[0], 'AllWISE J001052.02-541726.3')
        for k, v in self.hd674_mags.items():
            assert_almost_equal(c.mag_list(k)[0], v)

    @pytest.mark.parametrize('radius', search_radius)
    @pytest.mark.parametrize('center', sirius_coords)
    def test_allwise_query_input_types(self, center, radius):
        c = vizier.allwise(center, radius, band='W1')

        assert_equal(c.sources_id()[0], 'AllWISE J064508.94-164303.4')
        assert_almost_equal(c.ra_dec_list()[0], [101.2872632, -16.7176349], decimal=5)
        assert_almost_equal(c.mag_list('W1')[0], [1.883, np.nan])

    def test_allwise_properties_types(self):
        s = vizier.allwise(sirius_coords[0], search_radius[0], band='W1')

        assert_is_instance(s.sources_id(), np.ndarray)
        assert_equal(s.sources_id().shape, (len(s)))
        assert_is_instance(s.skycoord(), SkyCoord)
        assert_is_instance(s.magnitude('W1'), QFloat)
        assert_is_instance(s.ra_dec_list(), np.ndarray)
        assert_equal(s.ra_dec_list().shape, (len(s), 2))
        assert_is_instance(s.mag_list('W1'), np.ndarray)
        assert_equal(s.mag_list('W1').shape, (len(s), 2))

    def test_allwise_help(self):
        help = vizier.allwise('hd 674', '10 arcsec').help()
        assert_equal(help[:7], 'allwise')
        assert_in('Available filters are:', help)
        for i in self.hd674_mags.keys():
            assert_in(f'  - {i}:', help)

    def test_allwise_properties(self):
        s = vizier.allwise('hd 674', '10 arcsec', band=['W1', 'W2', 'W3', 'W4'])
        assert_equal(s.available_filters, ['W1', 'W2', 'W3', 'W4', 'J', 'H', 'K'])
        assert_equal(s.name, 'allwise')
        assert_is_instance(s.center, SkyCoord)
        assert_is_instance(s.radius, Angle)
        assert_equal(s.filters, ['W1', 'W2', 'W3', 'W4'])


@pytest.mark.flaky(reruns=5)
class Test_Tycho2VizierSourcesCatalog:
    hd674_mags = {
        'BT': [10.809, 0.034],
        'VT': [10.589, 0.037]
    }

    def test_tycho2_creation_errors(self):
        # Need arguments
        with pytest.raises(TypeError):
            vizier.tycho2()
        with pytest.raises(TypeError):
            vizier.tycho2('test')

        with pytest.raises(ValueError, match='Filter None not available.'):
            vizier.tycho2('Sirius', '0.05d', band='None')
        # Filter None should pass, no mag data
        vizier.tycho2('HD 674', '0.05d', None)

    def test_tycho2_creation_filters(self):
        c = vizier.tycho2(hd674_coords[0], '0.05d')

        assert_equal(c.sources_id()[0], 'TYC 8464-1386-1')
        for k, v in self.hd674_mags.items():
            assert_almost_equal(c.mag_list(k)[0], v)

    @pytest.mark.parametrize('radius', search_radius)
    @pytest.mark.parametrize('center', hd674_coords)
    def test_tycho2_query_input_types(self, center, radius):
        c = vizier.tycho2(center, radius, band='VT')

        assert_equal(c.sources_id()[0], 'TYC 8464-1386-1')
        assert_almost_equal(c.ra_dec_list()[0], [2.71675147, -54.29065894],
                            decimal=5)
        assert_almost_equal(c.mag_list('VT')[0], self.hd674_mags['VT'])

    def test_tycho2_properties_types(self):
        s = vizier.tycho2(hd674_coords[0], search_radius[0], band='VT')

        assert_is_instance(s.sources_id(), np.ndarray)
        assert_equal(s.sources_id().shape, (len(s)))
        assert_is_instance(s.skycoord(), SkyCoord)
        assert_is_instance(s.magnitude('VT'), QFloat)
        assert_is_instance(s.ra_dec_list(), np.ndarray)
        assert_equal(s.ra_dec_list().shape, (len(s), 2))
        assert_is_instance(s.mag_list('VT'), np.ndarray)
        assert_equal(s.mag_list('VT').shape, (len(s), 2))

    def test_tycho2_help(self):
        help = vizier.tycho2('hd 674', '10 arcsec').help()
        assert_equal(help[:6], 'tycho2')
        assert_in('Available filters are:', help)
        for i in self.hd674_mags.keys():
            assert_in(f'  - {i}:', help)

    def test_tycho2_properties(self):
        s = vizier.tycho2('hd 674', '10 arcsec', band=['BT', 'VT'])
        assert_equal(s.available_filters, ['BT', 'VT'])
        assert_equal(s.name, 'tycho2')
        assert_is_instance(s.center, SkyCoord)
        assert_is_instance(s.radius, Angle)
        assert_equal(s.filters, ['BT', 'VT'])


@pytest.mark.flaky(reruns=5)
# @pytest.mark.remote_data
class Test_GaiaDR3:
    hd674_mags = {
        'G': [10.552819, 0.000337826],
        'BP': [10.649535, 0.00091911],
        'RP': [10.365023, 0.00060423]
    }

    def test_gaiadr3_creation_errors(self):
        # Need arguments
        with pytest.raises(TypeError):
            GaiaDR3SourcesCatalog()
        with pytest.raises(TypeError):
            GaiaDR3SourcesCatalog('test')

        with pytest.raises(ValueError, match='Filter None not available.'):
            GaiaDR3SourcesCatalog('Sirius', '0.05d', band='None')
        # Filter None should pass, no mag data
        GaiaDR3SourcesCatalog('Sirius', '0.05d', None)

    def test_gaiadr3_creation_filters(self):
        c = GaiaDR3SourcesCatalog(hd674_coords[0], '0.01d')

        assert_equal(c.sources_id()[0], 'Gaia DR3 4923784391133336960')
        for k, v in self.hd674_mags.items():
            assert_almost_equal(c.mag_list(k)[0], v)

    def test_gaiadr3_properties_types(self):
        s = GaiaDR3SourcesCatalog(hd674_coords[0],
                                  search_radius[0],
                                  band='G')

        assert_is_instance(s.sources_id(), np.ndarray)
        assert_equal(s.sources_id().shape, (len(s)))
        assert_is_instance(s.skycoord(), SkyCoord)
        assert_is_instance(s.magnitude('G'), QFloat)
        assert_is_instance(s.ra_dec_list(), np.ndarray)
        assert_equal(s.ra_dec_list().shape, (len(s), 2))
        assert_is_instance(s.mag_list('G'), np.ndarray)
        assert_equal(s.mag_list('G').shape, (len(s), 2))
        # properties only for gaia
        assert_is_instance(s.parallax(), QFloat)
        assert_is_instance(s.radial_velocity(), QFloat)
        # flags
        assert_is_instance(s.non_single_star(), np.ndarray)
        assert_equal(s.non_single_star().shape, (len(s),))
        assert_equal(s.non_single_star().dtype, np.int16)
        assert_is_instance(s.phot_variable_flag(), np.ndarray)
        assert_equal(s.phot_variable_flag().shape, (len(s),))
        assert_equal(s.phot_variable_flag().dtype, bool)
        assert_is_instance(s.in_galaxy_candidates(), np.ndarray)
        assert_equal(s.in_galaxy_candidates().shape, (len(s),))
        assert_equal(s.in_galaxy_candidates().dtype, bool)

    def test_gaiadr3_properties(self):
        s = GaiaDR3SourcesCatalog(hd674_coords[0], search_radius[0],
                                  band=['G'])
        assert_equal(s.available_filters,
                     ['G', 'BP', 'RP'])
        assert_is_instance(s.center, SkyCoord)
        assert_is_instance(s.radius, Angle)
        assert_equal(s.filters, ['G'])

    def test_gaiadr3_sources_limit(self):
        # test for bug #94
        # when using non-async functions, the output is limited to 2000 sources
        g = GaiaDR3SourcesCatalog('RMC 40', radius='15 arcmin')
        assert_greater(len(g), 2001)

    def test_gaiadr3_max_mag(self):
        s = GaiaDR3SourcesCatalog(hd674_coords[0], '1.0 deg',
                                  band=['G'], max_g_mag=15)
        # GAIA DR3 has only 741 sources brighter than 15 mag in this radius
        assert_less(len(s), 1000)
        assert_false(np.any(s.magnitude('G').nominal > 15))

    def test_gaiadr3_pm_in_skycoord(self):
        s = GaiaDR3SourcesCatalog(hd674_coords[0], '0.1 deg',
                                  band=['G'])
        assert_is_instance(s.skycoord().pm_ra_cosdec, u.Quantity)
        assert_is_instance(s.skycoord().pm_dec, u.Quantity)
        assert_almost_equal(s.skycoord().pm_ra_cosdec.value[0],
                            6.221, decimal=3)
        assert_almost_equal(s.skycoord().pm_dec.value[0],
                            9.578, decimal=3)

    def test_gaiadr3_pm_in_get_coordinates(self):
        s = GaiaDR3SourcesCatalog(hd674_coords[0], '0.1 deg',
                                  band=['G'])
        a = s.get_coordinates()
        c = s.get_coordinates(obstime=Time('J10016.0'))
        # space motion must be applyied
        assert_not_equal(c.ra.degree, a.ra.degree)
        assert_not_equal(c.dec.degree, a.dec.degree)


@pytest.mark.flaky(reruns=5)
# @pytest.mark.remote_data
class Test_SMSSDR4:
    hd674_mags = {
        'u': [11.9227, 0.0135],
        'v': [11.1721, 0.0140],
        'g': [10.5768, 0.0110],
        'r': [10.6059, 0.0117],
        'i': [10.7294, 0.0104],
        'z': [10.8096, 0.0119]
    }

    def test_smssdr4_creation_errors(self):
        # Need arguments
        with pytest.raises(TypeError):
            vizier.skymapper()
        with pytest.raises(TypeError):
            vizier.skymapper('test')

        with pytest.raises(ValueError, match='Filter None not available.'):
            vizier.skymapper('Sirius', '0.05d', band='None')
        # Filter None should pass, no mag data
        vizier.skymapper('HD 674', '0.05d', None)

    def test_smssdr4_creation_filters(self):
        c = vizier.skymapper(hd674_coords[0], '0.01d')

        assert_equal(c.sources_id()[0], 'SMSS-DR4 503264202')
        for k, v in self.hd674_mags.items():
            assert_almost_equal(c.mag_list(k)[0], v, decimal=3)

    def test_smssdr4_properties_types(self):
        s = vizier.skymapper(hd674_coords[0], search_radius[0])

        assert_is_instance(s.sources_id(), np.ndarray)
        assert_equal(s.sources_id().shape, (len(s)))
        assert_is_instance(s.skycoord(), SkyCoord)
        for i in ['u', 'v', 'g', 'z', 'r', 'i']:
            assert_is_instance(s.magnitude(i), QFloat)
            assert_is_instance(s.mag_list(i), np.ndarray)
            assert_equal(s.mag_list(i).shape, (len(s), 2))
        assert_is_instance(s.ra_dec_list(), np.ndarray)
        assert_equal(s.ra_dec_list().shape, (len(s), 2))

    def test_skymapper_help(self):
        help = vizier.skymapper('hd 674', '10 arcsec').help()
        assert_equal(help[:9], 'skymapper')
        assert_in('Available filters are:', help)
        for i in self.hd674_mags.keys():
            assert_in(f'  - {i}:', help)

    def test_skymapper_properties(self):
        s = vizier.skymapper('hd 674', '10 arcsec',
                             band=['u', 'v', 'g', 'r', 'i', 'z'])
        assert_equal(s.available_filters, ['u', 'v', 'g', 'r', 'i', 'z'])
        assert_equal(s.name, 'skymapper')
        assert_is_instance(s.center, SkyCoord)
        assert_is_instance(s.radius, Angle)
        assert_equal(s.filters, ['u', 'v', 'g', 'r', 'i', 'z'])
