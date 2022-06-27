# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import os
import time
import pytest
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropop.catalogs.simbad import SimbadSourcesCatalog, simbad_query_id
from astropop.catalogs.vizier import _VizierSourcesCatalog, \
                                     UCAC4SourcesCatalog, \
                                     UCAC5SourcesCatalog
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


def delay_rerun(*args):
    time.sleep(10)
    return True


class DummySourcesCatalog(_OnlineSourcesCatalog):
    sources = Table({'id': ['id1', 'id2', 'id3', 'id4'],
                     'ra': [2.44644404, 0.52522258, 0.64638169, 4.16520547],
                     'dec': [4.92305031, 3.65404807, 4.50588171, 3.80703142],
                     'pm_ra': [278.6, 114.3, 8.6, 270.1],
                     'pm_dec': [25.7, 202.6, 122.3, 256.3],
                     'mag': [2.16, 3.00, 3.55, 4.81],
                     'mag_error': [0.01, 0.02, 0.01, 0.03]})
    _available_filters = ['A', 'B']

    def _setup_catalog(self):
        return

    def _do_query(self):
        self._query = Table(self.sources)
        mag = QFloat(self.sources['mag'], uncertainty=self.sources['mag_error'],
                     unit='mag')
        SourcesCatalog.__init__(self, ra=self.sources['ra'],
                                dec=self.sources['dec'], unit='degree',
                                pm_ra_cosdec=self.sources['pm_ra']*u.Unit('mas/year'),
                                pm_dec=self.sources['pm_dec']*u.Unit('mas/year'),
                                frame='icrs', obstime='J2005.0',
                                ids=self.sources['id'], mag=mag)


flaky_rerun = pytest.mark.flaky(max_runs=10, min_passes=1,
                                rerun_filter=delay_rerun)
sirius_coords = ["Sirius", "06h45m09s -16d42m58s", [101.28715, -16.7161158],
                 np.array([101.28715, -16.7161158]), (101.28715, -16.7161158),
                 SkyCoord(101.28715, -16.7161158, unit=('degree', 'degree'))]
search_radius = ['0.1d', '360s', '6m', 0.1, Angle('0.1d')]


@flaky_rerun
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
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0], band='B')

        with pytest.raises(TypeError):
            DummySourcesCatalog()

        with pytest.raises(ValueError, match='Filter C not available.'):
            DummySourcesCatalog(sirius_coords[0], search_radius[0], band='C')

    @pytest.mark.parametrize('radius', search_radius)
    @pytest.mark.parametrize('center', sirius_coords)
    def test_catalog_center_radius(self, center, radius):
        c = DummySourcesCatalog(center, radius, band='B')

        assert_is_instance(c.center, SkyCoord)
        assert_almost_equal(c.center.ra.degree, 101.287155, decimal=3)
        assert_almost_equal(c.center.dec.degree, -16.7161158, decimal=3)

        assert_is_instance(c.radius, Angle)
        assert_almost_equal(c.radius.degree, 0.1)

        assert_is_instance(c.band, str)
        assert_equal(c.band, 'B')

    def test_catalog_properties(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0], band='B')
        assert_is_instance(c.sources_id, np.ndarray)
        assert_equal(c.sources_id.shape, (4))
        assert_is_instance(c.skycoord, SkyCoord)
        assert_is_instance(c.magnitude, QFloat)
        assert_is_instance(c.ra_dec_list, np.ndarray)
        assert_equal(c.ra_dec_list.shape, (4, 2))
        assert_is_instance(c.mag_list, np.ndarray)
        assert_equal(c.mag_list.shape, (4, 2))

        assert_equal(c.sources_id, c.sources['id'])
        assert_almost_equal(c.ra_dec_list[:, 0], c.sources['ra'])
        assert_almost_equal(c.ra_dec_list[:, 1], c.sources['dec'])
        assert_almost_equal(c.mag_list[:, 0], c.sources['mag'])
        assert_almost_equal(c.mag_list[:, 1], c.sources['mag_error'])

    def test_catalog_table(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0], band='B')
        t = c.table
        assert_is_instance(t, Table)
        assert_equal(t.colnames, ['id', 'ra', 'dec', 'mag', 'mag_error'])
        assert_equal(t['id'], c.sources['id'])
        assert_almost_equal(t['ra'], c.sources['ra'])
        assert_almost_equal(t['dec'], c.sources['dec'])
        assert_almost_equal(t['mag'], c.sources['mag'])
        assert_almost_equal(t['mag_error'], c.sources['mag_error'])

    def test_catalog_array(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0], band='B')
        t = c.array
        assert_is_instance(t, np.ndarray)
        assert_equal(t['id'], c.sources['id'])
        assert_almost_equal(t['ra'], c.sources['ra'])
        assert_almost_equal(t['dec'], c.sources['dec'])
        assert_almost_equal(t['mag'], c.sources['mag'])
        assert_almost_equal(t['mag_error'], c.sources['mag_error'])

    def test_catalog_getitem_number(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0], band='B')
        nc = c[0]
        assert_equal(nc.sources_id, c.sources['id'][0])
        assert_almost_equal(nc.ra_dec_list,
                            [c.sources['ra'][0], c.sources['dec'][0]])
        assert_almost_equal(nc.mag_list,
                            [c.sources['mag'][0], c.sources['mag_error'][0]])

    def test_catalog_getitem_array(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0], band='B')
        for items in [[2, 3], slice(2, None), np.array([2, 3])]:
            nc = c[items]
            assert_equal(len(nc), 2)
            assert_equal(nc.sources_id, c.sources['id'][2:])
            assert_almost_equal(nc.ra_dec_list[:, 0], c.sources['ra'][2:])
            assert_almost_equal(nc.ra_dec_list[:, 1], c.sources['dec'][2:])
            assert_almost_equal(nc.mag_list[:, 0], c.sources['mag'][2:])
            assert_almost_equal(nc.mag_list[:, 1], c.sources['mag_error'][2:])
            assert_is_none(nc._query)

        # Ensure error is raised when a list of strings
        with pytest.raises(KeyError):
            c['id', 'ra']

        with pytest.raises(KeyError):
            c[(2, 3)]

    def test_catalog_getitem_columns(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0], band='B')
        for i in c.sources.colnames:
            assert_equal(c[i], c.sources[i])

        with pytest.raises(KeyError):
            c['no column']

    def test_catalog_getitem_emptyquery(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0], band='B')
        nc = c[[1, 2]]
        with pytest.raises(KeyError, match='Empty query.'):
            nc['id']

    def test_catalog_len(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0], band='B')
        assert_equal(len(c), 4)
        with pytest.raises(TypeError):
            len(c[0])

    def test_catalog_match_objects(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0], band='B')
        m = c.match_objects([0.52525258, 0.87265989, 4.16526547],
                            [3.65404807, 5.50588171, 3.80703142],
                            limit_angle='1 arcsec')
        assert_is_instance(m, SourcesCatalog)
        assert_equal(m.sources_id, ['id2', '', 'id4'])
        assert_almost_equal(m.ra_dec_list[:, 0], [0.52522258, np.nan, 4.16520547])
        assert_almost_equal(m.ra_dec_list[:, 1], [3.65404807, np.nan, 3.80703142])
        assert_almost_equal(m.mag_list[:, 0], [3.00, np.nan, 4.81], decimal=2)
        assert_almost_equal(m.mag_list[:, 1], [0.02, np.nan, 0.03], decimal=2)

    def test_catalog_match_objects_table(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0], band='B')
        m = c.match_objects([0.52525258, 0.87265989, 4.16526547],
                            [3.65404807, 5.50588171, 3.80703142],
                            limit_angle='1 arcsec',
                            table=True)
        assert_is_instance(m, Table)
        expect = Table({'id': ['id2', '', 'id4'],
                        'ra': [0.52522258, np.nan, 4.16520547],
                        'dec': [3.65404807, np.nan, 3.80703142],
                        'mag': [3.00, np.nan, 4.81],
                        'mag_error': [0.02, np.nan, 0.03]})
        assert_equal(m['id'], expect['id'])
        for k in ['ra', 'dec', 'mag', 'mag_error']:
            assert_almost_equal(m[k], expect[k])

    def test_catalog_empty_mag(self):
        c = DummySourcesCatalog(sirius_coords[0], search_radius[0], band='B')
        c._mags = None
        assert_is_none(c.magnitude)
        assert_is_none(c.mag_list)
        assert_equal(c.table.colnames, ['id', 'ra', 'dec'])


@flaky_rerun
class Test_Simbad():
    def test_simbad_creation_errors(self):
        # Need arguments
        with pytest.raises(TypeError):
            SimbadSourcesCatalog()
        with pytest.raises(TypeError):
            SimbadSourcesCatalog('test')

        with pytest.raises(ValueError, match='Filter None not available.'):
            SimbadSourcesCatalog('Sirius', '0.05d', band='None')
        # Filter None should pass, no mag data
        SimbadSourcesCatalog('Sirius', '0.05d', None)

    @pytest.mark.parametrize('radius', search_radius)
    @pytest.mark.parametrize('center', sirius_coords)
    def test_csimbad_creation_params(self, center, radius):
        s = SimbadSourcesCatalog(center, radius)
        assert_equal(s.sources_id[0], '* alf CMa')
        assert_almost_equal(s.ra_dec_list[0], [101.287155, -16.7161158], decimal=5)
        assert_is_none(s.magnitude)

        assert_is_instance(s.center, SkyCoord)
        assert_almost_equal(s.center.ra.degree, 101.287155, decimal=3)
        assert_almost_equal(s.center.dec.degree, -16.7161158, decimal=3)

        assert_is_instance(s.radius, Angle)
        assert_almost_equal(s.radius.degree, 0.1)

    def test_simbad_creation_photometry(self):
        s = SimbadSourcesCatalog(sirius_coords[0],
                                 search_radius[0],
                                 band='V')
        assert_equal(s.sources_id[0], '* alf CMa')
        assert_almost_equal(s.ra_dec_list[0], [101.28715, -16.7161158], decimal=5)
        assert_almost_equal(s.mag_list[0][0], -1.46)

    def test_simbad_properties_types(self):
        s = SimbadSourcesCatalog(sirius_coords[0],
                                 search_radius[0],
                                 band='V')

        assert_is_instance(s.sources_id, np.ndarray)
        assert_equal(s.sources_id.shape, (len(s)))
        assert_is_instance(s.skycoord, SkyCoord)
        assert_is_instance(s.magnitude, QFloat)
        assert_is_instance(s.ra_dec_list, np.ndarray)
        assert_equal(s.ra_dec_list.shape, (len(s), 2))
        assert_is_instance(s.mag_list, np.ndarray)
        assert_equal(s.mag_list.shape, (len(s), 2))
        assert_is_instance(s.coordinates_bibcode, np.ndarray)
        assert_equal(s.coordinates_bibcode.shape, (len(s),))
        assert_is_instance(s.magnitudes_bibcode, np.ndarray)
        assert_equal(s.magnitudes_bibcode.shape, (len(s),))

    def test_simbad_properties_table(self):
        s = SimbadSourcesCatalog(sirius_coords[0],
                                 search_radius[0],
                                 band='V')
        t = s.table
        assert_is_instance(t, Table)
        assert_equal(len(t), len(s))
        assert_equal(t.colnames, ['id', 'ra', 'dec', 'mag', 'mag_error'])
        assert_equal(t['id'][0], '* alf CMa')
        assert_almost_equal(t['ra'][0], 101.287155)
        assert_almost_equal(t['dec'][0], -16.7161158)
        assert_almost_equal(t['mag'][0], -1.46, decimal=2)
        assert_true(np.isnan(t['mag_error'][0]))


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


class Test_Vizier:
    def test_vizier_need_initialization(self):
        with pytest.raises(NotImplementedError, match='Some required methods'):
            _VizierSourcesCatalog(sirius_coords[0],
                                  search_radius[0],
                                  band=None)


@flaky_rerun
class Test_Vizier_UCAC4:
    def test_ucac4_creation_errors(self):
        # Need arguments
        with pytest.raises(TypeError):
            UCAC4SourcesCatalog()
        with pytest.raises(TypeError):
            UCAC4SourcesCatalog('test')

        with pytest.raises(ValueError, match='Filter None not available.'):
            UCAC4SourcesCatalog('Sirius', '0.05d', band='None')
        # Filter None should pass, no mag data
        UCAC4SourcesCatalog('Sirius', '0.05d', None)

    @pytest.mark.parametrize('radius', search_radius)
    @pytest.mark.parametrize('center', sirius_coords)
    def test_ucac4_query_input_types(self, center, radius):
        c = UCAC4SourcesCatalog(center, radius, band='V')

        assert_equal(c.sources_id[0], 'UCAC4 367-016700')
        assert_almost_equal(c.ra_dec_list[0], [101.28715, -16.7161158], decimal=5)
        assert_almost_equal(c.mag_list[0], [-1.440, 0.0])

    def test_ucac4_properties_types(self):
        s = UCAC4SourcesCatalog(sirius_coords[0],
                                 search_radius[0],
                                 band='V')

        assert_is_instance(s.sources_id, np.ndarray)
        assert_equal(s.sources_id.shape, (len(s)))
        assert_is_instance(s.skycoord, SkyCoord)
        assert_is_instance(s.magnitude, QFloat)
        assert_is_instance(s.ra_dec_list, np.ndarray)
        assert_equal(s.ra_dec_list.shape, (len(s), 2))
        assert_is_instance(s.mag_list, np.ndarray)
        assert_equal(s.mag_list.shape, (len(s), 2))
