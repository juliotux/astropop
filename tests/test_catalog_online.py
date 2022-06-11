# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import time
import pytest
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropop.catalogs.simbad import SimbadCatalog, simbad_query_id
from astropop.catalogs._online_tools import _timeout_retry, \
                                            _wrap_query_table, \
                                            astroquery_radius, \
                                            astroquery_skycoord, \
                                            get_center_radius
from astropop.testing import *
from astroquery.simbad import Simbad


def delay_rerun(*args):
    time.sleep(10)
    return True


flaky_rerun = pytest.mark.flaky(max_runs=10, min_passes=1,
                                rerun_filter=delay_rerun)
catalog_skip = pytest.mark.skipif(not os.environ.get('ASTROPOP_TEST_CATALOGS'),
                                  reason='avoid servers errors.')

sirius_coords = ["Sirius", "06h45m09s -16d42m58s", [101.28715, -16.7161158],
                 np.array([101.28715, -16.7161158]), (101.28715, -16.7161158),
                 SkyCoord(101.28715, -16.7161158, unit=('degree', 'degree'))]


@flaky_rerun
@catalog_skip
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

        _wrap_query_table(tab)

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

    def test_astroquery_radius(self):
        ang = Angle(1.0, unit='degree')
        stra = '1d'
        strb = '3600 arcsec'
        inta = 1

        assert_equal(astroquery_radius(ang), "1.0d")
        assert_equal(astroquery_radius(stra), "1.0d")
        assert_equal(astroquery_radius(strb), "1.0d")
        assert_equal(astroquery_radius(inta), "1.0d")

    def test_astroquery_radius_error(self):
        not_angle = 'not angle'
        with pytest.raises(ValueError):
            astroquery_radius(not_angle)


@flaky_rerun
@catalog_skip
class TestSimbadCatalog():
    @property
    def cat(self):
        return SimbadCatalog.copy()

    def test_simbad_catalog_get_simbad_copy(self):
        # always return a copy of the querier
        s = Simbad()

        # assign our simbad
        self.cat.simbad = s
        assert_is_not(self.cat.simbad, s)
        assert_equal(self.cat.simbad.ROW_LIMIT, 0)

    def test_simbad_catalog_set_simbad(self):
        s = Simbad()

        # ok for simbad
        self.cat.simbad = s

        # raise everything else
        for i in ['Simbad', 'simbad', 1.0, [], (), np.array([]), {}]:
            with pytest.raises(ValueError, match='is not a SimbadClass'):
                self.cat.simbad = i

    @pytest.mark.parametrize('band', ["U", "B", "V", "R", "I", "J", "H", "K",
                                      "u", "g", "r", "i", "z"])
    def test_simbad_catalog_check_filter(self, band):
        assert_true(self.cat.check_filter, band)

    def test_simbad_catalog_check_filter_errors(self):
        # default behavior is raise error
        with pytest.raises(ValueError, match='This catalog does not support'):
            self.cat.check_filter('inexisting')

        with pytest.raises(ValueError, match='This catalog does not support'):
            self.cat.check_filter('inexisting', raise_error=True)

        assert_false(self.cat.check_filter('inexisting', raise_error=False))

    def test_simbad_catalog_get_simbad(self):
        # must be free of fluxdata
        s = self.cat.get_simbad()
        assert_is_instance(s, Simbad.__class__)
        for i in s.get_votable_fields():
            assert_not_in('fluxdata', i)

        # must contain flux data
        s = self.cat.get_simbad(band='V')
        assert_is_instance(s, Simbad.__class__)
        assert_in('fluxdata(V)', s.get_votable_fields())

    def test_simbad_catalog_query_object(self):
        obj = self.cat.query_object('Sirius', band='V')
        assert_equal(obj[0]['MAIN_ID'], '* alf CMa')
        assert_equal(obj[0]['RA'], '06 45 08.9172')
        assert_equal(obj[0]['DEC'], '-16 42 58.017')
        assert_equal(obj[0]['COO_BIBCODE'], '2007A&A...474..653V')
        assert_almost_equal(obj[0]['FLUX_V'], -1.46)
        assert_equal(obj[0]['FLUX_SYSTEM_V'], 'Vega')
        assert_equal(obj[0]['FLUX_BIBCODE_V'], '2002yCat.2237....0D')

    def test_simbad_query_obj_no_band(self):
        obj = self.cat.query_object('Sirius', band=None)
        assert_equal(obj[0]['MAIN_ID'], '* alf CMa')
        assert_equal(obj[0]['RA'], '06 45 08.9172')
        assert_equal(obj[0]['DEC'], '-16 42 58.017')
        assert_equal(obj[0]['COO_BIBCODE'], '2007A&A...474..653V')
        for i in obj.colnames:
            if 'FLUX_' in i:
                raise ValueError('Simbad is getting flux data when requested '
                                 'to not.')

    @pytest.mark.parametrize('value', sirius_coords)
    def test_simbad_catalog_query_region(self, value):
        obj = self.cat.query_region(value, radius='10 arcmin',
                                    band='V')
        assert_equal(obj[0]['MAIN_ID'], '* alf CMa')
        assert_equal(obj[0]['RA'], '06 45 08.9172')
        assert_equal(obj[0]['DEC'], '-16 42 58.017')
        assert_equal(obj[0]['COO_BIBCODE'], '2007A&A...474..653V')
        assert_almost_equal(obj[0]['FLUX_V'], -1.46)
        assert_equal(obj[0]['FLUX_SYSTEM_V'], 'Vega')
        assert_equal(obj[0]['FLUX_BIBCODE_V'], '2002yCat.2237....0D')

    @pytest.mark.parametrize('value', sirius_coords)
    def test_simbad_catalog_query_region_no_band(self, value):
        obj = self.cat.query_region(value, radius='10 arcmin',
                                    band=None)
        assert_equal(obj[0]['MAIN_ID'], '* alf CMa')
        assert_equal(obj[0]['RA'], '06 45 08.9172')
        assert_equal(obj[0]['DEC'], '-16 42 58.017')
        assert_equal(obj[0]['COO_BIBCODE'], '2007A&A...474..653V')
        for i in obj.colnames:
            if 'FLUX_' in i:
                raise ValueError("Simbad is getting flux data when requested"
                                 "to not.")

    @pytest.mark.parametrize('value', sirius_coords[2:])
    def test_simbad_catalog_query_object_error(self, value):
        # Simbad query do not accept coordinates or SkyCoords
        with pytest.raises(ValueError, match="only accept object name"):
            self.cat.query_object(value)
        # but query region must pass
        self.cat.query_region(value, radius='1m')

    @pytest.mark.parametrize('radius', [Angle(0.01, unit='degree'), '0.01d',
                                        0.01, '1m', '10s', '10 arcsec'])
    def test_simbad_catalog_query_different_radius(self, radius):
        # query with different radius types
        self.cat.query_region('Sirius', radius)

    def test_simbad_catalog_filter_flux(self):
        query = Table()
        query['FLUX_V'] = np.arange(10)
        query['FLUX_ERROR_V'] = np.array([0.1]*10)
        query['FLUX_UNIT_V'] = np.array(['mag']*10)
        query['FLUX_BIBCODE_V'] = np.array(['gh2022&astropop']*10)

        flux = self.cat.filter_flux(band='V', query=query)
        assert_equal(len(flux), 4)
        assert_equal(flux[0], query['FLUX_V'])
        assert_equal(flux[1], query['FLUX_ERROR_V'])
        assert_equal(flux[2], query['FLUX_UNIT_V'])
        assert_equal(flux[3], query['FLUX_BIBCODE_V'])

    def test_simbad_catalog_filter_flux_error(self):
        query = Table()
        with pytest.raises(KeyError,
                           match='Simbad query must be performed with band'):
            self.cat.filter_flux(band='V', query=query)

    def test_simbad_catalog_filter_flux_query_none(self):
        cat = self.cat
        query = cat.query_region('Sirius', '5m', band='V')
        flux = cat.filter_flux('V')

        assert_equal(flux[0], query['FLUX_V'])
        assert_equal(flux[1], query['FLUX_ERROR_V'])
        assert_equal(flux[2], query['FLUX_UNIT_V'])
        assert_equal(flux[3], query['FLUX_BIBCODE_V'])

    def test_simbad_catalog_filter_id(self):
        # performe filter in a different class
        query = self.cat.query_region('Sirius', '5m')
        id = self.cat.filter_id(query)
        id_resolved = self.cat._id_resolve(query['MAIN_ID'])

        assert_equal(id, id_resolved)

    def test_simbad_catalog_filter_id_query_none(self):
        cat = self.cat
        query = cat.query_region('Sirius', '5m')
        idn = cat.filter_id()
        id_resolved = self.cat._id_resolve(query['MAIN_ID'])
        assert_equal(idn, id_resolved)

    def test_simbad_catalog_match_object_ids(self):
        ra = [101.28715, 88.79293875, 191.93028625]
        dec = [-16.7161158, 7.40706389, -59.68877194]
        name = ['alf CMa', 'alf Ori', 'bet Cru']
        res = self.cat.match_object_ids(ra, dec)
        assert_equal(res, name)

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
