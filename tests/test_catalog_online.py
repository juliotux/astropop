# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import time
import pytest
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropop.catalogs import SimbadCatalog
from astropop.catalogs._online_tools import _timeout_retry, \
                                            _wrap_query_table, \
                                            astroquery_radius, \
                                            astroquery_skycoord, \
                                            get_center_radius
from astropop.testing import assert_equal, assert_almost_equal, \
                             assert_is_instance


def delay_rerun(*args):
    time.sleep(10)
    return True


flaky_rerun = pytest.mark.flaky(max_runs=10, min_passes=1,
                                rerun_filter=delay_rerun)
catalog_skip = pytest.mark.skipif(not os.environ.get('ASTROPOP_TEST_CATALOGS'),
                                  reason='avoid servers errors.')


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

        def _return_table(str_a, str_b):
            t = Table()
            t['a'] = [str_a.encode('utf-8') for i in range(10)]
            t['b'] = [str_b for i in range(10)]
            t['c'] = [StrObj__(i) for i in range(10)]
            return t

        f = _wrap_query_table(_return_table)
        tab = f('A3#Â', str_b='B3#Ê')
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

    def test_astroquery_skycoord_string_skcord(self):
        value = "00h00m00s 00d00m00s"
        skcord = astroquery_skycoord(value)
        assert_is_instance(skcord, SkyCoord)
        assert_equal(skcord.ra.degree, 0.0)
        assert_equal(skcord.dec.degree, 0.0)

    def test_astroquery_skycoord_string_obj(self):
        value = "Sirius"
        skcord = astroquery_skycoord(value)
        assert_is_instance(skcord, SkyCoord)
        assert_almost_equal(skcord.ra.degree, 101.28715, decimal=4)
        assert_almost_equal(skcord.dec.degree, -16.7161158, decimal=4)

    def test_astroquery_skycoord_decimal_list(self):
        value = [101.28715, -16.7161158]
        skcord = astroquery_skycoord(value)
        assert_is_instance(skcord, SkyCoord)
        assert_almost_equal(skcord.ra.degree, 101.28715, decimal=4)
        assert_almost_equal(skcord.dec.degree, -16.7161158, decimal=4)

    def test_astroquery_skycoord_decimal_nparray(self):
        value = np.array([101.28715, -16.7161158])
        skcord = astroquery_skycoord(value)
        assert_is_instance(skcord, SkyCoord)
        assert_almost_equal(skcord.ra.degree, 101.28715, decimal=4)
        assert_almost_equal(skcord.dec.degree, -16.7161158, decimal=4)

    def test_astroquery_skycoord_decimal_tuple(self):
        value = (101.28715, -16.7161158)
        skcord = astroquery_skycoord(value)
        assert_is_instance(skcord, SkyCoord)
        assert_almost_equal(skcord.ra.degree, 101.28715, decimal=4)
        assert_almost_equal(skcord.dec.degree, -16.7161158, decimal=4)

    def test_astroquery_skycoord_decimal_skycoord(self):
        value = SkyCoord(101.28715, -16.7161158, unit=('degree', 'degree'))
        skcord = astroquery_skycoord(value)
        assert_is_instance(skcord, SkyCoord)
        assert_almost_equal(skcord.ra.degree, 101.28715, decimal=4)
        assert_almost_equal(skcord.dec.degree, -16.7161158, decimal=4)

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


# @pytest.mark.remote_data
# @flaky_rerun
# @catalog_skip
class TestSimbadCatalog():
    def test_simbad_catalog_query_object(self):
        cat = SimbadCatalog
        obj = cat.query_object('Sirius', band='V')
        assert_equal(obj[0]['MAIN_ID'], '* alf CMa')
        assert_equal(obj[0]['RA'], '06 45 08.9172')
        assert_equal(obj[0]['DEC'], '-16 42 58.017')
        assert_equal(obj[0]['COO_BIBCODE'], '2007A&A...474..653V')
        assert_almost_equal(obj[0]['FLUX_V'], -1.46)
        assert_equal(obj[0]['FLUX_SYSTEM_V'], 'Vega')
        assert_equal(obj[0]['FLUX_BIBCODE_V'], '2002yCat.2237....0D')

    def test_simbad_query_obj_no_band(self):
        cat = SimbadCatalog
        obj = cat.query_object('Sirius', band=None)
        assert_equal(obj[0]['MAIN_ID'], '* alf CMa')
        assert_equal(obj[0]['RA'], '06 45 08.9172')
        assert_equal(obj[0]['DEC'], '-16 42 58.017')
        assert_equal(obj[0]['COO_BIBCODE'], '2007A&A...474..653V')
        for i in obj.colnames:
            if 'FLUX_' in i:
                raise ValueError('Simbad is getting flux data when requested '
                                 'to not.')

    def test_simbad_catalog_query_region(self):
        cat = SimbadCatalog
        obj = cat.query_region('Sirius', radius='10 arcmin',
                               band='V')
        assert_equal(obj[0]['MAIN_ID'], '* alf CMa')
        assert_equal(obj[0]['RA'], '06 45 08.9172')
        assert_equal(obj[0]['DEC'], '-16 42 58.017')
        assert_equal(obj[0]['COO_BIBCODE'], '2007A&A...474..653V')
        assert_almost_equal(obj[0]['FLUX_V'], -1.46)
        assert_equal(obj[0]['FLUX_SYSTEM_V'], 'Vega')
        assert_equal(obj[0]['FLUX_BIBCODE_V'], '2002yCat.2237....0D')

    def test_simbad_catalog_query_region_no_band(self):
        cat = SimbadCatalog
        obj = cat.query_region('Sirius', radius='10 arcmin',
                               band=None)
        assert_equal(obj[0]['MAIN_ID'], '* alf CMa')
        assert_equal(obj[0]['RA'], '06 45 08.9172')
        assert_equal(obj[0]['DEC'], '-16 42 58.017')
        assert_equal(obj[0]['COO_BIBCODE'], '2007A&A...474..653V')
        for i in obj.colnames:
            if 'FLUX_' in i:
                raise ValueError("Simbad is getting flux data when requested"
                                 "to not.")
