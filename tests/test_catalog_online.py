# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropop.catalogs.online import SimbadCatalog
from astropop.testing import assert_equal, assert_almost_equal


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
