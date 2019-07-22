# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..online import *
import pytest
import numpy.testing as npt


def test_simbad_catalog_query_object():
    cat = SimbadCatalog
    obj = cat.query_object('Sirius', band='V')
    assert obj[0]['MAIN_ID'] == b'* alf CMa'
    assert obj[0]['RA'] == '06 45 08.9172'
    assert obj[0]['DEC'] == '-16 42 58.017'
    assert obj[0]['COO_BIBCODE'] == b'2007A&A...474..653V'
    npt.assert_approx_equal(obj[0]['FLUX_V'], -1.46)
    assert obj[0]['FLUX_SYSTEM_V'] == b'Vega'
    assert obj[0]['FLUX_BIBCODE_V'] == b'2002yCat.2237....0D'


def test_simbad_query_obj_no_band():
    cat = SimbadCatalog
    obj = cat.query_object('Sirius', band=None)
    assert obj[0]['MAIN_ID'] == b'* alf CMa'
    assert obj[0]['RA'] == '06 45 08.9172'
    assert obj[0]['DEC'] == '-16 42 58.017'
    assert obj[0]['COO_BIBCODE'] == b'2007A&A...474..653V'
    for i in obj.colnames:
        if 'FLUX_' in i:
            raise ValueError('Simbad is getting flux data when requested to'
                             ' not.')


def test_simbad_catalog_query_region():
    cat = SimbadCatalog
    obj = cat.query_region('Sirius', radius='10 arcmin',
                           band='V')
    print(len(obj))
    assert obj[0]['MAIN_ID'] == b'* alf CMa'
    assert obj[0]['RA'] == '06 45 08.9172'
    assert obj[0]['DEC'] == '-16 42 58.017'
    assert obj[0]['COO_BIBCODE'] == b'2007A&A...474..653V'
    npt.assert_approx_equal(obj[0]['FLUX_V'], -1.46)
    assert obj[0]['FLUX_SYSTEM_V'] == b'Vega'
    assert obj[0]['FLUX_BIBCODE_V'] == b'2002yCat.2237....0D'
    assert len(obj) == 61


def test_simbad_catalog_query_region_no_band():
    cat = SimbadCatalog
    obj = cat.query_region('Sirius', radius='10 arcmin',
                           band=None)
    print(len(obj))
    assert obj[0]['MAIN_ID'] == b'* alf CMa'
    assert obj[0]['RA'] == '06 45 08.9172'
    assert obj[0]['DEC'] == '-16 42 58.017'
    assert obj[0]['COO_BIBCODE'] == b'2007A&A...474..653V'
    for i in obj.colnames:
        if 'FLUX_' in i:
            raise ValueError('Simbad is getting flux data when requested to'
                             ' not.')
    assert len(obj) == 61
