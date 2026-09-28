# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Deterministic coverage for SIMBAD's TAP response conversion."""

import numpy as np
from astropy.table import Table, MaskedColumn

from astropop.catalogs.simbad import _normalize_tap_result


def test_tap_flux_rows_preserve_source_order_filters_and_missing_values():
    query = Table({'main_id': ['z', 'z', 'a'],
                   'ra': [20., 20., 10.], 'dec': [-2., -2., -1.],
                   'pmra': [1., 1., 2.], 'pmdec': [3., 3., 4.],
                   'coo_bibcode': ['coords-z', 'coords-z', 'coords-a'],
                   'flux.filter': ['R', 'r', 'R'],
                   'flux': [4., 5., 6.],
                   'flux.bibcode': ['upper', 'lower', 'other']})
    query['flux_err'] = MaskedColumn([0.1, 0.2, 999.], mask=[False, False, True])
    original = query.copy()
    result = _normalize_tap_result(query, ['R', 'r', 'V'])
    assert list(result['MAIN_ID']) == ['z', 'a']
    np.testing.assert_allclose(result['RA'], [20., 10.])
    np.testing.assert_allclose(result['FLUX_R'], [4., 6.])
    np.testing.assert_allclose(result['FLUX_r'], [5., np.nan])
    np.testing.assert_allclose(result['FLUX_ERROR_R'], [0.1, np.nan])
    assert list(result['FLUX_BIBCODE_R']) == ['upper', 'other']
    assert list(result['FLUX_BIBCODE_r']) == ['lower', '']
    assert np.all(np.isnan(result['FLUX_V']))
    assert query.colnames == original.colnames
    assert len(query) == 3


def test_tap_without_photometry():
    query = Table({'main_id': ['star'], 'ra': [20.], 'dec': [-2.],
                   'pmra': [1.], 'pmdec': [3.], 'coo_bibcode': ['reference']})
    result = _normalize_tap_result(query, [])
    assert list(result['MAIN_ID']) == ['star']
    assert 'FLUX_V' not in result.colnames


def test_tap_catalog_sorts_sources_and_preserves_photometry(monkeypatch):
    import importlib
    from astropy import units as u
    module = importlib.import_module('astropop.catalogs.simbad')

    class Simbad:
        tap = None

        def add_votable_fields(self, *fields):
            assert 'pm' not in fields

        def query_region(self, center, radius):
            assert self.ROW_LIMIT == -1
            table = Table({'main_id': ['far', 'near'],
                           'ra': [21., 20.], 'dec': [-2., -2.],
                           'coo_bibcode': ['far-coords', 'near-coords'],
                           'flux.filter': ['V', 'V'], 'flux': [8., 4.],
                           'flux_err': [0.2, 0.1],
                           'flux.bibcode': ['far-flux', 'near-flux']})
            table['pmra'] = [1., 2.] * u.mas / u.yr
            table['pmdec'] = [3., 4.] * u.mas / u.yr
            return table

    monkeypatch.setattr(module, 'Simbad', Simbad)
    catalog = module.SimbadSourcesCatalog((20., -2.), '2 deg', band='V')
    assert list(catalog.sources_id()) == ['near', 'far']
    np.testing.assert_allclose(catalog.mag_list('V'), [[4., 0.1], [8., 0.2]])
    assert list(catalog.magnitudes_bibcode('V')) == ['near-flux', 'far-flux']
    assert list(catalog.coordinates_bibcode()) == ['near-coords', 'far-coords']


def test_identifier_prefix_does_not_match_double_star(monkeypatch):
    import importlib
    module = importlib.import_module('astropop.catalogs.simbad')

    class Simbad:
        def query_region(self, center, radius):
            return Table({'main_id': ['* target'], 'ra': [20.], 'dec': [-2.]})

        def query_objectids(self, name):
            return Table({'id': ['** unrelated', '* target', 'NAME Target']})

    monkeypatch.setattr(module, 'Simbad', Simbad)
    assert module.simbad_query_id(20., -2., '1s', name_order=['*']) == 'target'
    assert module.simbad_query_id(20., -2., '1s', name_order=['NAME']) == 'Target'
