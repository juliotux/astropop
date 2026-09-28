# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test catalog column aliases without querying VizieR."""

import numpy as np
import pytest
import yaml
from astropy.table import Table
from astropy import units as u
from astropop.catalogs import vizier


@pytest.mark.parametrize('legacy_names', [False, True])
def test_column_aliases_preserve_filter_api(tmp_path, monkeypatch, legacy_names):
    config = {'table': 'example', 'columns': ['+_r', '**'],
              'available_filters': {'g_': 'Sloan g'},
              'column_aliases': {"g'mag": 'g_mag', "e_g'mag": 'e_g_mag',
                                 '2MASS': '_2MASS'},
              'coordinates': {'ra_column': 'RAJ2000', 'dec_column': 'DEJ2000'},
              'magnitudes': {'mag_column': '{band}mag',
                             'err_mag_column': 'e_{band}mag'},
              'ids': {'column': '_2MASS'}}
    path = tmp_path / 'catalog.yml'
    path.write_text(yaml.safe_dump(config))

    class Vizier:
        def __init__(self, catalog, columns):
            assert "g'mag" in columns
            assert 'g_mag' not in columns

        def query_region(self, center, radius):
            table = Table({'2MASS': ['source']})
            table['RAJ2000'] = [20.] * u.deg
            table['DEJ2000'] = [-2.] * u.deg
            table["g'mag"] = [12.] * u.mag
            table["e_g'mag"] = [0.1] * u.mag
            if legacy_names:
                for raw, alias in config['column_aliases'].items():
                    table.rename_column(raw, alias)
            return [table]

    monkeypatch.setattr(vizier, 'Vizier', Vizier)
    result = vizier.VizierSourcesCatalog(path, (20., -2.), '1s')
    assert result.filters == ['g_']
    assert list(result.sources_id()) == ['source']
    np.testing.assert_allclose(result.mag_list('g_'), [[12., 0.1]])
