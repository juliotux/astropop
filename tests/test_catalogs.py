# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropop.catalogs._sources_catalog import SourcesCatalog
from astropop.math import QFloat

from astropop.testing import *


sources = Table({'id': ['id1', 'id2', 'id3', 'id4'],
                 'ra': [2.44644404, 0.52522258, 0.64638169, 4.16520547],
                 'dec': [4.92305031, 3.65404807, 4.50588171, 3.80703142],
                 'pm_ra': [278.6, 114.3, 8.6, 270.1],
                 'pm_dec': [25.7, 202.6, 122.3, 256.3],
                 'mag': [2.16, 3.00, 3.55, 4.81],
                 'mag_error': [0.01, 0.02, 0.01, 0.03]})


class Test_SourcesCatalog_Conformance:
    def test_sourcescatalog_creation_error_id(self):
        with pytest.raises(ValueError, match='Sources ID must be a 1d array.'):
            SourcesCatalog(ra=sources['ra'], dec=sources['dec'], unit='degree',
                           ids=np.zeros((2, len(sources))))

        with pytest.raises(ValueError, match='Sources ID must be a 1d array.'):
            SourcesCatalog(ra=sources['ra'], dec=sources['dec'], unit='degree',
                           ids=None)

    def test_sourcescatalog_creation_skycoord_arguments(self):
        c = SourcesCatalog('1d 2d', ids=['Test'])
        assert_equal(c.sources_id[0], 'Test')
        assert_almost_equal(c.ra_dec_list, [(1, 2)])

        c = SourcesCatalog('1d', '2d', ids=['Test'])
        assert_equal(c.sources_id[0], 'Test')
        assert_almost_equal(c.ra_dec_list, [(1, 2)])

        c = SourcesCatalog('1d0m0s 2d0m0s', ids=['Test'])
        assert_equal(c.sources_id[0], 'Test')
        assert_almost_equal(c.ra_dec_list, [(1, 2)])

        c = SourcesCatalog('1d0m0s', '2d0m0s', ids=['Test'])
        assert_equal(c.sources_id[0], 'Test')
        assert_almost_equal(c.ra_dec_list, [(1, 2)])

        c = SourcesCatalog('1h 2d', ids=['Test'])
        assert_equal(c.sources_id[0], 'Test')
        assert_almost_equal(c.ra_dec_list, [(15, 2)])

        c = SourcesCatalog('1h', '2d', ids=['Test'])
        assert_equal(c.sources_id[0], 'Test')
        assert_almost_equal(c.ra_dec_list, [(15, 2)])

        c = SourcesCatalog('1h0m0s 2d0m0s', ids=['Test'])
        assert_equal(c.sources_id[0], 'Test')
        assert_almost_equal(c.ra_dec_list, [(15, 2)])

        c = SourcesCatalog('1h0m0s', '2d0m0s', ids=['Test'])
        assert_equal(c.sources_id[0], 'Test')
        assert_almost_equal(c.ra_dec_list, [(15, 2)])

        c = SourcesCatalog('1 00 00 +2 00 00', unit=(u.hour, u.degree),
                           ids=['Test'])
        assert_equal(c.sources_id[0], 'Test')
        assert_almost_equal(c.ra_dec_list, [(15, 2)])

        c = SourcesCatalog('1 00 00', '+2 00 00', unit=(u.hour, u.degree),
                           ids=['Test'])
        assert_equal(c.sources_id[0], 'Test')
        assert_almost_equal(c.ra_dec_list, [(15, 2)])

        c = SourcesCatalog('1 00 00 +2 00 00', unit=u.degree,
                           ids=['Test'])
        assert_equal(c.sources_id[0], 'Test')
        assert_almost_equal(c.ra_dec_list, [(1, 2)])

        c = SourcesCatalog('1 00 00', '+2 00 00', unit=u.degree,
                           ids=['Test'])
        assert_equal(c.sources_id[0], 'Test')
        assert_almost_equal(c.ra_dec_list, [(1, 2)])

        sk = SkyCoord('1 00 00', '+2 00 00', unit=(u.hour, u.degree))
        c = SourcesCatalog(sk, ids=['Test'])
        assert_equal(c.sources_id[0], 'Test')
        assert_almost_equal(c.ra_dec_list, [(15, 2)])

    def test_sourcescatalog_property_types(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           mag=sources['mag'],
                           mag_error=sources['mag_error'],
                           mag_unit='mag',
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'))

        assert_is_instance(s.sources_id, np.ndarray)
        assert_equal(s.sources_id.shape, (len(s)))
        assert_is_instance(s.skycoord, SkyCoord)
        assert_is_instance(s.magnitude, QFloat)
        assert_is_instance(s.ra_dec_list, np.ndarray)
        assert_equal(s.ra_dec_list.shape, (len(s), 2))
        assert_is_instance(s.mag_list, np.ndarray)
        assert_equal(s.mag_list.shape, (len(s), 2))

    def test_sourcescatalog_table(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           mag=sources['mag'],
                           mag_error=sources['mag_error'],
                           mag_unit='mag',
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'))

        assert_is_instance(s.table, Table)
        assert_equal(s.table.colnames, ['id', 'ra', 'dec', 'mag', 'mag_error'])

    def test_sourcescatalog_table_nomag(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'))

        assert_is_instance(s.table, Table)
        assert_equal(s.table.colnames, ['id', 'ra', 'dec'])

    def test_sourcescatalog_single_source(self):
        s = SourcesCatalog(ids=['Test'], ra='1d', dec='2d',
                           mag=QFloat(1, 0.1, 'mag'))
        assert_equal(s.sources_id, ['Test'])
        assert_equal(s.mag_list, [(1, 0.1)])
        assert_equal(s.ra_dec_list, [(1, 2)])

    def test_sourcescatalog_store_query(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'),
                           query_table=Table(sources))
        assert_equal(s['id'], sources['id'])
        assert_equal(s[2]['ra'], sources['ra'][2])

    def test_sourcescatalog_getitem(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'),
                           query_table=Table(sources))

        assert_equal(s['id'], sources['id'])
        assert_is_instance(s[2], SourcesCatalog)
        assert_equal(s[2]['ra'], sources['ra'][2])
        assert_equal(len(s[2]), 1)
        assert_equal(len(s[2]._query), 1)
        assert_equal(s[2].sources_id, [sources['id'][2]])
        assert_is_instance(s[2:4], SourcesCatalog)
        assert_equal(len(s[2:4]), 2)

    def test_sourcescatalog_getitem_invalid(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'))

        with pytest.raises(KeyError):
            s['id']  # no table must raise string

        with pytest.raises(KeyError):
            s[2, 3]  # tuple not accepted
