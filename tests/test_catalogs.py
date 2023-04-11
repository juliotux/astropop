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
                 'V': QFloat([2.1, 3.0, 3.55, 4.8], [0.01, 0.02, 0.01, 0.03]),
                 'B': QFloat([3.1, 3.0, 4.55, 4.8], [0.01, 0.02, 0.01, 0.03])})


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
        assert_equal(c.sources_id()[0], 'Test')
        assert_almost_equal(c.ra_dec_list(), [(1, 2)])
        assert_almost_equal(c.ra(), 1)
        assert_almost_equal(c.dec(), 2)

        c = SourcesCatalog('1d', '2d', ids=['Test'])
        assert_equal(c.sources_id()[0], 'Test')
        assert_almost_equal(c.ra_dec_list(), [(1, 2)])
        assert_almost_equal(c.ra(), 1)
        assert_almost_equal(c.dec(), 2)

        c = SourcesCatalog('1d0m0s 2d0m0s', ids=['Test'])
        assert_equal(c.sources_id()[0], 'Test')
        assert_almost_equal(c.ra_dec_list(), [(1, 2)])
        assert_almost_equal(c.ra(), 1)
        assert_almost_equal(c.dec(), 2)

        c = SourcesCatalog('1d0m0s', '2d0m0s', ids=['Test'])
        assert_equal(c.sources_id()[0], 'Test')
        assert_almost_equal(c.ra_dec_list(), [(1, 2)])
        assert_almost_equal(c.ra(), 1)
        assert_almost_equal(c.dec(), 2)

        c = SourcesCatalog('1h 2d', ids=['Test'])
        assert_equal(c.sources_id()[0], 'Test')
        assert_almost_equal(c.ra_dec_list(), [(15, 2)])
        assert_almost_equal(c.ra(), 15)
        assert_almost_equal(c.dec(), 2)

        c = SourcesCatalog('1h', '2d', ids=['Test'])
        assert_equal(c.sources_id()[0], 'Test')
        assert_almost_equal(c.ra_dec_list(), [(15, 2)])
        assert_almost_equal(c.ra(), 15)
        assert_almost_equal(c.dec(), 2)

        c = SourcesCatalog('1h0m0s 2d0m0s', ids=['Test'])
        assert_equal(c.sources_id()[0], 'Test')
        assert_almost_equal(c.ra_dec_list(), [(15, 2)])
        assert_almost_equal(c.ra(), 15)
        assert_almost_equal(c.dec(), 2)

        c = SourcesCatalog('1h0m0s', '2d0m0s', ids=['Test'])
        assert_equal(c.sources_id()[0], 'Test')
        assert_almost_equal(c.ra_dec_list(), [(15, 2)])
        assert_almost_equal(c.ra(), 15)
        assert_almost_equal(c.dec(), 2)

        c = SourcesCatalog('1 00 00 +2 00 00', unit=(u.hour, u.degree),
                           ids=['Test'])
        assert_equal(c.sources_id()[0], 'Test')
        assert_almost_equal(c.ra_dec_list(), [(15, 2)])
        assert_almost_equal(c.ra(), 15)
        assert_almost_equal(c.dec(), 2)

        c = SourcesCatalog('1 00 00', '+2 00 00', unit=(u.hour, u.degree),
                           ids=['Test'])
        assert_equal(c.sources_id()[0], 'Test')
        assert_almost_equal(c.ra_dec_list(), [(15, 2)])
        assert_almost_equal(c.ra(), 15)
        assert_almost_equal(c.dec(), 2)

        c = SourcesCatalog('1 00 00 +2 00 00', unit=u.degree,
                           ids=['Test'])
        assert_equal(c.sources_id()[0], 'Test')
        assert_almost_equal(c.ra_dec_list(), [(1, 2)])
        assert_almost_equal(c.ra(), 1)
        assert_almost_equal(c.dec(), 2)

        c = SourcesCatalog('1 00 00', '+2 00 00', unit=u.degree,
                           ids=['Test'])
        assert_equal(c.sources_id()[0], 'Test')
        assert_almost_equal(c.ra_dec_list(), [(1, 2)])
        assert_almost_equal(c.ra(), 1)
        assert_almost_equal(c.dec(), 2)

        sk = SkyCoord('1 00 00', '+2 00 00', unit=(u.hour, u.degree))
        c = SourcesCatalog(sk, ids=['Test'])
        assert_equal(c.sources_id()[0], 'Test')
        assert_almost_equal(c.ra_dec_list(), [(15, 2)])
        assert_almost_equal(c.ra(), 15)
        assert_almost_equal(c.dec(), 2)

    def test_sourcescatalog_property_types(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           mag={'V': sources['V'], 'B': sources['B']},
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'))

        assert_is_instance(s.sources_id(), np.ndarray)
        assert_equal(s.sources_id().shape, (len(s)))
        assert_is_instance(s.skycoord(), SkyCoord)
        assert_is_instance(s.magnitude('V'), QFloat)
        assert_is_instance(s.ra_dec_list(), np.ndarray)
        assert_equal(s.ra_dec_list().shape, (len(s), 2))
        assert_is_instance(s.mag_list('V'), np.ndarray)
        assert_equal(s.mag_list('V').shape, (len(s), 2))
        assert_is_instance(s.ra(), np.ndarray)
        assert_equal(s.ra().shape, (len(s),))
        assert_is_instance(s.dec(), np.ndarray)
        assert_equal(s.dec().shape, (len(s),))

    def test_sourcescatalog_table(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           mag={'V': sources['V'], 'B': sources['B']},
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'))

        tab = s.table()
        assert_is_instance(tab, Table)
        assert_equal(tab.colnames, ['id', 'ra', 'dec', 'pm_ra_cosdec',
                                    'pm_dec', 'V', 'V_error', 'B', 'B_error'])
        assert_equal(tab['ra'].unit, u.Unit('deg'))
        assert_equal(tab['dec'].unit, u.Unit('deg'))
        assert_equal(tab['pm_ra_cosdec'].unit, u.Unit('mas/yr'))
        assert_equal(tab['pm_dec'].unit, u.Unit('mas/yr'))

    def test_sourcescatalog_table_nopm(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           mag={'V': sources['V'], 'B': sources['B']})

        tab = s.table()
        assert_is_instance(tab, Table)
        assert_equal(tab.colnames, ['id', 'ra', 'dec', 'V', 'V_error',
                                    'B', 'B_error'])
        assert_equal(tab['ra'].unit, u.Unit('deg'))
        assert_equal(tab['dec'].unit, u.Unit('deg'))

    def test_sourcescatalog_table_nomag(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree)

        tab = s.table()
        assert_is_instance(tab, Table)
        assert_equal(tab.colnames, ['id', 'ra', 'dec'])
        assert_equal(tab['ra'].unit, u.Unit('deg'))
        assert_equal(tab['dec'].unit, u.Unit('deg'))

    def test_sourcescatalog_single_source(self):
        s = SourcesCatalog(ids=['Test'], ra='1d', dec='2d',
                           mag={'V': QFloat([1], [0.1], 'mag')})
        assert_equal(s.sources_id(), ['Test'])
        assert_equal(s.mag_list('V'), [(1, 0.1)])
        assert_equal(s.ra_dec_list(), [(1, 2)])

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

    def test_sourcescatalog_get_query(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'),
                           query_table=Table(sources))

        assert_is_instance(s.query_table, Table)
        assert_equal(s.query_table.colnames, sources.colnames)
        assert_equal(s.query_table['id'], sources['id'])

    def test_sourcescatalog_get_query_colnames(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'),
                           query_table=Table(sources))

        assert_equal(s.query_colnames, sources.colnames)

    def test_sourcescatalog_getitem(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'),
                           mag={'V': sources['V'], 'B': sources['B']},
                           query_table=Table(sources))

        assert_equal(s['id'], sources['id'])
        assert_is_instance(s[2]._base_table, Table)
        assert_is_instance(s[2]._mags_table, Table)
        assert_is_instance(s[2]._query, Table)
        assert_is_instance(s[2], SourcesCatalog)
        assert_equal(s[2]['ra'], sources['ra'][2])
        assert_equal(len(s[2]), 1)
        assert_equal(len(s[2]._query), 1)
        assert_equal(s[2].sources_id(), [sources['id'][2]])
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

    def test_sourcescatalog_filters(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'),
                           mag={'V': sources['V'], 'B': sources['B']},
                           query_table=Table(sources))

        assert_equal(s.filters, ['V', 'B'])

    def test_sourcescatalog_match_object_multiple_objects(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'))

        ra = sources['ra']
        ra[2] = 8.5
        dec = sources['dec']
        dec[2] = 6.2

        ncat = s.match_objects(ra, dec, '1 arcsec').table()
        expect = s.table()
        expect[2] = [''] + [np.nan]*(len(expect.colnames)-1)
        assert_equal(ncat['id'], expect['id'])
        for i in ncat.colnames[1:]:
            assert_almost_equal(ncat[i], expect[i])

    def test_sourcescatalog_match_object_one(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'))

        ra = sources['ra'][2]
        dec = sources['dec'][2]

        ncat = s.match_objects(ra, dec, '1 arcsec').table()
        expect = s.table()[2]
        assert_equal(len(ncat), 1)
        assert_equal(ncat['id'], expect['id'])
        for i in ncat.colnames[1:]:
            assert_almost_equal(ncat[i], expect[i])

    def test_sourcescatalog_match_object_no_pm(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree)

        ra = sources['ra'][2]
        dec = sources['dec'][2]

        ncat = s.match_objects(ra, dec, '1 arcsec').table()
        expect = s.table()[2]
        assert_equal(len(ncat), 1)
        assert_equal(ncat['id'], expect['id'])
        for i in ncat.colnames[1:]:
            assert_almost_equal(ncat[i], expect[i])

    def test_sourcescatalog_ensure_mag_dict(self):
        with pytest.raises(TypeError, match='mag must be a dict'):
            s = SourcesCatalog(ids=sources['id'],
                               ra=sources['ra'],
                               dec=sources['dec'],
                               unit=u.degree,
                               mag=sources['V'])

    def test_sourcescatalog_ensure_mag_lenght(self):
        with pytest.raises(ValueError, match='Lengths of magnitudes must be'):
            s = SourcesCatalog(ids=sources['id'],
                               ra=sources['ra'],
                               dec=sources['dec'],
                               unit=u.degree,
                               mag={'V': sources['V'][:-2]})

    def test_sourcescatalog_single_object(self):
        s = SourcesCatalog(ids=[sources['id'][2]],
                           ra=sources['ra'][2],
                           dec=sources['dec'][2],
                           unit=u.degree,
                           pm_ra_cosdec=sources['pm_ra'][2]*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec'][2]*u.Unit('mas/yr'))

        assert_equal(s.sources_id(), [sources['id'][2]])
        assert_almost_equal(s.ra_dec_list(),
                            [(sources['ra'][2], sources['dec'][2])])

    def test_sourcescatalog_copy(self):
        s = SourcesCatalog(ids=sources['id'],
                           ra=sources['ra'],
                           dec=sources['dec'],
                           unit=u.degree,
                           pm_ra_cosdec=sources['pm_ra']*u.Unit('mas/yr'),
                           pm_dec=sources['pm_dec']*u.Unit('mas/yr'))

        s2 = s.copy()
        assert_equal(s2.sources_id(), s.sources_id())
        assert_almost_equal(s2.ra_dec_list(), s.ra_dec_list())
