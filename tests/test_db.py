# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
from astropop._db import Database
import numpy as np
from astropy.table import Table
from astropop.testing import *


class Test_Database_Creation:
    def test_database_simple_creation(self):
        db = Database()
        assert_equal(len(db), 0)
        assert_equal(db._db, ':memory:')
        assert_equal(db._table, 'main')

    def test_database_simple_creation_file(self, tmp_path):
        db = Database(str(tmp_path / 'test.db'))
        assert_equal(db._table, 'main')
        assert_equal(db._db, str(tmp_path / 'test.db'))
        assert_equal(len(db), 0)

    def test_database_simple_creation_table(self):
        db = Database(table='test')
        assert_equal(db._table, 'test')
        assert_equal(db._db, ':memory:')
        assert_equal(len(db), 0)

    def test_database_simple_creation_file_table(self, tmp_path):
        db = Database(str(tmp_path / 'test.db'), table='test')
        assert_equal(db._table, 'test')
        assert_equal(db._db, str(tmp_path / 'test.db'))
        assert_equal(len(db), 0)

    def test_database_simple_creation_dtype(self):
        dtype = np.dtype([('a', 'i8'), ('b', 'f8'), ('c', 'U10'), ('d', 'U10')])
        db = Database(dtype=dtype)
        assert_equal(db.colnames(), ['__id__', 'a', 'b', 'c', 'd'])

    def test_database_simple_creation_data_nparray(self):
        data = np.array([(1, 2.0, 'a', 'b'), (3, 4.0, 'c', 'd')],
                        dtype=[('a', 'i8'), ('b', 'f8'), ('c', 'S10'), ('d', 'U10')])
        db = Database(data=data)
        assert_equal(db.colnames(), ['__id__', 'a', 'b', 'c', 'd'])
        assert_equal(len(db), 2)
        assert_equal(db['a'], [1, 3])
        assert_almost_equal(db['b'], [2.0, 4.0])
        assert_equal(db['c'], ['a', 'c'])
        assert_equal(db['d'], ['b', 'd'])

    def test_database_simple_creation_data_table(self):
        data = Table(rows=[[1, 2.0, 'a', 'b'], [3, 4.0, 'c', 'd']],
                     names=['a', 'b', 'c', 'd'])
        db = Database(data=data)
        assert_equal(db.colnames(), ['__id__', 'a', 'b', 'c', 'd'])
        assert_equal(len(db), 2)
        assert_equal(db['a'], [1, 3])
        assert_almost_equal(db['b'], [2.0, 4.0])
        assert_equal(db['c'], ['a', 'c'])
        assert_equal(db['d'], ['b', 'd'])

    def test_database_simple_creation_data_dict(self):
        data = {'a': [1, 3], 'b': [2.0, 4.0], 'c': ['a', 'c'], 'd': ['b', 'd']}
        db = Database(data=data)
        assert_equal(db.colnames(), ['__id__', 'a', 'b', 'c', 'd'])
        assert_equal(len(db), 2)
        assert_equal(db['a'], [1, 3])
        assert_almost_equal(db['b'], [2.0, 4.0])
        assert_equal(db['c'], ['a', 'c'])
        assert_equal(db['d'], ['b', 'd'])

        data = {'a': 1, 'b': 2.0, 'c': 'a', 'd': 'b'}
        db = Database(data=data)
        assert_equal(db.colnames(), ['__id__', 'a', 'b', 'c', 'd'])
        assert_equal(len(db), 1)
        assert_equal(db['a'], 1)
        assert_almost_equal(db['b'], 2.0)
        assert_equal(db['c'], 'a')
        assert_equal(db['d'], 'b')

    def test_database_simple_creation_data_invalid(self):
        with pytest.raises(TypeError):
            db = Database(data=1)

        with pytest.raises(TypeError):
            db = Database(data=['a', 'b', 'c'])

        with pytest.raises(TypeError):
            db = Database(data='abcdefg')

    def test_database_simple_creation_length(self):
        db = Database(length=10)
        assert_equal(len(db), 10)
        assert_equal(db.colnames(), ['__id__'])
        assert_equal(db['__id__'], np.arange(1, 11))

    def test_database_simple_creation_length_data(self):
        data = {'a': [1, 3], 'b': [2.0, 4.0],
                'c': ['a', 'c'], 'd': ['b', 'd']}
        with pytest.raises(ValueError,
                           match='data and length cannot be both set.'):
            db = Database(length=10, data=data)


class Test_Database_Modify:
    def test_database_add_column(self):
        db = Database()
        db.add_column('a')
        assert_equal(db.colnames(), ['__id__', 'a'])
        assert_equal(len(db), 0)

    def test_database_add_column_dtype(self):
        db = Database()
        db.add_column('a', dtype=np.dtype('i8'))
        assert_equal(db.colnames(), ['__id__', 'a'])
        assert_equal(len(db), 0)

    def test_database_add_column_dtype_invalid(self):
        db = Database()
        with pytest.raises(KeyError):
            db.add_column('a', dtype='M8')

    def test_database_add_column_data(self):
        db = Database()
        db.add_column('a', data=[1, 2, 3])
        assert_equal(db.colnames(), ['__id__', 'a'])
        assert_equal(len(db), 3)
        assert_equal(db['a'], [1, 2, 3])

    def test_database_add_column_data_invalid(self):
        db = Database()
        db.add_column('a', data=[1, 2, 3])
        assert_equal(db.colnames(), ['__id__', 'a'])
        assert_equal(len(db), 3)
        assert_equal(db['a'], [1, 2, 3])

        with pytest.raises(ValueError,
                           match="data must have the same length"):
            db.add_column('b', data=[1, 2, 3, 4])

    def test_database_set_column(self):
        db = Database()
        db.add_column('a', data=np.zeros(10))
        db.set_column('a', np.arange(10, 20))
        assert_equal(db.colnames(), ['__id__', 'a'])
        assert_equal(len(db), 10)
        assert_equal(db['a'], np.arange(10, 20))

    def test_database_set_column_invalid(self):
        db = Database()
        db.add_column('a', data=np.zeros(10))
        with pytest.raises(ValueError,
                           match="data must have the same length"):
            db.set_column('a', np.arange(10, 20, 2))
