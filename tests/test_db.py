# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
from astropop._db import Database, _ID_KEY, _sanitize_colnames
import numpy as np
from astropy.table import Table
from astropop.testing import *
import sqlite3


def test_sanitize_string():
    assert_equal(_sanitize_colnames('test'), 'test')
    assert_equal(_sanitize_colnames('test_'), 'test_')
    assert_equal(_sanitize_colnames('test_1'), 'test_1')
    assert_equal(_sanitize_colnames('test-2'), 'test_2')
    assert_equal(_sanitize_colnames('test!2'), 'test_2')
    assert_equal(_sanitize_colnames('test@2'), 'test_2')
    assert_equal(_sanitize_colnames('test#2'), 'test_2')
    assert_equal(_sanitize_colnames('test$2'), 'test_2')
    assert_equal(_sanitize_colnames('test&2'), 'test_2')
    assert_equal(_sanitize_colnames('test*2'), 'test_2')
    assert_equal(_sanitize_colnames('test(2)'), 'test_2_')
    assert_equal(_sanitize_colnames('test)2'), 'test_2')
    assert_equal(_sanitize_colnames('test[2]'), 'test_2_')
    assert_equal(_sanitize_colnames('test]2'), 'test_2')
    assert_equal(_sanitize_colnames('test{2}'), 'test_2_')
    assert_equal(_sanitize_colnames('test}2'), 'test_2')
    assert_equal(_sanitize_colnames('test|2'), 'test_2')
    assert_equal(_sanitize_colnames('test\\2'), 'test_2')
    assert_equal(_sanitize_colnames('test^2'), 'test_2')
    assert_equal(_sanitize_colnames('test~2'), 'test_2')
    assert_equal(_sanitize_colnames('test"2'), 'test_2')
    assert_equal(_sanitize_colnames('test\'2'), 'test_2')
    assert_equal(_sanitize_colnames('test`2'), 'test_2')
    assert_equal(_sanitize_colnames('test<2'), 'test_2')
    assert_equal(_sanitize_colnames('test>2'), 'test_2')
    assert_equal(_sanitize_colnames('test=2'), 'test_2')
    assert_equal(_sanitize_colnames('test,2'), 'test_2')
    assert_equal(_sanitize_colnames('test;2'), 'test_2')
    assert_equal(_sanitize_colnames('test:2'), 'test_2')
    assert_equal(_sanitize_colnames('test?2'), 'test_2')
    assert_equal(_sanitize_colnames('test/2'), 'test_2')

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
        assert_equal(db.colnames(), ['a', 'b', 'c', 'd'])

    def test_database_simple_creation_data_nparray(self):
        data = np.array([(1, 2.0, 'a', 'b'), (3, 4.0, 'c', 'd')],
                        dtype=[('a', 'i8'), ('b', 'f8'), ('c', 'S10'), ('d', 'U10')])
        db = Database(data=data)
        assert_equal(db.colnames(), ['a', 'b', 'c', 'd'])
        assert_equal(len(db), 2)
        assert_equal(db['a'], [1, 3])
        assert_almost_equal(db['b'], [2.0, 4.0])
        assert_equal(db['c'], ['a', 'c'])
        assert_equal(db['d'], ['b', 'd'])

    def test_database_simple_creation_data_table(self):
        data = Table(rows=[[1, 2.0, 'a', 'b'], [3, 4.0, 'c', 'd']],
                     names=['a', 'b', 'c', 'd'])
        db = Database(data=data)
        assert_equal(db.colnames(), ['a', 'b', 'c', 'd'])
        assert_equal(len(db), 2)
        assert_equal(db['a'], [1, 3])
        assert_almost_equal(db['b'], [2.0, 4.0])
        assert_equal(db['c'], ['a', 'c'])
        assert_equal(db['d'], ['b', 'd'])

    def test_database_simple_creation_data_dict(self):
        data = {'a': [1, 3], 'b': [2.0, 4.0], 'c': ['a', 'c'], 'd': ['b', 'd']}
        db = Database(data=data)
        assert_equal(db.colnames(), ['a', 'b', 'c', 'd'])
        assert_equal(len(db), 2)
        assert_equal(db['a'], [1, 3])
        assert_almost_equal(db['b'], [2.0, 4.0])
        assert_equal(db['c'], ['a', 'c'])
        assert_equal(db['d'], ['b', 'd'])

        data = {'a': 1, 'b': 2.0, 'c': 'a', 'd': 'b'}
        db = Database(data=data)
        assert_equal(db.colnames(), ['a', 'b', 'c', 'd'])
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
        assert_equal(db.colnames(), [])
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
        assert_equal(db.colnames(), ['a'])
        assert_equal(len(db), 0)

    def test_database_add_column_dtype(self):
        db = Database()
        db.add_column('a', dtype=np.dtype('i8'))
        assert_equal(db.colnames(), ['a'])
        assert_equal(len(db), 0)

    def test_database_add_column_dtype_invalid(self):
        db = Database()
        with pytest.raises(KeyError):
            db.add_column('a', dtype='M8')

    def test_database_add_column_invalid_name(self):
        db = Database()
        with pytest.raises(ValueError):
            db.add_column(_ID_KEY)
        with pytest.raises(ValueError):
            db.add_column('table')
        with pytest.raises(ValueError):
            db.add_column('default')

    def test_database_add_column_data(self):
        db = Database()
        db.add_column('a', data=[1, 2, 3])
        assert_equal(db.colnames(), ['a'])
        assert_equal(len(db), 3)
        assert_equal(db['a'], [1, 2, 3])

    def test_database_add_column_data_invalid(self):
        db = Database()
        db.add_column('a', data=[1, 2, 3])
        assert_equal(db.colnames(), ['a'])
        assert_equal(len(db), 3)
        assert_equal(db['a'], [1, 2, 3])

        with pytest.raises(ValueError,
                           match="data must have the same length"):
            db.add_column('b', data=[1, 2, 3, 4])

    def test_database_set_column(self):
        db = Database()
        db.add_column('a', data=np.zeros(10))
        db.set_column('a', np.arange(10, 20))
        assert_equal(db.colnames(), ['a'])
        assert_equal(len(db), 10)
        assert_equal(db['a'], np.arange(10, 20))

    def test_database_set_column_invalid(self):
        db = Database()
        db.add_column('a', data=np.zeros(10))
        with pytest.raises(ValueError,
                           match="data must have the same length"):
            db.set_column('a', np.arange(10, 20, 2))

    def test_database_set_item(self):
        # all setitem should fail
        db = Database()
        db.add_column('a', data=np.zeros(10))
        with pytest.raises(NotImplementedError,
                           match='use set_item, set_col or set_row instead.'):
            db['a'] = np.arange(10, 20)
        with pytest.raises(NotImplementedError,
                           match='use set_item, set_col or set_row instead.'):
            db[1] = 1
        with pytest.raises(NotImplementedError,
                           match='use set_item, set_col or set_row instead.'):
            db[1]['a'] = 1


class Test_Database_Access:
    def test_database_get_column(self):
        db = Database()
        db.add_column('a', data=np.zeros(10))
        db.add_column('b', data=np.ones(10))
        assert_equal(db.get_column('a'), np.zeros(10))
        assert_equal(db.get_column('b'), np.ones(10))

    def test_database_get_column_invalid(self):
        db = Database()
        db.add_column('a', data=np.zeros(10))
        db.add_column('b', data=np.ones(10))
        with pytest.raises(KeyError):
            db.get_column('c')

    def test_get_row(self):
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        assert_equal(db.get_row(0), {'a': 10, 'b': 0})
        assert_equal(db.get_row(5), {'a': 15, 'b': 5})
        assert_equal(db.get_row(9), {'a': 19, 'b': 9})
        assert_equal(db.get_row(-1), {'a': 19, 'b': 9})
        assert_equal(db.get_row(-5), {'a': 15, 'b': 5})
        assert_equal(db.get_row(-10), {'a': 10, 'b': 0})

    def test_get_row_invalid(self):
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        with pytest.raises(IndexError):
            db.get_row(11)
        with pytest.raises(IndexError):
            db.get_row(-11)

    def test_get_item_str(self):
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        assert_is_instance(db['a'], list)
        assert_is_instance(db['b'], list)
        assert_equal(db['a'], np.arange(10, 20))
        assert_equal(db['b'], np.arange(10))

        db = Database(data={'a': 1, 'b': 2})
        assert_is_not_instance(db['a'], list)
        assert_is_not_instance(db['a'], list)
        assert_equal(db['a'], 1)
        assert_equal(db['b'], 2)

    def test_get_item_int(self):
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        assert_is_instance(db[0], Database)
        assert_equal(db[0].colnames(), ['a', 'b'])
        assert_equal(db[0].values(), [[10, 0]])
        assert_is_instance(db[-1], Database)
        assert_equal(db[-1].colnames(), ['a', 'b'])
        assert_equal(db[-1].values(), [[19, 9]])
        assert_is_instance(db[5], Database)
        assert_equal(db[5].colnames(), ['a', 'b'])
        assert_equal(db[5].values(), [[15, 5]])
        assert_is_instance(db[-5], Database)
        assert_equal(db[-5].colnames(), ['a', 'b'])
        assert_equal(db[-5].values(), [[15, 5]])

    def test_get_item_int_invalid(self):
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        with pytest.raises(IndexError):
            db[11]
        with pytest.raises(IndexError):
            db[-11]

    def test_get_item_slice(self):
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        assert_is_instance(db[:], Database)
        assert_equal(db[:].colnames(), ['a', 'b'])
        assert_equal(db[:].values(), [[10, 0], [11, 1], [12, 2], [13, 3],
                                      [14, 4], [15, 5], [16, 6], [17, 7],
                                      [18, 8], [19, 9]])
        assert_is_instance(db[5:], Database)
        assert_equal(db[5:].colnames(), ['a', 'b'])
        assert_equal(db[5:].values(), [[15, 5], [16, 6], [17, 7], [18, 8],
                                       [19, 9]])
        assert_is_instance(db[-5:], Database)
        assert_equal(db[-5:].colnames(), ['a', 'b'])
        assert_equal(db[-5:].values(), [[15, 5], [16, 6], [17, 7], [18, 8],
                                        [19, 9]])
        assert_is_instance(db[:-5], Database)
        assert_equal(db[:-5].colnames(), ['a', 'b'])
        assert_equal(db[:-5].values(), [[10, 0], [11, 1], [12, 2], [13, 3],
                                        [14, 4]])
        assert_is_instance(db[5:-5], Database)
        assert_equal(db[5:-5].colnames(), ['a', 'b'])
        assert_equal(db[5:-5].values(), [])

    def test_get_item_list_str(self):
        leters = ['a', 'b', 'c', 'd', 'e',
                  'f', 'g', 'h', 'i', 'j']
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        db.add_column('c', data=leters)
        assert_is_instance(db[['a', 'b']], Database)
        assert_equal(db[['a', 'b']].colnames(), ['a', 'b'])
        assert_equal(db[['b', 'c']].values(), list(zip(np.arange(10), leters)))

    def test_get_item_list_str_invalid(self):
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        with pytest.raises(KeyError):
            db[['a', 'b', 'c']]

    def test_get_item_list_int(self):
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        assert_is_instance(db[[0, 1]], Database)
        assert_equal(db[[0, 1]].colnames(), ['a', 'b'])
        assert_equal(db[[0, 1]].values(), [[10, 0], [11, 1]])
        assert_is_instance(db[[-1, -2]], Database)
        assert_equal(db[[-1, -2]].colnames(), ['a', 'b'])
        assert_equal(db[[-1, -2]].values(), [[19, 9], [18, 8]])
        assert_is_instance(db[[-5, -4]], Database)
        assert_equal(db[[-5, -4]].colnames(), ['a', 'b'])
        assert_equal(db[[-5, -4]].values(), [[15, 5], [16, 6]])

    def test_get_item_list_int_invalid(self):
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        with pytest.raises(IndexError):
            db[[11, 12]]
        with pytest.raises(IndexError):
            db[[-11, -12]]


class Test_Database_Select:
    def test_select_one_column(self):
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        sel = db.select(columns=['a'])
        assert_is_instance(sel, list)
        assert_equal(sel, list(zip(np.arange(10, 20))))

    def test_select_two_columns(self):
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        sel = db.select(columns=['a', 'b'])
        assert_is_instance(sel, list)
        assert_equal(sel, list(zip(np.arange(10, 20), np.arange(10))))

    def test_select_two_columns_invalid(self):
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        with pytest.raises(sqlite3.OperationalError):
            db.select(columns=['a', 'c'])

    def test_select_two_columns_invalid_type(self):
        db = Database()
        db.add_column('a', data=np.arange(10, 20))
        db.add_column('b', data=np.arange(10))
        with pytest.raises(TypeError):
            db.select(columns=['a', 1])
