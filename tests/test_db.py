# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
from astropop._db import SQLColumn, SQLRow, SQLTable, _ID_KEY, \
                         SQLDatabase, _sanitize_colnames
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


class Test_SQLDatabase_Creation_Modify:
    def test_sql_db_creation(self):
        db = SQLDatabase(':memory:')
        assert_equal(db.table_names, [])
        assert_equal(len(db), 0)

        db.add_table('test')
        assert_equal(db.table_names, ['test'])
        assert_equal(len(db), 1)

    def test_sql_add_column_name_and_data(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        assert_equal(db.get_column('test', 'a').values, np.arange(10, 20))
        assert_equal(db.get_column('test', 'b').values, np.arange(20, 30))
        assert_equal(db.column_names('test'), ['a', 'b'])
        assert_equal(len(db), 1)
        assert_equal(db.table_names, ['test'])

    def test_sql_add_column_only_name(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a')
        db.add_column('test', 'b')

        assert_equal(db.get_column('test', 'a').values, [])
        assert_equal(db.get_column('test', 'b').values, [])
        assert_equal(db.column_names('test'), ['a', 'b'])
        assert_equal(len(db), 1)
        assert_equal(db.table_names, ['test'])

    def test_sql_add_column_name_and_dtype(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', dtype=np.int32)
        db.add_column('test', 'b', dtype='f8')

        assert_equal(db.get_column('test', 'a').values, [])
        assert_equal(db.get_column('test', 'b').values, [])
        assert_equal(db.column_names('test'), ['a', 'b'])
        assert_equal(len(db), 1)
        assert_equal(db.table_names, ['test'])

    def test_sql_add_table_from_data_table(self):
        db = SQLDatabase(':memory:')
        d = Table(names=['a', 'b'], data=[np.arange(10, 20), np.arange(20, 30)])
        db.add_table('test', data=d)

        assert_equal(db.get_column('test', 'a').values, np.arange(10, 20))
        assert_equal(db.get_column('test', 'b').values, np.arange(20, 30))
        assert_equal(db.column_names('test'), ['a', 'b'])
        assert_equal(len(db), 1)
        assert_equal(db.table_names, ['test'])

    def test_sql_add_table_from_data_ndarray(self):
        dtype = [('a', 'i4'), ('b', 'f8')]
        data = np.array([(1, 2.0), (3, 4.0), (5, 6.0), (7, 8.0)], dtype=dtype)
        db = SQLDatabase(':memory:')
        db.add_table('test', data=data)

        assert_equal(db.get_column('test', 'a').values, [1, 3, 5, 7])
        assert_equal(db.get_column('test', 'b').values, [2.0, 4.0, 6.0, 8.0])
        assert_equal(db.column_names('test'), ['a', 'b'])
        assert_equal(len(db), 1)
        assert_equal(db.table_names, ['test'])

    def test_sql_add_table_from_data_ndarray_untyped(self):
        # Untyped ndarray should fail in get column names
        data = np.array([(1, 2.0), (3, 4.0), (5, 6.0), (7, 8.0)])
        db = SQLDatabase(':memory:')
        with pytest.raises(TypeError):
            db.add_table('test', data=data)

    def test_sql_add_table_from_data_invalid(self):
        db = SQLDatabase(':memory:')
        with assert_raises(TypeError):
            db.add_table('test', data=[1, 2, 3])
        with assert_raises(TypeError):
            db.add_table('test', data=1)
        with assert_raises(TypeError):
            db.add_table('test', data=1.0)
        with assert_raises(TypeError):
            db.add_table('test', data='test')


class Test_SQLDatabase_Access:
    def test_sql_get_table(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        assert_equal(db.get_table('test').values, list(zip(np.arange(10, 20),
                                                           np.arange(20, 30))))
        assert_is_instance(db.get_table('test'), SQLTable)

        with pytest.raises(KeyError):
            db.get_table('not_a_table')

    def test_sql_get_column(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        assert_equal(db.get_column('test', 'a').values, np.arange(10, 20))
        assert_equal(db.get_column('test', 'b').values, np.arange(20, 30))
        assert_is_instance(db.get_column('test', 'a'), SQLColumn)
        assert_is_instance(db.get_column('test', 'b'), SQLColumn)

        # same access from table
        assert_equal(db.get_table('test').get_column('a').values, np.arange(10, 20))
        assert_equal(db.get_table('test').get_column('b').values, np.arange(20, 30))
        assert_is_instance(db.get_table('test').get_column('a'), SQLColumn)
        assert_is_instance(db.get_table('test').get_column('b'), SQLColumn)

        with pytest.raises(KeyError):
            db.get_column('test', 'c')
        with pytest.raises(KeyError):
            db.get_table('test').get_column('c')

    def test_sql_get_row(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        assert_equal(db.get_row('test', 4).values, (14, 24))
        assert_is_instance(db.get_row('test', 4), SQLRow)

        assert_equal(db.get_row('test', -1).values, (19, 29))
        assert_is_instance(db.get_row('test', -1), SQLRow)

        # same access from table
        assert_equal(db.get_table('test').get_row(4).values, [14, 24])
        assert_is_instance(db.get_table('test').get_row(4), SQLRow)

        with pytest.raises(IndexError):
            db.get_row('test', 11)
        with pytest.raises(IndexError):
            db.get_row('test', -11)
        with pytest.raises(IndexError):
            db.get_table('test').get_row(11)
        with pytest.raises(IndexError):
            db.get_table('test').get_row(-11)

    def test_sql_getitem(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        assert_equal(db['test']['a'].values, np.arange(10, 20))
        assert_equal(db['test']['b'].values, np.arange(20, 30))
        assert_is_instance(db['test']['a'], SQLColumn)
        assert_is_instance(db['test']['b'], SQLColumn)

        assert_equal(db['test'][4].values, (14, 24))
        assert_is_instance(db['test'][4], SQLRow)
        assert_equal(db['test'][-1].values, (19, 29))
        assert_is_instance(db['test'][-1], SQLRow)

        with pytest.raises(KeyError):
            db['test']['c']
        with pytest.raises(KeyError):
            db['not_a_table']['a']

        with pytest.raises(IndexError):
            db['test'][11]
        with pytest.raises(IndexError):
            db['test'][-11]

    def test_sql_getitem_tuple(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        assert_equal(db['test', 'a'].values, np.arange(10, 20))
        assert_equal(db['test', 'b'].values, np.arange(20, 30))
        assert_is_instance(db['test', 'a'], SQLColumn)
        assert_is_instance(db['test', 'b'], SQLColumn)

        assert_equal(db['test', 4].values, (14, 24))
        assert_is_instance(db['test', 4], SQLRow)
        assert_equal(db['test', -1].values, (19, 29))
        assert_is_instance(db['test', -1], SQLRow)

        # TODO: tests with column and row

    def test_sql_getitem_table_force(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        with pytest.raises(ValueError):
            db[1]
        with pytest.raises(ValueError):
            db[1, 2]
        with pytest.raises(ValueError):
            db[1, 2, 'test']
        with pytest.raises(ValueError):
            db[[1, 2], 'test']


class Test_SQLDatabase_SQLCommands:
    def test_sql_select_where(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        a = db.select('test', columns='a', where={'a': 15})
        assert_equal(a, 15)

        a = db.select('test', columns=['a', 'b'], where={'b': 22})
        assert_equal(a, [(12, 22)])

        a = db.select('test', columns=['a', 'b'], where=None)
        assert_equal(a, list(zip(np.arange(10, 20), np.arange(20, 30))))

        a = db.select('test', columns=['a', 'b'], where=['a > 12', 'b < 26'])
        assert_equal(a, [(13, 23), (14, 24), (15, 25)])

    def test_sql_select_limit_offset(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        a = db.select('test', columns='a', limit=1)
        assert_equal(a, 10)

        a = db.select('test', columns='a', limit=3, offset=2)
        assert_equal(a, [[12], [13], [14]])

    def test_sql_select_invalid(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        with pytest.raises(sqlite3.OperationalError,
                           match='no such column: c'):
            db.select('test', columns=['c'])

        with pytest.raises(ValueError,
                           match='offset cannot be used without limit.'):
            db.select('test', columns='a', offset=1)

        with pytest.raises(TypeError, match='where must be'):
            db.select('test', columns='a', where=1)

        with pytest.raises(TypeError, match='if where is a list'):
            db.select('test', columns='a', where=[1, 2, 3])

        with pytest.raises(sqlite3.IntegrityError):
            db.select('test', limit=3.14)

        with pytest.raises(sqlite3.OperationalError):
            db.select('test', order=5)

    def test_sql_select_order(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30)[::-1])

        a = db.select('test', order='b')
        assert_equal(a, list(zip(np.arange(10, 20),
                                 np.arange(20, 30)[::-1]))[::-1])

        a = db.select('test', order='b', limit=2)
        assert_equal(a, [(19, 20), (18, 21)])

        a = db.select('test', order='b', limit=2, offset=2)
        assert_equal(a, [(17, 22), (16, 23)])

        a = db.select('test', order='b', where='a < 15')
        assert_equal(a, [(14, 25), (13, 26), (12, 27), (11, 28), (10, 29)])


        a = db.select('test', order='b', where='a < 15', limit=3)
        assert_equal(a, [(14, 25), (13, 26), (12, 27)])

        a = db.select('test', order='b', where='a < 15', limit=3, offset=2)
        assert_equal(a, [(12, 27), (11, 28), (10, 29)])

    def test_sql_count(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        assert_equal(db.count('test'), 10)
        assert_equal(db.count('test', where={'a': 15}), 1)
        assert_equal(db.count('test', where={'a': 15, 'b': 22}), 0)
        assert_equal(db.count('test', where='a > 15'), 4)
        assert_equal(db.count('test', where=['a > 15', 'b < 27']), 1)


class Test_SQLRow:
    pass


class Test_SQLTable:
    pass


class Test_SQLColumn:
    pass
