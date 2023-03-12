# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
from astropop._db import SQLColumn, SQLRow, SQLTable, \
                         SQLDatabase, _sanitize_colnames, \
                         _b32_decode, _b32_encode
import numpy as np
from astropy.table import Table
from astropop.testing import *
import sqlite3
import copy


class Test_SanitizeColnames:
    def test_sanitize_encode_decode(self):
        for i in ['test-2', 'test!2', 'test@2', 'test#2', 'test$2',
                  'test&2', 'test*2', 'test(2)', 'test)2', 'test[2]', 'test]2',
                  'test{2}', 'test}2', 'test|2', 'test\\2', 'test^2', 'test~2',
                  'test"2', 'test\'2', 'test`2', 'test<2', 'test>2', 'test=2',
                  'test,2', 'test;2', 'test:2', 'test?2', 'test/2']:
            result = f'__b32c__{_b32_encode(i)}'.lower()
            assert_equal(_sanitize_colnames(i, allow_b32_encode=True),
                         result)
            assert_equal(_b32_decode(result[8:]), i.lower())

    def test_sanitize_string_fails(self):
        for i in ['test-2', 'test!2', 'test@2', 'test#2', 'test$2',
                  'test&2', 'test*2', 'test(2)', 'test)2', 'test[2]', 'test]2',
                  'test{2}', 'test}2', 'test|2', 'test\\2', 'test^2', 'test~2',
                  'test"2', 'test\'2', 'test`2', 'test<2', 'test>2', 'test=2',
                  'test,2', 'test;2', 'test:2', 'test?2', 'test/2']:
            with pytest.raises(ValueError, match='Invalid column name'):
                _sanitize_colnames(i)

    def test_sanitize_string_passes(self):
        for i in ['test', 'test_1', 'test_1_2', 'test_1_2', 'Test', 'Test_1']:
            assert_equal(_sanitize_colnames(i), i.lower())

    def test_sanitize_list_passes(self):
        for i in [['test', 'test_1', 'test_1_2', 'test_1_2', 'Test', 'Test_1']]:
            assert_equal(_sanitize_colnames(i), [j.lower() for j in i])

    def test_sanitize_list_fails(self):
        for i in ['test-2', 'test!2', 'test@2', 'test#2', 'test$2',
                  'test&2', 'test*2', 'test(2)', 'test)2', 'test[2]', 'test]2',
                  'test{2}', 'test}2', 'test|2', 'test\\2', 'test^2', 'test~2',
                  'test"2', 'test\'2', 'test`2', 'test<2', 'test>2', 'test=2',
                  'test,2', 'test;2', 'test:2', 'test?2', 'test/2']:
            with pytest.raises(ValueError, match='Invalid column name'):
                _sanitize_colnames([i, 'test'])
            with pytest.raises(ValueError, match='Invalid column name'):
                _sanitize_colnames(['test', i])

    def test_sanitize_dict_passes(self):
        d = {'test': 1, 'test_1': 2, 'test_1_2': 3}
        assert_equal(_sanitize_colnames(d),
                     {'test': 1, 'test_1': 2, 'test_1_2': 3})

    def test_sanitize_dict_fails(self):
        d = {'test': 1, 'test_1': 2, 'test_1_2': 3}
        for i in ['test-2', 'test!2', 'test@2', 'test#2', 'test$2',
                  'test&2', 'test*2', 'test(2)', 'test)2', 'test[2]', 'test]2',
                  'test{2}', 'test}2', 'test|2', 'test\\2', 'test^2', 'test~2',
                  'test"2', 'test\'2', 'test`2', 'test<2', 'test>2', 'test=2',
                  'test,2', 'test;2', 'test:2', 'test?2', 'test/2']:
            with pytest.raises(ValueError, match='Invalid column name'):
                _sanitize_colnames({**d, i: 4})


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

    def test_sql_add_table_from_data_dict(self):
        d = {'a': np.arange(10, 20), 'b': np.arange(20, 30)}
        db = SQLDatabase(':memory:')
        db.add_table('test', data=d)

        assert_equal(db.get_column('test', 'a').values, np.arange(10, 20))
        assert_equal(db.get_column('test', 'b').values, np.arange(20, 30))
        assert_equal(db.column_names('test'), ['a', 'b'])
        assert_equal(len(db), 1)
        assert_equal(db.table_names, ['test'])

    def test_sql_add_table_from_data_ndarray_untyped(self):
        # Untyped ndarray should fail in get column names
        data = np.array([(1, 2.0), (3, 4.0), (5, 6.0), (7, 8.0)])
        db = SQLDatabase(':memory:')
        with pytest.raises(ValueError):
            db.add_table('test', data=data)

    @pytest.mark.parametrize('data, error', [([1, 2, 3], ValueError),
                                             (1, TypeError),
                                             (1.0, TypeError),
                                             ('test', TypeError)])
    def test_sql_add_table_from_data_invalid(self, data, error):
        db = SQLDatabase(':memory:')
        with assert_raises(error):
            db.add_table('test', data=data)

    def test_sql_add_table_columns(self):
        db = SQLDatabase(':memory:')
        db.add_table('test', columns=['a', 'b'])

        assert_equal(db.get_column('test', 'a').values, [])
        assert_equal(db.get_column('test', 'b').values, [])
        assert_equal(db.column_names('test'), ['a', 'b'])
        assert_equal(len(db), 1)
        assert_equal(db.table_names, ['test'])

    def test_sql_add_table_columns_data(self):
        db = SQLDatabase(':memory:')
        with pytest.raises(ValueError):
            db.add_table('test', columns=['a', 'b'], data=[1, 2, 3])

    def test_sql_add_row(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a')
        db.add_column('test', 'b')
        db.add_rows('test', dict(a=1, b=2))
        db.add_rows('test', dict(a=[3, 5], b=[4, 6]))

        assert_equal(db.get_column('test', 'a').values, [1, 3, 5])
        assert_equal(db.get_column('test', 'b').values, [2, 4, 6])
        assert_equal(len(db), 1)
        assert_equal(db.table_names, ['test'])

    def test_sql_add_row_types(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        for k in ['a', 'b', 'c', 'd', 'e']:
            db.add_column('test', k)

        db.add_rows('test', dict(a=1, b='a', c=True, d=b'a', e=3.14))
        db.add_rows('test', dict(a=2, b='b', c=False, d=b'b', e=2.71))

        assert_equal(db.get_column('test', 'a').values, [1, 2])
        assert_equal(db.get_column('test', 'b').values, ['a', 'b'])
        assert_equal(db.get_column('test', 'c').values, [1, 0])
        assert_equal(db.get_column('test', 'd').values, [b'a', b'b'])
        assert_almost_equal(db.get_column('test', 'e').values, [3.14, 2.71])
        assert_equal(len(db), 1)
        assert_equal(db.table_names, ['test'])

    def test_sql_add_row_invalid(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a')
        db.add_column('test', 'b')
        with assert_raises(ValueError):
            db.add_rows('test', [1, 2, 3])

    def test_sql_add_row_add_columns(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a')
        db.add_column('test', 'b')
        db.add_rows('test', dict(a=1, b=2))
        db.add_rows('test', dict(a=3, c=4), add_columns=False)
        db.add_rows('test', dict(a=5, d=6), add_columns=True)

        assert_equal(db.get_column('test', 'a').values, [1, 3, 5])
        assert_equal(db.get_column('test', 'b').values, [2, None, None])
        assert_equal(len(db), 1)
        assert_equal(db.table_names, ['test'])
        assert_equal(db.column_names('test'), ['a', 'b', 'd'])

    def test_sql_add_row_superpass_64_limit(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_rows('test', {f'col{i}': np.arange(10) for i in range(128)},
                     add_columns=True)
        assert_equal(db.column_names('test'), [f'col{i}' for i in range(128)])

        for i, v in enumerate(db.select('test')):
            assert_equal(v, [i]*128)

    def test_sqltable_add_column(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db['test'].add_column('a')
        db['test'].add_column('b')
        db['test'].add_column('c', data=[1, 2, 3])

        assert_equal(db.get_column('test', 'a').values, [None, None, None])
        assert_equal(db.get_column('test', 'b').values, [None, None, None])
        assert_equal(db.get_column('test', 'c').values, [1, 2, 3])
        assert_equal(len(db), 1)
        assert_equal(db.table_names, ['test'])
        assert_equal(db.column_names('test'), ['a', 'b', 'c'])
        assert_equal(len(db['test']), 3)

    def test_sqltable_add_row_add_columns(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a')
        db.add_column('test', 'b')
        db['test'].add_rows(dict(a=1, b=2))
        db['test'].add_rows(dict(a=3, c=4), add_columns=False)
        db['test'].add_rows(dict(a=5, d=6), add_columns=True)

        assert_equal(db.get_column('test', 'a').values, [1, 3, 5])
        assert_equal(db.get_column('test', 'b').values, [2, None, None])
        assert_equal(len(db), 1)
        assert_equal(db.table_names, ['test'])
        assert_equal(db.column_names('test'), ['a', 'b', 'd'])

    def test_sql_set_column(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', [1, 3, 5])
        db.add_column('test', 'b', [2, 4, 6])

        db.set_column('test', 'a', [10, 20, 30])
        db.set_column('test', 'b', [20, 40, 60])

        assert_equal(db.get_column('test', 'a').values, [10, 20, 30])
        assert_equal(db.get_column('test', 'b').values, [20, 40, 60])

        with pytest.raises(KeyError):
            db.set_column('test', 'c', [10, 20, 30])
        with pytest.raises(ValueError):
            db.set_column('test', 'a', [10, 20, 30, 40])

    def test_sql_set_row(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', [1, 3, 5])
        db.add_column('test', 'b', [2, 4, 6])

        db.set_row('test', 0, dict(a=10, b=20))
        db.set_row('test', 1, [20, 40])
        db.set_row('test', 2, np.array([30, 60]))

        assert_equal(db.get_column('test', 'a').values, [10, 20, 30])
        assert_equal(db.get_column('test', 'b').values, [20, 40, 60])

        with pytest.raises(IndexError):
            db.set_row('test', 3, dict(a=10, b=20))
        with pytest.raises(IndexError):
            db.set_row('test', -4, dict(a=10, b=20))

    def test_sql_set_item(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', [1, 3, 5])
        db.add_column('test', 'b', [2, 4, 6])

        db.set_item('test', 'a', 0, 10)
        db.set_item('test', 'b', 1, 'a')
        assert_equal(db.get_column('test', 'a').values, [10, 3, 5])
        assert_equal(db.get_column('test', 'b').values, [2, 'a', 6])

    def test_sql_setitem_tuple_only(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        with pytest.raises(KeyError):
            db[1] = 0
        with pytest.raises(KeyError):
            db['notable'] = 0
        with pytest.raises(KeyError):
            db[['test', 0]] = 0
        with pytest.raises(KeyError):
            db[1, 0] = 0

    def test_sql_setitem(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        db['test', 'a'] = np.arange(50, 60)
        db['test', 0] = {'a': 1, 'b': 2}
        db['test', 'b', 5] = -999

        expect = np.transpose([np.arange(50, 60), np.arange(20, 30)])
        expect[0] = [1, 2]
        expect[5, 1] = -999

        assert_equal(db['test'].values, expect)

    def test_sql_droptable(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', [1, 3, 5])
        db.add_column('test', 'b', [2, 4, 6])

        db.drop_table('test')
        assert_equal(db.table_names, [])
        with pytest.raises(KeyError):
            db['test']

    def test_sql_copy(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', [1, 3, 5])
        db.add_column('test', 'b', [2, 4, 6])

        db2 = db.copy()
        assert_equal(db2.table_names, ['test'])
        assert_equal(db2.column_names('test'), ['a', 'b'])
        assert_equal(db2.get_column('test', 'a').values, [1, 3, 5])
        assert_equal(db2.get_column('test', 'b').values, [2, 4, 6])

    def test_sql_copy_indexes(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', np.arange(1, 101, 2))
        db.add_column('test', 'b', np.arange(2, 102, 2))

        db2 = db.copy(indexes={'test': [30, 24, 32, 11]})
        assert_equal(db2.table_names, ['test'])
        assert_equal(db2.column_names('test'), ['a', 'b'])
        assert_equal(db2.get_column('test', 'a').values, [23, 49, 61, 65])
        assert_equal(db2.get_column('test', 'b').values, [24, 50, 62, 66])

    def test_sql_delete_row(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', [1, 3, 5])
        db.add_column('test', 'b', [2, 4, 6])

        db.delete_row('test', 1)
        assert_equal(db.get_column('test', 'a').values, [1, 5])
        assert_equal(db.get_column('test', 'b').values, [2, 6])

        with pytest.raises(IndexError):
            db.delete_row('test', 2)
        with pytest.raises(IndexError):
            db.delete_row('test', -4)

    def test_sql_delete_column(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', [1, 3, 5])
        db.add_column('test', 'b', [2, 4, 6])

        db.delete_column('test', 'b')
        assert_equal(db.column_names('test'), ['a'])
        assert_equal(db.get_column('test', 'a').values, [1, 3, 5])

        with pytest.raises(KeyError, match='does not exist'):
            db.delete_column('test', 'b')

    def test_sql_add_nonstd_column_keys(self):
        db = SQLDatabase(':memory:', allow_colname_encode=True)
        db.add_table('test')
        db.add_column('test', 'a', [1, 3, 5])
        db.add_column('test', 'b', [2, 4, 6])
        db.add_column('test', 'c a', [7, 8, 9])
        db.add_column('test', 'd-!&*#@%_+=/<>,.?^~`[]{}', [7, 8, 9])

        assert_equal(db.column_names('test'), ['a', 'b', 'c a',
                                               'd-!&*#@%_+=/<>,.?^~`[]{}'])
        assert_equal(db.get_column('test', 'c a').values, [7, 8, 9])
        assert_equal(db.get_column('test', 'd-!&*#@%_+=/<>,.?^~`[]{}').values,
                     [7, 8, 9])

    def test_sql_add_nonstd_column_keys_fail(self):
        # Standard behavior is to disallow non-standard column names
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', [1, 3, 5])
        db.add_column('test', 'b', [2, 4, 6])
        with pytest.raises(ValueError, match='Invalid column name'):
            db.add_column('test', 'c a', [7, 8, 9])
        with pytest.raises(ValueError, match='Invalid column name'):
            db.add_column('test', 'd-!&*#@%_+=/<>,.?^~`[]{}', [7, 8, 9])


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

    def test_sql_get_table_empty(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')

        assert_equal(len(db.get_table('test')), 0)
        assert_is_instance(db.get_table('test'), SQLTable)

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

        assert_equal(db['test', 'a', 4], 14)
        assert_equal(db['test', 'b', 4], 24)
        assert_equal(db['test', 'a', -1], 19)
        assert_equal(db['test', 'b', -1], 29)
        assert_equal(db['test', 4, 'a'], 14)
        assert_equal(db['test', 4, 'b'], 24)
        assert_equal(db['test', -1, 'a'], 19)
        assert_equal(db['test', -1, 'b'], 29)

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


class Test_SQLDatabase_PropsComms:
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

        with pytest.raises(TypeError):
            db.select('test', limit=3.14)

        with pytest.raises(TypeError):
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

    def test_sql_prop_db(self, tmp_path):
        db = SQLDatabase(':memory:')
        assert_equal(db.db, ':memory:')

        db = SQLDatabase(str(tmp_path / 'test.db'))
        assert_equal(db.db, str(tmp_path / 'test.db'))

    def test_sql_prop_table_names(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_table('test2')
        assert_equal(db.table_names, ['test', 'test2'])

    def test_sql_prop_column_names(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))
        assert_equal(db.column_names('test'), ['a', 'b'])

    def test_sql_repr(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))
        db.add_table('test2')
        db.add_column('test2', 'a', data=np.arange(10, 20))
        db.add_column('test2', 'b', data=np.arange(20, 30))

        expect = f"SQLDatabase ':memory:' at {hex(id(db))}:\n"
        expect += "\ttest: 2 columns 10 rows\n"
        expect += "\ttest2: 2 columns 10 rows"
        assert_equal(repr(db), expect)

        db = SQLDatabase(':memory:')
        expect = f"SQLDatabase ':memory:' at {hex(id(db))}:\n"
        expect += "\tEmpty database."
        assert_equal(repr(db), expect)

    def test_sql_len(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_table('test2')
        db.add_table('test3')

        assert_equal(len(db), 3)

    def test_sql_index_of(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))

        assert_equal(db.index_of('test', {'a': 15}), 5)
        assert_equal(db.index_of('test', 'b >= 27'), [7, 8, 9])
        assert_equal(db.index_of('test', {'a': 1, 'b': 2}), [])


class Test_SQLRow:
    @property
    def db(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))
        return db

    def test_row_copy_error(self):
        db = self.db
        row = db['test'][1]
        with pytest.raises(NotImplementedError, match='Cannot copy'):
            copy.copy(row)
        with pytest.raises(NotImplementedError, match='Cannot copy'):
            copy.deepcopy(row)

    def test_row_basic_properties(self):
        db = self.db
        row = db['test'][0]
        assert_is_instance(row, SQLRow)
        assert_equal(row.table, 'test')
        assert_equal(row.index, 0)
        assert_equal(row.column_names, ['a', 'b'])
        assert_equal(row.keys, ['a', 'b'])
        assert_equal(row.values, [10, 20])
        assert_equal(row.as_dict(), {'a': 10, 'b': 20})

    def test_row_iter(self):
        db = self.db
        row = db['test'][0]
        assert_is_instance(row, SQLRow)

        v = 10
        for i in row:
            assert_equal(i, v)
            v += 10

    def test_row_getitem(self):
        db = self.db
        row = db['test'][0]
        assert_is_instance(row, SQLRow)

        assert_equal(row['a'], 10)
        assert_equal(row['b'], 20)

        with pytest.raises(KeyError):
            row['c']

        assert_equal(row[0], 10)
        assert_equal(row[1], 20)
        assert_equal(row[-1], 20)
        assert_equal(row[-2], 10)

        with pytest.raises(IndexError):
            row[2]
        with pytest.raises(IndexError):
            row[-3]

    def test_row_setitem(self):
        db = self.db
        row = db['test'][0]
        assert_is_instance(row, SQLRow)

        row['a'] = 1
        row['b'] = 1
        assert_equal(db['test']['a'], [1, 11, 12, 13, 14,
                                       15, 16, 17, 18, 19])
        assert_equal(db['test']['b'], [1, 21, 22, 23, 24,
                                       25, 26, 27, 28, 29])

        with pytest.raises(KeyError):
            row['c'] = 1
        with pytest.raises(KeyError):
            row[2] = 1
        with pytest.raises(KeyError):
            row[-3] = 1

    def test_row_contains(self):
        db = self.db
        row = db['test'][0]
        assert_is_instance(row, SQLRow)

        assert_true(10 in row)
        assert_true(20 in row)
        assert_false('c' in row)
        assert_false('a' in row)
        assert_false('b' in row)

    def test_row_repr(self):
        db = self.db
        row = db['test'][0]
        assert_is_instance(row, SQLRow)
        assert_equal(repr(row), "SQLRow 0 in table 'test' {'a': 10, 'b': 20}")


class Test_SQLTable:
    @property
    def db(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))
        return db

    def test_table_copy_error(self):
        db = self.db
        table = db['test']
        with pytest.raises(NotImplementedError, match='Cannot copy'):
            copy.copy(table)
        with pytest.raises(NotImplementedError, match='Cannot copy'):
            copy.deepcopy(table)

    def test_table_basic_properties(self):
        db = self.db
        table = db['test']
        assert_equal(table.name, 'test')
        assert_equal(table.db, db.db)
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(table.values, list(zip(np.arange(10, 20),
                                            np.arange(20, 30))))

    def test_table_select(self):
        db = self.db
        table = db['test']

        a = table.select()
        assert_equal(a, list(zip(np.arange(10, 20),
                                 np.arange(20, 30))))

        a = table.select(order='a')
        assert_equal(a, list(zip(np.arange(10, 20),
                                 np.arange(20, 30))))

        a = table.select(order='a', limit=2)
        assert_equal(a, [(10, 20), (11, 21)])

        a = table.select(order='a', limit=2, offset=2)
        assert_equal(a, [(12, 22), (13, 23)])

        a = table.select(order='a', where='a < 15')
        assert_equal(a, [(10, 20), (11, 21), (12, 22), (13, 23), (14, 24)])

        a = table.select(order='a', where='a < 15', limit=3)
        assert_equal(a, [(10, 20), (11, 21), (12, 22)])

        a = table.select(order='a', where='a < 15', limit=3, offset=2)
        assert_equal(a, [(12, 22), (13, 23), (14, 24)])

        a = table.select(columns=['a'], where='a < 15')
        assert_equal(a, [(10,), (11,), (12,), (13,), (14,)])

    def test_table_as_table(self):
        db = self.db
        table = db['test']

        a = table.as_table()
        assert_is_instance(a, Table)
        assert_equal(a.colnames, ['a', 'b'])
        assert_equal(a, Table(names=['a', 'b'], data=[np.arange(10, 20),
                                                      np.arange(20, 30)]))

    def test_table_as_table_empty(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        table = db['test']

        a = table.as_table()
        assert_is_instance(a, Table)
        assert_equal(a.colnames, [])
        assert_equal(a, Table())

    def test_table_len(self):
        db = self.db
        table = db['test']
        assert_equal(len(table), 10)

    def test_table_iter(self):
        db = self.db
        table = db['test']

        v = 10
        for i in table:
            assert_equal(i, (v, v + 10))
            v += 1

    def test_table_contains(self):
        db = self.db
        table = db['test']

        assert_false(10 in table)
        assert_false(20 in table)
        assert_false('c' in table)
        assert_true('a' in table)
        assert_true('b' in table)

    def test_table_repr(self):
        db = self.db
        table = db['test']
        i = hex(id(table))

        expect = "SQLTable 'test' in database ':memory:':"
        expect += f"(2 columns x 10 rows)\n"
        expect += '\n'.join(table.as_table().__repr__().split('\n')[1:])
        assert_is_instance(table, SQLTable)
        assert_equal(repr(table), expect)

    def test_table_add_column(self):
        db = self.db
        table = db['test']

        table.add_column('c', data=np.arange(10, 20))
        assert_equal(table.column_names, ['a', 'b', 'c'])
        assert_equal(table.values, list(zip(np.arange(10, 20),
                                            np.arange(20, 30),
                                            np.arange(10, 20))))

        table.add_column('d', data=np.arange(20, 30))
        assert_equal(table.column_names, ['a', 'b', 'c', 'd'])
        assert_equal(table.values, list(zip(np.arange(10, 20),
                                            np.arange(20, 30),
                                            np.arange(10, 20),
                                            np.arange(20, 30))))

    def test_table_get_column(self):
        db = self.db
        table = db['test']

        a = table.get_column('a')
        assert_is_instance(a, SQLColumn)
        assert_equal(a.values, np.arange(10, 20))

        a = table.get_column('b')
        assert_is_instance(a, SQLColumn)
        assert_equal(a.values, np.arange(20, 30))

    def test_table_set_column(self):
        db = self.db
        table = db['test']

        table.set_column('a', np.arange(5, 15))
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(table.values, list(zip(np.arange(5, 15),
                                            np.arange(20, 30))))

    def test_table_set_column_invalid(self):
        db = self.db
        table = db['test']

        with assert_raises(ValueError):
            table.set_column('a', np.arange(5, 16))

        with assert_raises(KeyError):
            table.set_column('c', np.arange(5, 15))

    def test_table_add_row(self):
        db = self.db
        table = db['test']

        table.add_rows({'a': -1, 'b': -1})
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(len(table), 11)
        assert_equal(table[-1].values, (-1, -1))

        table.add_rows({'a': -2, 'c': -2}, add_columns=True)
        assert_equal(table.column_names, ['a', 'b', 'c'])
        assert_equal(len(table), 12)
        assert_equal(table[-1].values, (-2, None, -2))

        table.add_rows({'a': -3, 'd': -3}, add_columns=False)
        assert_equal(table.column_names, ['a', 'b', 'c'])
        assert_equal(len(table), 13)
        assert_equal(table[-1].values, (-3, None, None))

        # defult add_columns must be false
        table.add_rows({'a': -4, 'b': -4, 'c': -4, 'd': -4})
        assert_equal(table.column_names, ['a', 'b', 'c'])
        assert_equal(len(table), 14)
        assert_equal(table[-1].values, (-4, -4, -4))

    def test_table_add_row_invalid(self):
        db = self.db
        table = db['test']

        with assert_raises(ValueError):
            table.add_rows([1, 2, 3, 4])

        with assert_raises(TypeError):
            table.add_rows(2)

    def test_table_get_row(self):
        db = self.db
        table = db['test']

        a = table.get_row(0)
        assert_is_instance(a, SQLRow)
        assert_equal(a.values, (10, 20))

        a = table.get_row(1)
        assert_is_instance(a, SQLRow)
        assert_equal(a.values, (11, 21))

    def test_table_set_row(self):
        db = self.db
        table = db['test']

        table.set_row(0, {'a': 5, 'b': 15})
        expect = np.transpose([np.arange(10, 20), np.arange(20, 30)])
        expect[0] = [5, 15]
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(table.values, expect)

        expect[-1] = [-1, -1]
        table.set_row(-1, {'a': -1, 'b': -1})
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(table.values, expect)

        expect[-1] = [5, 5]
        table.set_row(-1, [5, 5])
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(table.values, expect)

    def test_table_set_row_invalid(self):
        db = self.db
        table = db['test']

        with pytest.raises(IndexError):
            table.set_row(10, {'a': -1, 'b': -1})
        with pytest.raises(IndexError):
            table.set_row(-11, {'a': -1, 'b': -1})

        with pytest.raises(TypeError):
            table.set_row(0, 'a')

    def test_table_getitem_int(self):
        db = self.db
        table = db['test']
        assert_is_instance(table, SQLTable)

        assert_equal(table[0].values, (10, 20))
        assert_equal(table[-1].values, (19, 29))

        with pytest.raises(IndexError):
            table[10]
        with pytest.raises(IndexError):
            table[-11]

    def test_table_getitem_str(self):
        db = self.db
        table = db['test']
        assert_is_instance(table, SQLTable)

        assert_equal(table['a'].values, np.arange(10, 20))
        assert_equal(table['b'].values, np.arange(20, 30))

        with pytest.raises(KeyError):
            table['c']

    def test_table_getitem_tuple(self):
        db = self.db
        table = db['test']
        assert_is_instance(table, SQLTable)

        assert_equal(table[('a',)].values, np.arange(10, 20))
        assert_is_instance(table[('a',)], SQLColumn)
        assert_equal(table[(1,)].values, (11, 21))
        assert_is_instance(table[(1,)], SQLRow)

        with pytest.raises(KeyError):
            table[('c')]
        with pytest.raises(IndexError):
            table[(11,)]

    def test_table_getitem_tuple_rowcol(self):
        db = self.db
        table = db['test']
        assert_is_instance(table, SQLTable)

        assert_equal(table['a', 0], 10)
        assert_equal(table['a', 1], 11)
        assert_equal(table['b', 0], 20)
        assert_equal(table['b', 1], 21)

        assert_equal(table[0, 'a'], 10)
        assert_equal(table[1, 'a'], 11)
        assert_equal(table[0, 'b'], 20)
        assert_equal(table[1, 'b'], 21)

        assert_equal(table['a', [0, 1, 2]], [10, 11, 12])
        assert_equal(table['b', [0, 1, 2]], [20, 21, 22])
        assert_equal(table[[0, 1, 2], 'b'], [20, 21, 22])
        assert_equal(table[[0, 1, 2], 'a'], [10, 11, 12])

        assert_equal(table['a', 2:5], [12, 13, 14])
        assert_equal(table['b', 2:5], [22, 23, 24])
        assert_equal(table[2:5, 'b'], [22, 23, 24])
        assert_equal(table[2:5, 'a'], [12, 13, 14])

        with pytest.raises(KeyError):
            table['c', 0]
        with pytest.raises(IndexError):
            table['a', 11]

        with pytest.raises(KeyError):
            table[0, 0]
        with pytest.raises(KeyError):
            table['b', 'a']
        with pytest.raises(KeyError):
            table[0, 1, 2]
        with pytest.raises(KeyError):
            table[0, 'a', 'b']

    def test_table_setitem_int(self):
        db = self.db
        table = db['test']
        assert_is_instance(table, SQLTable)

        table[0] = {'a': 5, 'b': 15}
        expect = np.transpose([np.arange(10, 20), np.arange(20, 30)])
        expect[0] = [5, 15]
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(table.values, expect)

        table[-1] = {'a': -1, 'b': -1}
        expect[-1] = [-1, -1]
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(table.values, expect)

        with pytest.raises(IndexError):
            table[10] = {'a': -1, 'b': -1}
        with pytest.raises(IndexError):
            table[-11] = {'a': -1, 'b': -1}

    def test_table_setitem_str(self):
        db = self.db
        table = db['test']
        assert_is_instance(table, SQLTable)

        table['a'] = np.arange(40, 50)
        expect = np.transpose([np.arange(40, 50), np.arange(20, 30)])
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(table.values, expect)

        table['b'] = np.arange(10, 20)
        expect = np.transpose([np.arange(40, 50), np.arange(10, 20)])
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(table.values, expect)

        with pytest.raises(KeyError):
            table['c'] = np.arange(10, 20)

    def test_table_setitem_tuple(self):
        db = self.db
        table = db['test']
        assert_is_instance(table, SQLTable)

        table[('a',)] = np.arange(40, 50)
        expect = np.transpose([np.arange(40, 50), np.arange(20, 30)])
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(table.values, expect)

        table[(1,)] = {'a': -1, 'b': -1}
        expect[1] = [-1, -1]
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(table.values, expect)

        with pytest.raises(KeyError):
            table[('c',)] = np.arange(10, 20)
        with pytest.raises(IndexError):
            table[(11,)] = np.arange(10, 20)

    def test_table_setitem_tuple_multiple(self):
        db = self.db
        table = db['test']
        assert_is_instance(table, SQLTable)
        expect = np.transpose([np.arange(10, 20), np.arange(20, 30)])

        table[('a', 1)] = 57
        expect[1, 0] = 57
        table['b', -1] = 32
        expect[-1, 1] = 32
        table[0, 'a'] = -1
        expect[0, 0] = -1
        table[5, 'b'] = 99
        expect[5, 1] = 99
        table['a', 3:6] = -999
        expect[3:6, 0] = -999
        table['b', [2, 7]] = -888
        expect[[2, 7], 1] = -888
        assert_equal(table.values, expect)

        with pytest.raises(KeyError):
            table[('c',)] = np.arange(10, 20)
        with pytest.raises(IndexError):
            table[(11,)] = np.arange(10, 20)
        with pytest.raises(KeyError):
            table['a', 'c'] = None
        with pytest.raises(KeyError):
            table[2:5] = 2
        with pytest.raises(KeyError):
            table[1, 2, 3] = 3

    def test_table_indexof(self):
        db = self.db
        table = db['test']
        assert_equal(table.index_of({'a': 15}), 5)
        assert_equal(table.index_of({'a': 50}), [])
        assert_equal(table.index_of('a < 13'), [0, 1, 2])

    def test_table_delete_row(self):
        db = self.db
        table = db['test']
        assert_is_instance(table, SQLTable)

        table.delete_row(0)
        expect = np.transpose([np.arange(11, 20), np.arange(21, 30)])
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(table.values, expect)

        table.delete_row(-1)
        expect = np.transpose([np.arange(11, 19), np.arange(21, 29)])
        assert_equal(table.column_names, ['a', 'b'])
        assert_equal(table.values, expect)

        with pytest.raises(IndexError):
            table.delete_row(10)
        with pytest.raises(IndexError):
            table.delete_row(-11)

    def test_table_delete_rows_indexer_robustness(self):
        db = self.db
        table = db['test']

        # Test that the row index is updated correctly after deleting rows
        row = table[5]
        table.delete_row(4)
        assert_equal(row.index, 4)
        assert_equal(table[4].values, row.values)

    def test_table_delete_column(self):
        db = self.db
        table = db['test']
        assert_is_instance(table, SQLTable)

        table.delete_column('a')
        expect = np.transpose([np.arange(20, 30)])
        assert_equal(table.column_names, ['b'])
        assert_equal(table.values, expect)

        with pytest.raises(KeyError):
            table.delete_column('a')
        with pytest.raises(KeyError):
            table.delete_column('c')


class Test_SQLColumn:
    @property
    def db(self):
        db = SQLDatabase(':memory:')
        db.add_table('test')
        db.add_column('test', 'a', data=np.arange(10, 20))
        db.add_column('test', 'b', data=np.arange(20, 30))
        return db

    def test_column_copy_error(self):
        db = self.db
        col = db['test']['a']
        with pytest.raises(NotImplementedError, match='Cannot copy'):
            copy.copy(col)
        with pytest.raises(NotImplementedError, match='Cannot copy'):
            copy.deepcopy(col)

    def test_column_basic_properties(self):
        db = self.db
        table = db['test']
        column = table['a']

        assert_equal(column.name, 'a')
        assert_equal(column.table, 'test')
        assert_equal(column.values, np.arange(10, 20))

    def test_column_len(self):
        db = self.db
        table = db['test']
        column = table['a']
        assert_equal(len(column), 10)

    def test_column_repr(self):
        db = self.db
        table = db['test']
        column = table['a']
        assert_equal(repr(column), "SQLColumn a in table 'test' (10 rows)")

    def test_column_contains(self):
        db = self.db
        table = db['test']
        column = table['a']
        assert_true(15 in column)
        assert_false(25 in column)

    def test_column_iter(self):
        db = self.db
        table = db['test']
        column = table['a']

        v = 10
        for i in column:
            assert_equal(i, v)
            v += 1

    def test_column_getitem_int(self):
        db = self.db
        table = db['test']
        column = table['a']

        assert_equal(column[0], 10)
        assert_equal(column[-1], 19)

        with pytest.raises(IndexError):
            column[10]
        with pytest.raises(IndexError):
            column[-11]

    def test_column_getitem_list(self):
        db = self.db
        table = db['test']
        column = table['a']

        assert_equal(column[[0, 1]], [10, 11])
        assert_equal(column[[-2, -1]], [18, 19])

        with pytest.raises(IndexError):
            column[[10, 11]]
        with pytest.raises(IndexError):
            column[[-11, -12]]

    def test_column_getitem_slice(self):
        db = self.db
        table = db['test']
        column = table['a']

        assert_equal(column[:2], [10, 11])
        assert_equal(column[-2:], [18, 19])
        assert_equal(column[2:5], [12, 13, 14])
        assert_equal(column[::-1], [19, 18, 17, 16, 15, 14, 13, 12, 11, 10])

    def test_column_getitem_tuple(self):
        db = self.db
        table = db['test']
        column = table['a']
        with pytest.raises(IndexError):
            column[('a',)]
        with pytest.raises(IndexError):
            column[(1,)]
        with pytest.raises(IndexError):
            column[1, 2]

    def test_column_setitem_int(self):
        db = self.db
        table = db['test']
        column = table['a']

        column[0] = 5
        assert_equal(db.get_row('test', 0).values, [5, 20])

        column[-1] = -1
        assert_equal(db.get_row('test', -1).values, [-1, 29])

    def test_column_setitem_list_slice(self):
        db = self.db
        table = db['test']
        column = table['a']

        column[:] = -1
        assert_equal(db.get_column('test', 'a').values, [-1]*10)
        column[[2, 4]] = 2
        assert_equal(db.get_column('test', 'a').values, [-1, -1, 2, -1, 2,
                                                         -1, -1, -1, -1, -1])
    def test_column_setitem_invalid(self):
        db = self.db
        table = db['test']
        column = table['a']

        with pytest.raises(IndexError):
            column[10] = 10
        with pytest.raises(IndexError):
            column[-11] = 10
        with pytest.raises(IndexError):
            column[2, 4] = [10, 11]
