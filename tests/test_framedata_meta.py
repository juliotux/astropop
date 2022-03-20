# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropop.framedata._meta import FrameMeta
from astropop.testing import assert_equal, assert_is_instance, assert_true, \
                             assert_in, assert_not_in
import pytest


class Test_FrameMeta():
    def test_framemeta_create_othertypekeys(self):
        a = FrameMeta({'TesT': 'AaA'})
        for k in [1, 3.1415, b'\x00']:
            with pytest.raises(TypeError, match='Only string keys accepted.'):
                a[k] = 1
        assert_true(a, {'test': 'AaA'})

    def test_framemeta_create_nonascii(self):
        a = FrameMeta({'😀': 1, 'Ç': 'cedilha', 'TesT': 'AaA'})
        assert_equal(a['😀'], 1)
        assert_equal(a['ç'], 'cedilha')
        assert_equal(a['test'], 'AaA')

    def test_framemeta_pythonerrors(self):
        with pytest.raises(TypeError,
                           match='dict expected at most 1 argument'):
            FrameMeta([('a', 1), ('B', 2)], {'TesT': 'AAA'})

        with pytest.raises(TypeError,
                           match='cannot convert dictionary update *'):
            FrameMeta([1, 2, 3])

        with pytest.raises(TypeError,
                           match='cannot convert dictionary update *'):
            FrameMeta({1, 2, 3})

    def test_framemeta_update_error(self):
        a = FrameMeta({'A': 1, 'B': 2, 'c': 3})
        with pytest.raises(TypeError,
                           match='cannot convert dictionary update'):
            a.update([1, 2, 3])

    def test_framemeta_item_assignment_tuple(self):
        a = FrameMeta()
        a['test1'] = 1
        a['test2'] = ((1))
        a['test3'] = (1, 'with comment')
        with pytest.raises(ValueError, match='Tuple items should be'):
            a['test4'] = (1, 2, 3)

        assert_equal(list(a.values()), [1, 1, 1])
        assert_equal(list(a.comments()), ['', '', 'with comment'])

    def test_framemeta_reassign_comment(self):
        a = FrameMeta()
        a['a'] = (1, 'first comment')
        a['a'] = (2, 'second comment')
        assert_equal(list(a.comments()), ['second comment'])

    def test_framemeta_create_empty(self):
        a = FrameMeta()
        assert_is_instance(a, FrameMeta)
        assert_equal(len(a), 0)

    def test_framemeta_create_full(self):
        a = FrameMeta(a=1, b=2, c=None, d=[1, 2, 3])
        assert_is_instance(a, FrameMeta)
        assert_equal(a['a'], 1)
        assert_equal(a['b'], 2)
        assert_equal(a['c'], None)
        assert_equal(a['d'], [1, 2, 3])

    def test_framemeta_create_full_upper(self):
        a = FrameMeta(A=1, B=2, C=None, D=[1, 2, 3])
        assert_is_instance(a, FrameMeta)
        assert_equal(a['a'], 1)
        assert_equal(a['b'], 2)
        assert_equal(a['c'], None)
        assert_equal(a['d'], [1, 2, 3])

    def test_framemeta_create_iterable(self):
        a = FrameMeta([('A', 1), ('b', 'test')])
        assert_is_instance(a, FrameMeta)
        assert_equal(a['a'], 1)
        assert_equal(a['b'], 'test')

    def test_framemeta_create_dict(self):
        a = FrameMeta({'A': 1, 'b': 'test', 'TesT': 'AaA'})
        assert_is_instance(a, FrameMeta)
        assert_equal(a['a'], 1)
        assert_equal(a['b'], 'test')
        assert_equal(a['test'], 'AaA')

    def test_framemeta_equal(self):
        a = FrameMeta({'A': 1, 'b': 2})
        b = FrameMeta({'a': 1, 'B': 2})
        assert_true(a == b)

    def test_framemeta_equal_dict(self):
        a = FrameMeta({'A': 1, 'b': 2})
        b = {'a': 1, 'B': 2}
        assert_true(a == b)
        assert_true(b == a)

    def test_framemeta_get(self):
        a = FrameMeta({'A': 1, 'b': 'test', 'TesT': 'AaA'})
        res = a.get('test')
        assert_equal(res, 'AaA')
        assert_in('test', a)
        assert_in('TEST', a)

    def test_framemeta_pop(self):
        a = FrameMeta({'A': 1, 'b': 'test', 'TesT': 'AaA'})
        res = a.pop('test')
        assert_equal(res, 'AaA')
        assert_not_in('test', a.keys())
        assert_not_in('TesT', a.keys())

    def test_framemeta_getitem(self):
        a = FrameMeta({'A': 1, 'b': 'test', 'TesT': 'AaA'})
        assert_equal(a['A'], 1)
        assert_equal(a['a'], 1)
        assert_equal(a['B'], 'test')
        assert_equal(a['b'], 'test')
        assert_equal(a['test'], 'AaA')
        assert_equal(a['TEST'], 'AaA')
        assert_equal(a['tESt'], 'AaA')
        assert_equal(a['TesT'], 'AaA')

    def test_framemeta_setitem(self):
        a = FrameMeta({'A': 1, 'b': 'test', 'TesT': 'AaA'})
        a['a'] = 2
        a['B'] = 3
        a['TEST'] = None
        assert_equal(a['A'], 2)
        assert_equal(a['a'], 2)
        assert_equal(a['B'], 3)
        assert_equal(a['b'], 3)
        assert_equal(a['test'], None)
        assert_equal(a['TEST'], None)
        assert_equal(a['tESt'], None)
        assert_equal(a['TesT'], None)
        assert_equal(len(a), 3)

    def test_framemeta_deltitem(self):
        a = FrameMeta({'A': 1, 'b': 2, 'TesT': 'AaA'})
        del a['test']
        assert_true(a == {'a': 1, 'b': 2})

    def test_framemeta_update(self):
        a = FrameMeta({'A': 1, 'B': 2, 'c': 3})
        a.update({'a': 4})
        assert_equal(a['A'], 4)
        assert_equal(a['a'], 4)
        assert_equal(a['B'], 2)
        assert_equal(a['b'], 2)
        assert_equal(a['C'], 3)
        assert_equal(a['c'], 3)

    def test_framemeta_get(self):
        a = FrameMeta({'A': 1, 'B': 2, 'c': 3})
        assert_equal(a.get('a'), 1)
        assert_equal(a.get('a', 2), 1)
        assert_equal(a.get('d'), None)
        assert_equal(a.get('d', 2), 2)

    def test_framemeta_pop(self):
        a = FrameMeta({'A': 1, 'B': 2, 'c': 3})
        assert_equal(a.pop('a'), 1)
        assert_not_in('a', a)
        assert_equal(a.pop('a'), None)

    def test_framemeta_popitem(self):
        a = FrameMeta({'A': 1, 'B': 2, 'c': 3})
        assert_equal(a.popitem('a'), ('a', 1))
        assert_not_in('a', a)
        assert_equal(a.popitem('a'), ('a', None))

    def test_framememta_repr(self):
        a = FrameMeta({'A': 1, 'B': (2, 'with comment'), 'c': 3})
        expect = "FrameMeta:\n"
        expect += "a = 1  # \n"
        expect += "b = 2  # with comment\n"
        expect += "c = 3  # \n"

        assert_equal(repr(a), expect)

    def test_framemeta_index(self):
        a = FrameMeta({'A': 1, 'B': (2, 'with comment'), 'c': 3})
        assert_equal(a.index('a'), 0)
        assert_equal(a.index('b'), 1)
        assert_equal(a.index('c'), 2)
        with pytest.raises(KeyError):
            a.index('d')

    def test_framemeta_len(self):
        a = FrameMeta({'A': 1, 'B': (2, 'with comment'), 'c': 3})
        assert_equal(len(a), 3)

    def test_framemeta_del_remove(self):
        a = FrameMeta({'A': 1, 'B': (2, 'with comment'), 'c': 3})
        del a['a']
        a.remove('b')
        assert_equal(a, {'c': 3})

    def test_framemeta_add(self):
        a = FrameMeta({'A': 1, 'B': (2, 'with comment'), 'c': 3})
        a.add('D', 4)
        assert_equal(a, {'a': 1, 'b': 2, 'c': 3, 'd': 4})
