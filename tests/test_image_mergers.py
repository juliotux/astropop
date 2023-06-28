# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from astropop.image._tools import merge_header, merge_flag
from astropop.testing import *


class TestMergeHeaders:
    def create_headers(self):
        h1 = {
            'A': 1,
            'B': 'AAA',
            'C': 3.1415,
            'D': True,
            'E': None
        }

        h2 = h1.copy()
        h2['B'] = 'BBB'

        h3 = h1.copy()
        h3['C'] = 2.7182

        h4 = h1.copy()
        h4['D'] = False

        return [h1, h2, h3, h4]

    def test_merge_headers_only_equal(self):
        headers = self.create_headers()

        # For all headers, only A and E are the same
        merged = merge_header(*headers, method='only_equal')
        assert_is_instance(merged, dict)
        assert_equal(merged, {'A': 1, 'E': None})

        # for the first 2 headers, A, C, D and E are the same
        merged = merge_header(*headers[:2], method='only_equal')
        assert_is_instance(merged, dict)
        assert_equal(merged, {'A': 1, 'C': 3.1415, 'D': True, 'E': None})

    def test_merge_headers_first(self):
        headers = self.create_headers()

        merged = merge_header(*headers, method='first')
        assert_is_instance(merged, dict)
        assert_equal(merged, headers[0])
        # ensure copy
        assert_is_not(merged, headers[0])

    def test_merge_headers_selected_keys(self):
        headers = self.create_headers()

        merged = merge_header(*headers, method='selected_keys',
                              selected_keys=['A', 'C'])
        assert_is_instance(merged, dict)
        # Default is to use the first
        assert_equal(merged, {'A': 1, 'C': 3.1415})

    def test_merge_headers_no_merge(self):
        headers = self.create_headers()

        merged = merge_header(*headers, method='no_merge')
        assert_is_instance(merged, dict)
        assert_equal(merged, {})

    def test_merge_headers_invalid_method(self):
        headers = self.create_headers()

        with pytest.raises(ValueError, match='Unknown method'):
            merge_header(*headers, method='invalid')

        with pytest.raises(ValueError,
                           match='selected_keys must be provided if'):
            merge_header(*headers, method='selected_keys')

    def test_merge_headers_different_keys(self):
        headers = self.create_headers()
        headers[2]['F'] = 1
        merged = merge_header(*headers, method='only_equal')
        assert_not_in('F', merged)


class TestMergeFlags:
    def get_flags_4x4(self):
        f1 = np.zeros((4, 4), dtype=np.uint8)
        f1[0, 0] = 1
        f1[1, 1] = 1
        f1[2, 2] = 1

        f2 = np.zeros((4, 4), dtype=np.uint8)
        f2[0, 0] = 1
        f2[1, 1] = 1
        f2[3, 3] = 1

        f3 = np.zeros((4, 4), dtype=np.uint8)
        f3[0, 0] = 1
        f3[2, 2] = 1

        return f1, f2, f3

    def get_flags_4x4_multiple(self):
        f1 = np.zeros((4, 4), dtype=np.uint8)
        f1[0, 0] = 1
        f1[1, 1] = 5
        f1[2, 2] = 2

        f2 = np.zeros((4, 4), dtype=np.uint8)
        f2[0, 0] = 2
        f2[1, 1] = 4
        f2[3, 3] = 2

        f3 = np.zeros((4, 4), dtype=np.uint8)
        f3[0, 0] = 1
        f3[1, 1] = 7

        return f1, f2, f3

    def test_merge_flags_and(self):
        fs = self.get_flags_4x4()
        merged = merge_flag(*fs, method='and')
        expect = np.zeros((4, 4), dtype=np.uint8)
        # only 0,0 is equal in all
        expect[0, 0] = 1
        assert_equal(merged, expect)
        assert_equal(merged.dtype.kind, 'u')

    def test_merge_flags_or(self):
        fs = self.get_flags_4x4()
        merged = merge_flag(*fs, method='or')
        expect = np.zeros((4, 4), dtype=np.uint8)
        # 0,0, 1,1 and 2,2 will be present
        expect[0, 0] = 1
        expect[1, 1] = 1
        expect[2, 2] = 1
        expect[3, 3] = 1
        assert_equal(merged, expect)
        assert_equal(merged.dtype.kind, 'u')

    def test_merge_flags_and_2(self):
        fs = self.get_flags_4x4()
        merged = merge_flag(*fs[:2], method='and')
        expect = np.zeros((4, 4), dtype=np.uint8)
        # only 0,0 and 1,1 are equal
        expect[0, 0] = 1
        expect[1, 1] = 1
        assert_equal(merged, expect)
        assert_equal(merged.dtype.kind, 'u')

    def test_merge_flags_or_2(self):
        fs = self.get_flags_4x4()
        merged = merge_flag(*fs[:2], method='or')
        expect = np.zeros((4, 4), dtype=np.uint8)
        expect[0, 0] = 1
        expect[1, 1] = 1
        expect[2, 2] = 1
        expect[3, 3] = 1
        assert_equal(merged, expect)
        assert_equal(merged.dtype.kind, 'u')

    def test_merge_flags_no(self):
        fs = self.get_flags_4x4()
        merged = merge_flag(*fs, method='no_merge')
        expect = np.zeros((4, 4), dtype=np.uint8)
        assert_equal(merged, expect)
        assert_equal(merged.dtype.kind, 'u')

    def test_merge_flags_invalid_method(self):
        fs = self.get_flags_4x4()
        with assert_raises(ValueError):
            merge_flag(*fs, method='invalid')

    def test_merge_flags_and_multiple(self):
        fs = self.get_flags_4x4_multiple()
        merged = merge_flag(*fs, method='and')
        expect = np.zeros((4, 4), dtype=np.uint8)
        expect[1, 1] = 4
        assert_equal(merged, expect)
        assert_equal(merged.dtype.kind, 'u')

    def test_merge_flags_or_multiple(self):
        fs = self.get_flags_4x4_multiple()
        merged = merge_flag(*fs, method='or')
        expect = np.zeros((4, 4), dtype=np.uint8)
        expect[0, 0] = 3
        expect[1, 1] = 7
        expect[2, 2] = 2
        expect[3, 3] = 2
        assert_equal(merged, expect)
        assert_equal(merged.dtype.kind, 'u')

    def test_merge_flags_or_multiple_2(self):
        fs = self.get_flags_4x4_multiple()
        merged = merge_flag(*fs[:2], method='or')
        expect = np.zeros((4, 4), dtype=np.uint8)
        expect[0, 0] = 3
        expect[1, 1] = 5
        expect[2, 2] = 2
        expect[3, 3] = 2
        assert_equal(merged, expect)
        assert_equal(merged.dtype.kind, 'u')

    def test_merge_flags_and_multiple_2(self):
        fs = self.get_flags_4x4_multiple()
        merged = merge_flag(*fs[:2], method='and')
        expect = np.zeros((4, 4), dtype=np.uint8)
        expect[1, 1] = 4
        assert_equal(merged, expect)
        assert_equal(merged.dtype.kind, 'u')

    def test_merge_flags_no_multiple(self):
        fs = self.get_flags_4x4_multiple()
        merged = merge_flag(*fs, method='no_merge')
        expect = np.zeros((4, 4), dtype=np.uint8)
        assert_equal(merged, expect)
        assert_equal(merged.dtype.kind, 'u')
