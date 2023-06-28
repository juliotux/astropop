# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
from enum import Flag
import numpy as np

from astropop.flags import mask_from_flags
from astropop.testing import *


class CustomFlags(Flag):
    A = 1
    B = 2
    C = 4
    D = 8
    E = 16


class OtherFlags(Flag):
    A = 1
    B = 2
    C = 4
    D = 8
    E = 16


class TestMaskFromFlag:
    @pytest.mark.parametrize('flags', [None, 1, 1.0, 'a', [1, 2, 3],
                                       [1, 2.0, 3], [1, 'a', 3]])
    def test_non_flag(self, flags):
        data = np.arange(10)
        with pytest.raises(TypeError):
            mask_from_flags(data, flags)

    def test_not_allowed_flags(self):
        data = np.arange(10)
        with pytest.raises(TypeError):
            mask_from_flags(data, [CustomFlags.A, OtherFlags.B],
                            allowed_flags_class=CustomFlags)

    def test_mask_single_value(self):
        assert_true(mask_from_flags(1, CustomFlags.A))
        assert_true(mask_from_flags(2, CustomFlags.B))
        assert_true(mask_from_flags(4, CustomFlags.C))
        assert_true(mask_from_flags(8, CustomFlags.D))
        assert_true(mask_from_flags(16, CustomFlags.E))
        assert_false(mask_from_flags(1, CustomFlags.B))
        assert_false(mask_from_flags(2, CustomFlags.A))
        assert_false(mask_from_flags(4, CustomFlags.D))
        assert_false(mask_from_flags(8, CustomFlags.C))

    def test_mask_1d(self):
        data = np.arange(10)
        mask = mask_from_flags(data,
                               flags_to_mask=[CustomFlags.A, CustomFlags.B])
        # masked = 1, 2, 3, 5, 6, 7, 9
        assert_equal(mask, [False, True, True, True, False, True, True,
                            True, False, True])

    def test_mask_2d(self):
        data = np.arange(10).reshape(2, 5)
        mask = mask_from_flags(data,
                               flags_to_mask=[CustomFlags.A, CustomFlags.B])
        # masked = 1, 2, 3, 5, 6, 7, 9
        assert_equal(mask, [[False, True, True, True, False],
                            [True, True, True, False, True]])
