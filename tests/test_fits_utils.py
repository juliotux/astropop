# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
from astropop.fits_utils import string_to_header_key
from astropop.testing import assert_equal


class Test_Fits_Utils():
    @pytest.mark.parametrize('string, expect', [('aaa', 'aaa'),
                                                ('AAA', 'aaa'),
                                                ('AaA', 'aaa'),
                                                ('AaAaAaAa', 'aaaaaaaa'),
                                                ('A1', 'a1'),
                                                ('A1A1A1A1', 'a1a1a1a1'),
                                                ('A1A1A1A1A1', 'A1A1A1A1A1'),
                                                ('with spa', 'with spa'),
                                                ('with space', 'with space'),
                                                ('With Space', 'With Space'),
                                                ('With-', 'with-'), # dash allowed
                                                ('With_', 'with_'), # underscore allowed
                                                ('With.', 'With.'), # dot not allowed
                                                ('HIERARCH Remove', 'Remove')])
    def test_string_to_header_key(self, string, expect):
        res = string_to_header_key(string)
        assert_equal(res, expect)
