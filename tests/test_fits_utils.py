# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
from astropop.fits_utils import string_to_header_key
from astropop.testing import assert_equal


class Test_Fits_Utils():
    @pytest.mark.parametrize('string, expect', [('aaa', 'AAA'),
                                                ('AAA', 'AAA'),
                                                ('AaA', 'AAA'),
                                                ('AaAaAaAa', 'AAAAAAAA'),
                                                ('A1', 'A1'),
                                                ('A1a1A1a1', 'A1A1A1A1'),
                                                ('A1a1A1a1A1', 'A1a1A1a1A1'),
                                                ('with spa', 'with spa'),
                                                ('with space', 'with space'),
                                                ('With Space', 'With Space'),
                                                ('With-', 'WITH-'),  # dash allowed
                                                ('With_', 'WITH_'),  # underscore allowed
                                                ('With.', 'With.'),  # dot not allowed
                                                ('HIERARCH Remove', 'Remove')])
    def test_string_to_header_key(self, string, expect):
        res = string_to_header_key(string)
        assert_equal(res, expect)
