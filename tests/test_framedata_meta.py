# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

from astropop.testing import *
from .test_framedata import create_framedata


class Test_FrameData_History:
    def test_framedata_set_history(self):
        frame = create_framedata()
        frame.history = 'once upon a time'
        frame.history = 'a small fits file'
        frame.history = ['got read', 'by astropop']
        frame.history = ('and stay in', 'computer memory.')

        assert_equal(len(frame.history), 6)
        assert_equal(frame.history[0], 'once upon a time')
        assert_equal(frame.history[1], 'a small fits file')
        assert_equal(frame.history[2], 'got read')
        assert_equal(frame.history[3], 'by astropop')
        assert_equal(frame.history[4], 'and stay in')
        assert_equal(frame.history[5],  'computer memory.')


class Test_FrameData_Comment:
    def test_framedata_set_comment(self):
        frame = create_framedata()
        frame.comment = 'this is a test'
        frame.comment = 'to make commenst in astropop'
        frame.comment = ['that can', 'be lists']
        frame.comment = ('or also', 'tuples.')

        assert_equal(len(frame.comment), 6)
        assert_equal(frame.comment[0], 'this is a test')
        assert_equal(frame.comment[1], 'to make commenst in astropop')
        assert_equal(frame.comment[2], 'that can')
        assert_equal(frame.comment[3], 'be lists')
        assert_equal(frame.comment[4], 'or also')
        assert_equal(frame.comment[5],  'tuples.')
