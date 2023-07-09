# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

from astropop.testing import *
from .test_framedata import create_framedata


class Test_FrameData_History:
    hist = ['once upon a time', 'a small fits file',
            'got read', 'by astropop',
            'and stay in', 'computer memory.']

    def test_framedata_set_history_string(self):
        frame = create_framedata()
        for i in self.hist:
            frame.history = i
        assert_equal(len(frame.history), 6)
        assert_equal(frame.history, self.hist)

    def test_framedata_set_history_none(self):
        frame = create_framedata()
        frame.history = None
        assert_equal(frame.history, [])

    def test_framedata_set_history_list(self):
        frame = create_framedata()
        frame.history = list(self.hist)
        assert_equal(len(frame.history), 6)
        assert_equal(frame.history, self.hist)

    def test_framedata_set_history_tuple(self):
        frame = create_framedata()
        frame.history = tuple(self.hist)
        assert_equal(len(frame.history), 6)
        assert_equal(frame.history, self.hist)


class Test_FrameData_Comment:
    comm = ['this is a test', 'to make commenst in astropop', 'that can',
            'be lists', 'or also', 'tuples.']

    def test_framedata_set_comment_string(self):
        frame = create_framedata()
        for i in self.comm:
            frame.comment = i

        assert_equal(len(frame.comment), 6)
        assert_equal(frame.comment, self.comm)

    def test_frame_set_comment_none(self):
        frame = create_framedata()
        frame.comment = None
        assert_equal(frame.comment, [])

    def test_frame_set_comment_list(self):
        frame = create_framedata()
        frame.comment = list(self.comm)
        assert_equal(len(frame.comment), 6)
        assert_equal(frame.comment, self.comm)

    def test_frame_set_comment_tuple(self):
        frame = create_framedata()
        frame.comment = tuple(self.comm)
        assert_equal(len(frame.comment), 6)
        assert_equal(frame.comment, self.comm)
