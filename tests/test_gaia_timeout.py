# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Regression checks for the Gaia query connection timeout."""

import copy
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from astropop.catalogs.gaia import Gaia, _gaia_query_timeout
from astropop.catalogs._online_tools import astroquery_query


def test_timeout_is_local_to_copied_gaia_client(monkeypatch):
    client = copy.deepcopy(Gaia)
    handler = client._TapPlus__getconnhandler()._TapConn__connectionHandler
    original = handler.get_connection
    global_timeout = socket.getdefaulttimeout()
    global_handler = Gaia._TapPlus__getconnhandler()._TapConn__connectionHandler
    global_connection = global_handler.get_connection(ishttps=True)
    original_connection_timeout = global_connection.timeout
    global_connection.close()
    now = [10.]
    monkeypatch.setattr('astropop.catalogs.gaia.time.monotonic', lambda: now[0])
    with _gaia_query_timeout(client, timeout=5):
        connection = handler.get_connection(ishttps=True)
        assert connection.timeout == 5
        now[0] = 14.
        assert handler.get_connection(ishttps=True).timeout == 1
        assert socket.getdefaulttimeout() == global_timeout
        global_connection = global_handler.get_connection(ishttps=True)
        assert global_connection.timeout is original_connection_timeout
        global_connection.close()
        now[0] = 15.
        with pytest.raises(TimeoutError, match='Gaia query'):
            handler.get_connection(ishttps=True)
    assert handler.get_connection == original


def test_expired_retry_cannot_create_another_connection(monkeypatch):
    now = [0.]
    monkeypatch.setattr('astropop.catalogs.gaia.time.monotonic', lambda: now[0])
    connection = Mock()
    factory = Mock(return_value=connection)
    handler = SimpleNamespace(get_connection=factory)
    client = Mock()
    client._TapPlus__getconnhandler.return_value = SimpleNamespace(
        _TapConn__connectionHandler=handler)

    def stalled_query():
        handler.get_connection(ishttps=True)
        now[0] = 5.
        raise TimeoutError('stalled socket')

    with pytest.raises(TimeoutError):
        with _gaia_query_timeout(client, timeout=5):
            astroquery_query(stalled_query)
    factory.assert_called_once_with(ishttps=True)
    connection.close.assert_called_once()
    assert handler.get_connection is factory
