# Licensed under a 3-clause BSD style license - see LICENSE.rst


"""Configure Astropop logging and capture messages in caller-owned lists."""
from __future__ import annotations

import typing as _t
import logging


__all__ = ['logger', 'ListHandler', 'log_to_list', 'resolve_level_string']


logging.basicConfig(format='%(asctime)-15s %(name)s - %(levelname)s -'
                           ' %(message)s  [%(module)s]')
logger = logging.getLogger('astropop')
logger.setLevel('INFO')


class ListHandler(logging.Handler):
    """Logging handler to save messages in a list. No thread safe!

    Parameters
    ----------
    log_list
        List to store the logs. If None, a new list will be created and
        can be accesses with `ListHandler.log_list`
    full_record
        Store the full Python log record instead of just the message.

    Examples
    --------
    >>> import logging
    >>> from astropop.logger import ListHandler
    >>> handler = ListHandler()
    >>> record = logging.LogRecord('example', logging.INFO, '', 0,
    ...                            'Processed %d frames', (3,), None)
    >>> _ = handler.handle(record)
    >>> print(handler.log_list)
    ['Processed 3 frames']
    """
    def __init__(
        self,
        log_list: list[_t.Any] | None=None,
        full_record: bool=False,
    ) -> None:
        """Retain a destination list or create one, choosing messages or full records."""
        logging.Handler.__init__(self)
        if log_list is None:
            log_list = []
        self._log_list = log_list
        self._full_record = full_record

    @property
    def log_list(self) -> list[str | logging.LogRecord]:
        """List where records are being stored."""
        return self._log_list

    @property
    def full_record(self) -> bool:
        """True if full records are being stored in list. False if only
        messages."""
        return self._full_record

    def emit(self, record: logging.LogRecord) -> None:
        """Append the log record to the list."""
        if self.full_record:
            self.log_list.append(record)
        else:
            self.log_list.append(record.getMessage())


def log_to_list(
    logger: logging.Logger,
    log_list: list[_t.Any] | None=None,
    full_record: bool=False,
) -> ListHandler:
    """Add a ListHandler and a log_list to a Logger.

    Parameters
    ----------
    logger
        `~logging.Logger` instance where handler will be added.
    log_list
        List to store log records.
    full_record
        Store full log records instead of just the message.

    Returns
    -------
        `ListHandler` created during the process.

    Examples
    --------
    >>> import logging
    >>> from astropop.logger import log_to_list
    >>> local_logger = logging.Logger('example', level=logging.INFO)
    >>> handler = log_to_list(local_logger)
    >>> try:
    ...     local_logger.info('Processed %d frames', 3)
    ...     print(handler.log_list)
    ... finally:
    ...     # Detach the capture so it does not retain later messages.
    ...     local_logger.removeHandler(handler)
    ['Processed 3 frames']
    """
    lh = ListHandler(log_list, full_record)
    logger.addHandler(lh)
    return lh


def resolve_level_string(value: str | int) -> int:
    """Resolve the log level of a string.

    Examples
    --------
    >>> from astropop.logger import resolve_level_string
    >>> print(resolve_level_string('INFO'), resolve_level_string('WARNING'))
    20 30
    """
    try:
        # Handle already int values
        value = int(value)
        return value
    except ValueError:
        return getattr(logging, value)
