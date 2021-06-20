# Licensed under a 3-clause BSD style license - see LICENSE.rst

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
    log_list : list or None, optional
        List to store the logs. If None, a new list will be created and
        can be accesses with `ListHandler.log_list`
    full_record : bool, optional
        Store the full Python log record instead of just the message.
    """
    def __init__(self, log_list=None, full_record=False):
        logging.Handler.__init__(self)
        if log_list is None:
            log_list = []
        self._log_list = log_list
        self._full_record = full_record

    @property
    def log_list(self):
        """List where records are being stored."""
        return self._log_list

    @property
    def full_record(self):
        """True if full records are being stored in list. False if only
        messages."""
        return self._full_record

    def emit(self, record):
        """Append the log record to the list."""
        if self.full_record:
            self.log_list.append(record)
        else:
            self.log_list.append(record.getMessage())


def log_to_list(logger, log_list=None, full_record=False):
    """Add a ListHandler and a log_list to a Logger.

    Parameters
    ----------
    logger : `logging.Logger`
        `Logger` instance where handler will be added.
    log_list : list or None, optional
        List to store log records.
    full_record : bool, optional
        Store full log records instead of just the message.

    Returns
    -------
        `ListHandler` created during the process.
    """
    lh = ListHandler(log_list, full_record)
    logger.addHandler(lh)
    return lh


def resolve_level_string(value):
    """Resolve the log level of a string."""
    try:
        # Handle already int values
        value = int(value)
        return value
    except ValueError:
        return getattr(logging, value)
