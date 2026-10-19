.. include:: ../references.txt

Logging system
==============

Astropop uses the standard-library ``logging`` package. Its named logger,
``astropop``, defaults to level ``INFO``; ordinary Python handlers, formatters,
and child loggers can be used to integrate it with an application.

Choosing a level
----------------

Levels are ``DEBUG`` (10), ``INFO`` (20), ``WARNING`` (30), ``ERROR`` (40), and
``CRITICAL`` (50). Messages below the logger's effective level are discarded.
Set the level on the shared logger and restore it after temporary changes::

    >>> import logging
    >>> from astropop.logger import logger, resolve_level_string
    >>> previous = logger.level
    >>> logger.setLevel('WARNING')
    >>> resolve_level_string('INFO') == logging.INFO
    True
    >>> logger.setLevel(previous)

``resolve_level_string`` accepts an integer, a numeric string, or an uppercase
logging level name. Invalid names are not silently replaced by a default.

Capturing messages in a list
----------------------------

``log_to_list`` attaches a ``ListHandler`` and returns it. By default, the list
contains formatted message strings, without timestamps or level prefixes.
Always remove the handler when finished so later messages do not accumulate::

    >>> from astropop.logger import log_to_list
    >>> messages = []
    >>> previous = logger.level
    >>> logger.setLevel('INFO')
    >>> handler = log_to_list(logger, messages)
    >>> try:
    ...     logger.info('Processing image %d', 3)
    ... finally:
    ...     logger.removeHandler(handler)
    ...     logger.setLevel(previous)
    >>> messages
    ['Processing image 3']

Omit the list argument to create a new list accessible as ``handler.log_list``.
Use ``full_record=True`` to store ``logging.LogRecord`` objects instead of
strings; their ``levelname``, ``name``, and ``getMessage()`` provide structured
information. The handler captures only records that pass logging's level and
filter checks. List storage has no size limit or additional thread-safety
guarantees; applications control its lifetime.

Logging API
-----------

.. automodapi:: astropop.logger
    :no-inheritance-diagram:

.. currentmodule:: None

.. py:data:: astropop.logger.logger

    Shared standard-library Logger named ``astropop``. Its initial level is
    ``INFO``; changing its level affects subsequent library log messages.
