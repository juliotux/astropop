.. include:: ../../references.txt

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

    >>> print('INFO level:', resolve_level_string('INFO'))
    INFO level: 20

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
    ...
    ...     # Detach this capture so later application messages do not keep filling the list.
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

The ``ListHandler`` class
-------------------------

Construct ``ListHandler(log_list=None, full_record=False)`` directly when an
application needs to configure a handler before attaching it. The default
stores each record's ``getMessage()`` text; ``full_record=True`` retains the
LogRecord, including its logger name, severity and formatted message. The
``log_list`` and ``full_record`` properties expose these settings.

Attach the handler to the logger and remove it when finished::

    >>> from astropop.logger import ListHandler

    >>> records = ListHandler(full_record=True)
    >>> previous = logger.level
    >>> logger.setLevel('INFO')
    >>> logger.addHandler(records)
    >>> try:
    ...     logger.info('Combined %d frames', 3)
    ... finally:
    ...
    ...     # Remove the handler even if processing raises an exception.
    ...     logger.removeHandler(records)
    ...     logger.setLevel(previous)

    >>> record = records.log_list[0]

    >>> print(record.name, record.levelname, record.getMessage())
    astropop INFO Combined 3 frames

``emit`` appends an already-created LogRecord; normal application code calls
``logger.info`` or another severity method so logger/handler filtering occurs
before emission. A formatter assigned to the handler does not change the
stored ``getMessage()`` text. Format retained records explicitly if a timestamp
or another presentation is needed.




Reference/API
-------------

.. automodapi:: astropop.logger
    :no-inheritance-diagram:

.. currentmodule:: None

.. py:data:: astropop.logger.logger

    Shared standard-library Logger named ``astropop``. Its initial level is
    ``INFO``; changing its level affects subsequent library log messages.


.. currentmodule:: None
