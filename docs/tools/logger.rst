.. include:: ../references.txt

Logging System
==============


Astropop has its own logger, with special abillities, by default.

It hierits Python's default logger module (not Astropy's one), due to bugs found in Astropy. It allows multiple childres, like for pipeline products, and have the special abillity to log to a list, for after use.

Using logger are far recommended over simple `print` funtions, due to level filtering, storing, properly displaying, etc. The log levels are:

=============  =======  =======================================================
``DEBUG``       ``10``  Diagnostic informations. Very verbose level.
-------------  -------  -------------------------------------------------------
``INFO``        ``20``  Important diagnostic informations. Low verbosity level.
-------------  -------  -------------------------------------------------------
``WARNING``     ``30``  Something is possibly wrong, but not a properly error.
-------------  -------  -------------------------------------------------------
``ERROR``       ``40``  Error.
-------------  -------  -------------------------------------------------------
``CRITICAL``    ``50``  Very serious error.
=============  =======  =======================================================

The function `~astropop.logger.resolve_level_string` can be used to convert a string to an integer log level.

Using Logger
------------

To use the logger, just import it and set a log level:

.. ipython:: python

    from astropop.logger import logger
    logger.setLevel('WARN')

You are now read to print simple logging:

.. ipython:: python

    logger.error('Matrix error. Agents needeed.')

The general behavior of Astropop logger is the same of Python default logger, very well documentated in `Python Log docs <https://docs.python.org/3/library/logging.html>`_.

.. TODO:: log_to_list doc

.. automodapi:: astropop.logger
    :no-inheritance-diagram:
