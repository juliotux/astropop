.. include:: ../references.txt

Astrometry
==========

Use these entries for signatures, parameter choices and return contracts.
For workflows and scientific assumptions, read the linked guides.

Guide: :doc:`../reduction/astrometry`.

.. automodapi:: astropop.astrometry
    :no-inheritance-diagram:

External command interface
--------------------------

The convenience solver functions use ``SolveFieldCommand`` to locate and invoke
Astrometry.net. Supply a custom command path when ``solve-field`` is not on
``PATH``. Constructing this wrapper checks the external executable; it is not
required for using image arithmetic or a manually constructed WCS.

.. autoclass:: astropop.astrometry.astrometrynet.SolveFieldCommand
    :members:
