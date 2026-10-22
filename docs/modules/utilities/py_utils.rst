.. include:: ../../references.txt

Python utilities
================

:mod:`astropop.py_utils` contains lightweight conversion, iteration, template,
and subprocess helpers. They do not depend on FrameData or reduction settings.

Strings and scalar numbers
--------------------------

``string_fix`` decodes bytes using ``encode='utf-8'`` by default, preserves
strings, and applies ``str`` to other objects. ``check_number`` accepts real
scalar numbers, excluding booleans, complex numbers, and arrays::

    >>> from astropop.py_utils import string_fix, check_number

    >>> string_fix(b'object name')
    'object name'
    >>> string_fix(42)
    '42'
    >>> check_number(1.5), check_number(True), check_number([1.5])
    (True, False, False)

Iterating with scalar defaults
------------------------------

``broadcast`` repeats scalars and ``None`` alongside iterable inputs of equal
length. Unlike NumPy broadcasting, it compares iterable lengths rather than
multidimensional shapes. With only scalars it yields one tuple; unequal
iterable lengths raise ``ValueError``::

    >>> from astropop.py_utils import broadcast

    >>> list(broadcast([1, 2, 3], 5, None))
    [(1, 5, None), (2, 5, None), (3, 5, None)]

Dictionary templates
--------------------

``batch_key_replace`` substitutes ``{key}`` references using other entries in
the same dictionary, including string entries in lists. It modifies the
input in place and returns ``None``. References can be nested, but must not
form a cycle::

    >>> from astropop.py_utils import batch_key_replace

    >>> settings = {'root': '/data', 'file': '{root}/image.fits'}

    >>> # Substitutions update the dictionary in place, including references to other keys.
    >>> batch_key_replace(settings)

    >>> settings['file']
    '/data/image.fits'

Running an external command
---------------------------

``run_command`` runs a command, logs stdout/stderr lines, and returns a tuple
``(completed_process, stdout_lines, stderr_lines)``. Prefer an argument list
so paths containing spaces remain single arguments::

    >>> import sys
    >>> from astropop.py_utils import run_command

    >>> completed, stdout, stderr = run_command([sys.executable, '-c', 'print(42)'])

    >>> completed.returncode, stdout, stderr
    (0, ['42'], [])

Supplied ``stdout`` and ``stderr`` lists are appended to. The corresponding
``CompletedProcess`` attributes are newline-joined strings. Set
``stdout_loglevel`` and ``stderr_loglevel`` to configure logging (defaults:
``'DEBUG'`` and ``'ERROR'``), or provide a custom ``logger``. A nonzero return
code raises :class:`subprocess.CalledProcessError` with captured output.

The command is synchronous to the caller, including inside Jupyter. Additional
keyword arguments are forwarded to asyncio's process launcher: execution uses
an argument list on Windows and a quoted command on other platforms. Shell
pipes and redirections are not interpreted as a user-supplied shell script.




Reference/API
-------------

.. automodapi:: astropop.py_utils
    :no-inheritance-diagram:


.. currentmodule:: None
