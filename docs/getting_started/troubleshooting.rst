Troubleshoot a reduction
========================

Start with the smallest failing frame or source list. Record software versions,
configuration, input shape/dtype/unit and the exact exception. Separate a
scientific limitation from a reproducible implementation error; do not loosen
numerical tolerances just to make a failing check pass.

.. list-table:: Symptoms, checks and recovery
    :header-rows: 1
    :widths: 24 40 36

    * - Symptom
      - Minimal diagnostic
      - Recovery and contract
    * - Notebook behaves differently from a terminal
      - Print ``sys.executable``, ``astropop.__file__`` and ``astropop.__version__``.
      - Select the kernel in the intended environment, restart it, rerun from
        the beginning. See :doc:`installation`.
    * - Astrometry solver cannot start or solve
      - Check the ``solve-field`` executable, index paths, scale/position hints
        and solver output.
      - Install the external program/indexes and choose applicable hints;
        pip does not install them. See :doc:`../modules/astrometry/index`.

When reporting a suspected bug, include a tiny reproducible input or synthetic
example, expected versus actual result, version information and full traceback.
State whether external services, local indexes or instrument-specific data are
required. See :doc:`../development/testing` for reproducing offline versus live checks.
