Performance measurements
------------------------

Iterate over science products and write each result before processing the next.
For measurements, record input count, image shape, dtype, optional-array policy,
method/rejection settings, cache mode, memory budget, software versions and
hardware. Benchmark in a fresh process and report both elapsed time and peak
resident memory, distinguishing baseline interpreter/import cost from the
increment during combination. Sampling RSS can miss short peaks and includes
mapped pages; allocator tracing alone may omit native allocations. Keep those
measurements separate from numerical correctness checks.

See :doc:`../modules/utilities/configuration` for restoring global settings and configuring
spawned workers; :doc:`../modules/utilities/cache` describes cache ownership.
