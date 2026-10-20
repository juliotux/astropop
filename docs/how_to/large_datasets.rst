Plan memory for many images
===========================

Choose storage dtype and optional arrays before loading data. A stack of
``N`` images with ``P`` pixels and item size ``b`` needs ``N*P*b`` bytes per
stored stack array. For 150 images of 1024 by 1024 pixels at float32, nominal
input storage alone is 600 MiB. Same-sized uncertainties add another 600 MiB;
one uint8 flag array per image adds 150 MiB. Caller-owned frames and the
combiner's copies can coexist. These estimates exclude objects, headers,
outputs, validity masks, sample counts, promoted scratch buffers and libraries.

``memory_limit`` budgets combination work tiles, **not peak process memory**.
At least one output pixel's samples must fit even for a smaller requested
budget. NumPy may promote temporary arithmetic while the selected dtype governs
stack/output storage. Integer outputs discard fractions on final conversion;
choose range and precision deliberately. See :doc:`image_combination`.

Pass FITS paths instead of retaining a second list of caller-owned image arrays.
Enable disk caching with a dedicated disposable directory::

    from astropop.config import AstropopConfig as conf
    from astropop.image import imcombine
    conf.FRAMEDATA_DISABLE_UNCERTAINTY = True  # before loading any frames
    # Set this only if your reduction does not require FrameData masks/flags.
    conf.FRAMEDATA_DISABLE_FLAGS = True
    combined = imcombine(paths, method='median', dtype='float32',
                         memory_limit=64 * 1024**2, use_disk_cache=True)

This is a recipe with application-supplied ``paths``, not an executed benchmark.
Memmapping relocates stored inputs to disk; the loader still reads arrays before
mapping them. It does not stream FITS pixels directly into a tile. Filesystem
page cache, disk space and I/O speed also matter. Use the automatic temporary
directory or a dedicated directory the combiner may delete, never a shared
working directory. Caller-owned FrameData storage remains your responsibility.

Iterate over science products and write each result before processing the next.
For measurements, record input count, image shape, dtype, optional-array policy,
method/rejection settings, cache mode, memory budget, software versions and
hardware. Benchmark in a fresh process and report both elapsed time and peak
resident memory, distinguishing baseline interpreter/import cost from the
increment during combination. Sampling RSS can miss short peaks and includes
mapped pages; allocator tracing alone may omit native allocations. Keep those
measurements separate from numerical correctness checks.

See :doc:`../concepts/configuration` for restoring global settings and configuring
spawned workers; :doc:`../data_structures/cache` describes cache ownership.
