.. include:: ../references.txt

MemMapArrays
============

|MemMapArray| is a data container, very likely |ndarray|, intended to offer an ease use of data chaching, also called |memmap| in NumPy. The big difference here is that our implementation make possible a dynamic enabling/disabling of data memmapping without change the container instance itself.

This class use |numpy| s implementation of |ndarray| and |memmap| as an intern contained data, and switch between them according user needs. It redirects all main operators and class properties to the NumPy contained data, transforming it to |MemMapArray| if the result is an array.

The class also supports physical units assignment, using |Unit| wrapping.

Creating an MemMapArray
-----------------------

|MemMapArray| are very simple to create, just like a `~numpy.array`. For example:

.. ipython:: python

    from astropop.framedata import MemMapArray
    m = MemMapArray([[0, 1], [2, 3]], filename='/tmp/mymemmaparray.npy', dtype='float32', memmap=True)
    print(m)


This will create a |MemMapArray| instance, accessible just like Numpy arrays:

.. ipython:: python

    m[0, 0]
    m[0, 1]
    m.dtype

However, this data is not in memory, but is cached in disk, in file ``./mymemmaparray.npy``.

|MemMapArray| also can create empty data instances, and handle it. You can just pass `None` data to it and it's ok.

.. ipython:: python

    m_empty = MemMapArray(None)
    print(m_empty)

Enabling/Disabling Disk Memmapping
----------------------------------

|MemMapArray| has a `~astropop.framedata.MemMapArray.memmap` property that tell us if the memmapping is enabled for the instance. This property is `True` if the memmapping is enabled.:

.. ipython:: python

    m.memmap

To enable or disable memmapping dynamically, you can use `~astropop.framedata.MemMapArray.enable_memmap` and `~astropop.framedata.MemMapArray.disable_memmap` methods. Each of them has its own arguments.

.. TODO:: `enable_memmap` and `disable_memmap` usage

Setting Properties
------------------


Math Operations
---------------


MemMapArray API
---------------

.. automodapi:: astropop.framedata.memmap
    :no-inheritance-diagram:
