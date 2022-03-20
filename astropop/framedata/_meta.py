# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Special class for store FrameData metadata."""

import copy


__all__ = ['FrameMeta']


class FrameMeta():
    """
    Dict-like container to store frames MetaData.

    It is intended to be memory efficient, but a bit slower then default
    python dict. The main differences between FrameMeta and python default
    are:
    - FrameMeta is case-insensitive, while Python default is case-sensitive;
    - FrameMeta only accept string keys, while Python default accept any
      immutable value.
    - FrameMeta can accept comments for fields, just like
      `~astropy.io.fits.Header`.
    """

    __keys = None
    __values = None
    __comms = None

    def __init__(self, *args, **kwargs):
        """Initialize a FrameMeta using the same method of a python dict."""
        self.clear()

        # convert our arguments to a dict and assign it to the meta.
        d = dict(*args, **kwargs)
        self.update(d)

    def __setitem__(self, key, value):
        """Set an item following the standards."""
        if not isinstance(key, str):
            raise TypeError('Only string keys accepted.'
                            f' Not {key} <{type(key)}>')

        comment = None
        # recognize if a value is passed with comment
        if isinstance(value, tuple):
            if len(value) > 2:
                raise ValueError('Tuple items should be 1-sized for single '
                                 'value, or 2-sized for (value, comment).')
            elif len(value) == 2:
                value, comment = value
            else:
                value = value[0]
        # any other format is passed as value-only.

        key = key.lower()
        # if key exists, just update it
        if key in self.__keys:
            indx = self.__keys.index(key)
            self.__values[indx] = value
            if comment is not None:
                self.__comms[indx] = comment
        else:
            self.__keys.append(key)
            self.__values.append(value)
            self.__comms.append(comment)

    def __getitem__(self, key):
        """Get an item following meta standards."""
        return self.__values[self.index(key)]

    def __delitem__(self, key):
        """Delete an item from the metadata."""
        index = self.index(key)
        self.__keys.pop(index)
        self.__values.pop(index)
        self.__comms.pop(index)

    def __contains__(self, key):
        """Check if meta contains a key."""
        return str(key).lower() in self.__keys

    def __eq__(self, other):
        d1 = self.to_dict()
        d2 = FrameMeta(other).to_dict()
        return d1 == d2

    def __len__(self):
        return len(self.__keys)

    def __repr__(self):
        s = 'FrameMeta:\n'
        for k, v, c in zip(self.keys(), self.values(), self.comments()):
            s += f"{k} = {v}  # {c}\n"
        return s

    def keys(self):
        """Iterate over the metadata keys."""
        for key in self.__keys:
            yield key

    def values(self):
        """Iterate over the metadata values."""
        for value in self.__values:
            yield value

    def comments(self):
        """Iterate over the metadata comments."""
        for comm in self.__comms:
            yield comm if comm is not None else ''

    def items(self):
        """Iterate over metadata items (key, value)."""
        for key, value in zip(self.__keys, self.__values):
            yield (key, value)

    def index(self, key):
        """Get the index of a given key in the meta."""
        key = str(key).lower()
        if key not in self.__keys:
            raise KeyError(key)
        return self.__keys.index(key)

    def update(self, other):
        """Update the meta values from other, overwriting existing keys."""
        d = dict(other)
        for k, v in d.items():
            self.__setitem__(k, v)

    def get(self, key, default=None):
        """Get the value of the key or the default."""
        try:
            v = self[key]
        except KeyError:
            v = default
        return v

    def pop(self, key, default=None):
        """Get the value of the key, removing the key from the meta."""
        v = self.get(key, default)
        try:
            self.__delitem__(key)
        except KeyError:
            pass
        return v

    def popitem(self, key, default=None):
        """Get the pair (key, value), removing it from the meta."""
        return (key, self.pop(key, default))

    def clear(self):
        """Clear all the dictionary."""
        self.__keys = []
        self.__values = []
        self.__comms = []

    def add(self, key, value=None, comment=None):
        """Add a key to the meta."""
        self.__setitem__(key, (value, comment))

    def remove(self, key):
        """Delete a key from the meta."""
        del self[key]

    def to_dict(self):
        return dict([(k, v) for k, v in self.items()])

    def copy(self):
        """Copy this meta to a new instance."""
        new = FrameMeta()
        new.__keys = [copy.copy(i) for i in self.__keys]
        new.__values = [copy.copy(i) for i in self.__values]
        new.__comms = [copy.copy(i) for i in self.__comms]
        return new

    def __copy__(self):
        return self.copy()
