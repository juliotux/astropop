# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ._base import _GenericConfigClass, info_dumper


class Instrument(_GenericConfigClass):
    """Store all the informations and needed functions of a instrument."""
    _frozen = False
    _prop_dict = {}
    _mutable_vars = ['_frozen']
    _identifier = 'dummy_instrument'

    def __init__(self, *args, **kwargs):
        super(Instrument, self).__init__(*args, **kwargs)

    def list_functions(self):
        """List the class functions."""
        # Pass any callable object that do not start with '_' may cause problem
        # This may pass unwanted functions. Commented out.
        # l = [i.__name__ for i in self.__class__.__dict__.values()
        #      if callable(i) and i.__name__[0] != '_']

        funcs = [i.__name__ for i in self.__class__.__dict__.values()
                 if type(i) in ['function', 'builtin_function_or_method'] and
                 i.__name__[0] != '_']

        # If needed to remove another class function, put here
        for i in ['list_functions']:
            if i in funcs:
                funcs.remove(i)
        return funcs

    def __str__(self):
        info = "{} ({})\n\n".format(self.__class__.__name__, self._identifier)
        info += info_dumper({'Properties': self.properties,
                             'Functions': self.list_functions()})
        return info

    def __repr__(self):
        return "{} ({})".format(self.__class__.__name__, self._identifier)
