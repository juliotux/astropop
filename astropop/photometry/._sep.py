# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Wrapper for SEP import, handling the multiple possible forks."""


try:
    import sep_pjw as sep
except ImportError:
    try:
        import sep
    except ImportError:
        raise ImportError('SEP not found. Please install it.')


__all__ = ['sep']
