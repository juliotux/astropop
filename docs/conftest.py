"""Keep Sphinx implementation modules outside ordinary doctest collection."""

# pytest-doctestplus imports Python modules while collecting documentation.
# These helpers require the docs extra; the Sphinx build validates them.
collect_ignore = ['_ext']
