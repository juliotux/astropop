import sys, os, re
import sphinx_rtd_theme

# Minimum version, enforced by sphinx
needs_sphinx = '4.3.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'nbsphinx',
]

nbsphinx_allow_errors = True
nbsphinx_execute = 'never'
nbsphinx_kernel_name = 'python3'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# The suffix of source filenames.
source_suffix = '.rst'
# Doc entrypoint
master_doc = 'index'

# Project info
project = 'astropop'
copyright = '2018-2021, Julio Campagnolo and contributors'
import astropop
version = astropop.__version__.split('-', 1)[0]
# The full version, including alpha/beta/rc tags.
release = astropop.__version__
today_fmt = '%B %d, %Y'

# General theme infos
exclude_patterns = ['_templates', '_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
imgmath_image_format = 'svg'
htmlhelp_basename = 'astropop'

autosummary_generate = True

default_role = 'py:obj'
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "astropy": ('http://docs.astropy.org/en/latest/', None)
}
