import sys
import os

ap_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ap_dir)
sys.tracebacklimit = 0

# Minimum version, enforced by sphinx
needs_sphinx = '4.3.0'

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx_automodapi.automodapi',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'nbsphinx',
]

extlinks = {
    'doi': ('https://dx.doi.org/%s', 'doi: %s'),
    'bibcode': ('https://ui.adsabs.harvard.edu/abs/%s', '%s')
}

nbsphinx_allow_errors = True
nbsphinx_execute = 'never'
nbsphinx_kernel_name = 'python3'

plot_html_show_source_link = False
plot_include_source = False
plot_html_show_formats = False

todo_include_todos = True

numpydoc_show_class_members = False
numpydoc_attributes_as_param_list = False
# numpydoc_xref_aliases = {}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# The suffix of source filenames.
source_suffix = '.rst'
# Doc entrypoint
master_doc = 'index'

# Project info
project = 'astropop'
copyright = '2018-2023, Julio Campagnolo and contributors'
import astropop
version = astropop.__version__.split('-', 1)[0]
# The full version, including alpha/beta/rc tags.
release = astropop.__version__
today_fmt = '%B %d, %Y'

# General theme infos
exclude_patterns = ['_templates', '_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
# html_theme = 'pydata_sphinx_theme'
html_theme = "sphinx_book_theme"
html_static_path = ['_static']
htmlhelp_basename = 'astropop'
html_theme_options = {
  "show_prev_next": False,
  "footer_items": ["copyright", "sphinx-version", "theme-version"]
}

autosummary_generate = True

default_role = 'py:obj'
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "astropy": ('http://docs.astropy.org/en/latest/', None),
    'astroquery': ('https://astroquery.readthedocs.io/en/latest/', None),
    'uncertainties': ('https://uncertainties-python-package.readthedocs.io/en/latest/', None)
}
