import sys
import os

ap_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ap_dir)
sys.path.insert(0, os.path.join(ap_dir, 'docs', '_ext'))

# Minimum version, enforced by sphinx
needs_sphinx = '9.1'

# Check missing object references in local and hosted builds alike.
nitpicky = True

extensions = [
    'annotation_names',
    'catalog_help',
    'tutorial_data',
    'workflow_figures',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinxcontrib.mermaid',
    'sphinx_automodapi.automodapi',
    'numpydoc',
    'sphinx_autodoc_typehints',
    'matplotlib.sphinxext.plot_directive',
]

extlinks = {
    'doi': ('https://dx.doi.org/%s', 'doi: %s'),
    'bibcode': ('https://ui.adsabs.harvard.edu/abs/%s', '%s')
}

plot_html_show_source_link = False
plot_include_source = False
plot_html_show_formats = False

# Pin the browser renderer so diagram appearance changes only when reviewed.
mermaid_version = '12.2.0'
mermaid_light_theme = 'redux'
mermaid_dark_theme = 'redux-dark'
mermaid_fullscreen = True
mermaid_height = 'auto'
mermaid_init_config = {
    'startOnLoad': False,
    'look': 'neo',
    'fontFamily': 'system-ui, sans-serif',
    'flowchart': {'curve': 'basis', 'wrappingWidth': 260},
}

todo_include_todos = True

# The type-hint extension fills NumPyDoc parameter/return fields directly.
# Disable Sphinx's separate injector to avoid a second parameter list.
autodoc_typehints = 'none'
always_document_param_types = True
typehints_document_rtype_none = False
from annotation_names import format_alias
typehints_formatter = format_alias
autodoc_typehints_format = 'short'
# Preserve shared aliases instead of expanding large unions or resolving cyclic
# FrameData/QFloat names in the importing function's namespace.
autodoc_type_aliases = {
    name: 'astropop._typing.' + name[1:]
    for name in ('_ImageLike', '_ImageInput', '_QuantityInput', '_NumberArray', '_UnitLike',
                 '_HeaderLike', '_PathLike', '_ImageHDU', '_Index', '_Axis',
                 '_Window', '_Statistic')
}
autodoc_type_aliases.update({
    '_Extension': 'astropop.file_collection._Extension',
    '_npt.ArrayLike': 'numpy.typing.ArrayLike',
    '_npt.DTypeLike': 'numpy.typing.DTypeLike',
})


# NumPy exposes this private context-manager type in two imported test helpers;
# its implementation class has no public inventory target.
nitpick_ignore = [('py:class', 'unittest.case._AssertRaisesContext')]

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
author = 'Astropop contributors'
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
html_css_files = ['diagrams.css']
htmlhelp_basename = 'astropop'
html_theme_options = {
  "show_prev_next": False,
}

autosummary_generate = True

default_role = 'py:obj'
intersphinx_mapping = {
    'photutils': ('https://photutils.readthedocs.io/en/stable/', None),
    'skimage': ('https://scikit-image.org/docs/stable/', None),
    'astroalign': ('https://astroalign.quatrope.org/en/latest/', None),
    'ccdproc': ('https://ccdproc.readthedocs.io/en/latest/', None),
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ('https://docs.astropy.org/en/stable/', None),
    'astroquery': ('https://astroquery.readthedocs.io/en/latest/', None),
    'packaging': ('https://packaging.pypa.io/en/stable/', None),
    'uncertainties': ('https://uncertainties.readthedocs.io/en/latest/', None)
}
