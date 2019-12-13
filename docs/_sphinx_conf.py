import sys, os, re
import sphinx_rtd_theme

if not os.path.exists('api'):
    os.mkdir('api')

# Minimum version, enforced by sphinx
needs_sphinx = '2.2.0'

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.imgmath',
]
numpydoc_show_class_members = False
autosummary_generate = ["reference.rst"]
autoclass_content = "class"
autodoc_default_flags = ["members", "no-special-members"]

exclude_patterns = ['_templates', '_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = []
imgmath_image_format = 'svg'
htmlhelp_basename = 'astropop'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

master_doc = 'index'

project = 'astropop'
copyright = '2018-2020, Julio Campagnolo and contributors'
import astropop
version = astropop.__version__.split('-', 1)[0]
# The full version, including alpha/beta/rc tags.
release = astropop.__version__
today_fmt = '%B %d, %Y'


nitpicky = True
nitpick_ignore = []
# Some warnings are impossible to suppress, and you can list specific references
# that should be ignored in a nitpick-exceptions file which should be inside
# the docs/ directory. The format of the file should be:
#
# <type> <class>
#
# for example:
#
# py:class astropy.io.votable.tree.Element
# py:class astropy.io.votable.tree.SimpleElement
# py:class astropy.io.votable.tree.SimpleElementWithContent
#
# Uncomment the following lines to enable the exceptions:
#
# for line in open('nitpick-exceptions'):
#     if line.strip() == "" or line.startswith("#"):
#         continue
#     dtype, target = line.split(None, 1)
#     target = target.strip()
#     nitpick_ignore.append((dtype, target))
