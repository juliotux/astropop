# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import sys
import datetime
from importlib import import_module

try:
    from sphinx_astropy.conf.v1 import *  # noqa
except ImportError:
    print('ERROR: the documentation requires the sphinx-astropy package '
          'to be installed')
    sys.exit(1)

from configparser import ConfigParser
conf = ConfigParser()

conf.read([os.path.join(os.path.dirname(__file__), '..', 'setup.cfg')])
setup_cfg = dict(conf.items('metadata'))
highlight_language = 'python3'
exclude_patterns += ['_templates', '_build', 'Thumbs.db', '.DS_Store']

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog += """
"""

project = setup_cfg['name']
author = setup_cfg['author']
copyright = '{0}, {1}'.format(
    datetime.datetime.now().year, setup_cfg['author'])

import_module(setup_cfg['name'])
package = sys.modules[setup_cfg['name']]

version = package.__version__.split('-', 1)[0]
release = package.__version__

html_theme = "sphinx_rtd_theme"
html_title = '{0} v{1}'.format(project, release)
htmlhelp_basename = project + 'doc'

latex_documents = [('index', project + '.tex', project + u' Documentation',
                    author, 'manual')]

man_pages = [('index', project.lower(), project + u' Documentation',
              [author], 1)]

if eval(setup_cfg.get('edit_on_github')):
    extensions += ['sphinx_astropy.ext.edit_on_github']

    versionmod = __import__(setup_cfg['package_name'] + '.version')
    edit_on_github_project = setup_cfg['github_project']
    if versionmod.version.release:
        edit_on_github_branch = "v" + versionmod.version.version
    else:
        edit_on_github_branch = "master"

    edit_on_github_source_root = ""
    edit_on_github_doc_root = "docs"

github_issues_url = ('https://github.com/{0}/issues/'
                     .format(setup_cfg['github_project']))

extensions += ['sphinx.ext.todo']
todo_include_todos = True


extensions += ['sphinx_automodapi.automodapi',
               'sphinx_automodapi.smart_resolver']
numpydoc_show_class_members = False
