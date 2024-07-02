# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import sys
from os.path import join
quapy_path = join(pathlib.Path(__file__).parents[2].resolve().as_posix(), 'quapy')
wiki_path = join(pathlib.Path(__file__).parents[0].resolve().as_posix(), 'wiki')
print(f'quapy path={quapy_path}')
sys.path.insert(0, quapy_path)
sys.path.insert(0, wiki_path)


project = 'QuaPy: A Python-based open-source framework for quantification'
copyright = '2024, Alejandro Moreo'
author = 'Alejandro Moreo'



import quapy

release = quapy.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

source_suffix = ['.rst', '.md']

templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_theme = 'furo'
# need to be installed: pip install furo (not working...)
# html_static_path = ['_static']

# intersphinx configuration
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

