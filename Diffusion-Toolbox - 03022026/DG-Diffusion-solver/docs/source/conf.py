# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EvalDdiffusionExp'
copyright = '2025, Georg Kosakowski'
author = 'Georg Kosakowski'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sys
from pathlib import Path

sys.path.insert(0, str(Path('../../', 'sources').resolve())) # works only for source files...not for notebooks

extensions = [
#    'nbsphinx',
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.duration',
    'sphinx.ext.doctest'
]


#source_suffix = [".rst"]
nbsphinx_execute = 'auto' # for nbsphinx not executing every time the notebooks

templates_path = ['_templates']
exclude_patterns = []

numpydoc_show_class_members = False # should solve problems with automodule/autosummary??

# -- myst-nb 
nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
extensions.append("sphinx_wagtail_theme")
html_theme = 'sphinx_wagtail_theme'
#html_static_path = ['_static']

# These are options specifically for the Wagtail Theme.


html_theme_options = dict(
    project_name = "Fitting diffusion experiments with jupyter & ADE_DG",
    logo = "../_images/ADE_DG_logo.png",
    logo_alt = "ADE_DG",
    logo_height = 60,
    logo_url = "./index.html",
    logo_width = 60,
    header_links = "Documentation |./index.html, Templates |./templates.html",
    footer_links = ",".join([
        "About Us|https://www.psi.ch/de/les",
        "Contact|https://www.psi.ch/en/les/people/georg-kosakowski",
    ]),
    github_url = "https://gitea.psi.ch/kosakowski/FEniCSx-GEMS-Toolbox/src/branch/main/Diffusion-Test-Models/DG-Diffusion-solver/docs/source/"
 )
copyright = "2025, Paul Scherrer Institut"
html_show_copyright = True

html_last_updated_fmt = "%b %d, %Y"

html_show_sphinx = False