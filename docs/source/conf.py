# Configuration file for the Sphinx documentation builder.

# -- Project information
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

project = "BEExAI"
author = "Sithakoul"

release = "0.1"
version = "0.0.1"

# -- General configuration

extensions = [
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "nbsphinx_link",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "torchvision": ("https://pytorch.org/docs/master/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

source_suffix = [".rst", ".md"]
master_doc = "index"
language = "en"
pygments_style = "sphinx"
nbsphinx_allow_errors = True
nbsphinx_execute = "never"

intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#343131",
}

#html_logo = "images/name.png" #add logo here

epub_show_urls = "footnote"

nbsphinx_thumbnails = {
    "notebooks/nnclass": "../../source/images/test.png",
    "notebooks/xgbreg": "../../source/images/xgboost.png",
    "notebooks/starter": "../../source/images/1.png",
    "notebooks/explain": "../../source/images/2.png",
    "notebooks/metrics": "../../source/images/3.png",
}
