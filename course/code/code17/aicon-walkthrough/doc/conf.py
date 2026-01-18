extensions = ["myst_nb"]
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

project = 'AICON Walkthrough'
author = 'Florian Prill (DWD), May 2025'

exclude_patterns = ['aicon-anemoi/*']

html_title = 'AICON Walkthrough'

html_theme = "sphinx_book_theme"
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_logo = "_static/logo.png"

jupyter_execute_notebooks = "off"

myst_enable_extensions = ["dollarmath", "amsmath", "html_image"]


html_theme_options = {
    "navbar_end": ["navbar-icon-links"],
    "default_mode": "light"
}
