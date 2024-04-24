# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Convert Markdown files to reStructuredText before building HTML
markdown_to_rst:
	@echo "Converting Markdown files to reStructuredText"
	@mkdir -p $(SOURCEDIR)/wiki/wiki_examples/selected_plots
	@cp $(SOURCEDIR)/wiki_editable/wiki_examples/selected_plots/* $(SOURCEDIR)/wiki/wiki_examples/selected_plots/
	@find $(SOURCEDIR)/wiki_editable -name '*.md' -exec sh -c 'pandoc -f markdown -t rst "$$1" -o "$(SOURCEDIR)/wiki/$$(basename "$$1" .md).rst"' _ {} \;
	@echo "Conversion complete."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html: markdown_to_rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# # Catch-all target: route all unknown targets to Sphinx using the new
# # "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
