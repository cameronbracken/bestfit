# Dev / CI entry points. core/ + fixtures/ are the single source of truth; each
# package vendors them via subtree symlinks (see docs/superpowers/specs). Builds
# dereference the symlinks: R CMD build does it automatically, Python via
# tools/materialize_core.py in a throwaway worktree.
.PHONY: test-core test-r test-py build-r build-py materialize oracles docs docs-serve

test-core:
	cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build

test-r:
	R CMD build bestfitr
	R CMD INSTALL --preclean $$(ls -t bestfitr_*.tar.gz | head -1)
	Rscript -e 'testthat::test_local("bestfitr")'

test-py:
	python3 -m pip install --force-reinstall --no-deps ./bestfitpy
	python3 -m pytest bestfitpy/tests -q

build-r:
	R CMD build bestfitr

build-py:
	git worktree add -q /tmp/bf-buildpy HEAD
	cd /tmp/bf-buildpy && python3 tools/materialize_core.py && python3 -m build bestfitpy -o $(CURDIR)/dist
	git worktree remove --force /tmp/bf-buildpy

materialize:
	python3 tools/materialize_core.py

oracles:
	python3 tools/verify_oracles.py

# Build the full documentation site into site/_site (Quarto + quartodoc + pkgdown).
# Requires: quarto, an R install with pkgdown, and a Python env with bestfitpy +
# site/requirements.txt installed (the dev venv at ~/venv/bestfitpy works).
docs:
	cd site && quartodoc build
	quarto render site
	Rscript -e 'pkgdown::build_site("bestfitr", preview = FALSE)'
	mkdir -p site/_site/r
	cp -R bestfitr/docs/. site/_site/r/
	touch site/_site/.nojekyll

# Serve the ASSEMBLED site (quarto preview will not serve the pkgdown half at /r/).
docs-serve:
	python3 -m http.server -d site/_site 8000
