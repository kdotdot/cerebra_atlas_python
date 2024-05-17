rm -rf _autosummary source _build
sphinx-apidoc -o source ../cerebra_atlas_python/  -e
# rm -rf source/cerebra_atlas_python.rst
make html