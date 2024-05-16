cd docs
pip install -r requirements.txt
sphinx-apidoc -o _auto_gen ../cerebra_atlas_python
make html

cd ..