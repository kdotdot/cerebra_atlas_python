cd docs
pip3 install -r requirements.txt
rm -rf _auto_gen
sphinx-apidoc -o _auto_gen ../cerebra_atlas_python -l -t=_templates/apidoc -e -M -d 1
# -e  -a   -f -T 
make html

cd ..