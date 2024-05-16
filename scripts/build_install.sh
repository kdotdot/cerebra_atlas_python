pip install -r requirements.txt
pip install build
python -m build

latest_wheel=$(ls -t dist/*.whl | head -n 1)
pip install "$latest_wheel" --force-reinstall

# install_loc=$(pip show cerebra_atlas_python | grep Location | cut -d " " -f 2)

