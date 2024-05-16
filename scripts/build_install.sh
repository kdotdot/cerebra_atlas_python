pip3 install -r requirements.txt
pip3 install build
python3 -m build

latest_wheel=$(ls -t dist/*.whl | head -n 1)
pip3 install "$latest_wheel" --force-reinstall

# install_loc=$(pip show cerebra_atlas_python | grep Location | cut -d " " -f 2)

