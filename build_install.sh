pip install -r requirements.txt
pip install build
python -m build

latest_wheel=$(ls -t dist/*.whl | head -n 1)
pip install "$latest_wheel" --force-reinstall

install_loc=$(pip show cerebra_atlas_python | grep Location | cut -d " " -f 2)
rm -rf $install_loc/cerebra_atlas_python/cerebra_data
cp -r cerebra_atlas_python/cerebra_data $install_loc/cerebra_atlas_python/cerebra_data

