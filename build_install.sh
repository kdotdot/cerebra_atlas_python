pip install -r requirements.txt
pip install build
python -m build

latest_wheel=$(ls -t dist/*.whl | head -n 1)
pip install "$latest_wheel" --force-reinstall
