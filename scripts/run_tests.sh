cd tests
pip install -r requirements.txt
PYTHONPATH=".." && pytest
cd ..