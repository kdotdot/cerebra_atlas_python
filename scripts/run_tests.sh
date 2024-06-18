./scripts/clean.sh
cd tests
pip3 install -r requirements.txt
PYTHONPATH=".." && pytest
cd ..