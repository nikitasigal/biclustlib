rm -rf dist/*
python -m build
pip install --upgrade dist/*.whl