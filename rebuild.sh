rm -rf dist/*
python -m build
pip uninstall biclustlib
pip install dist/*.whl