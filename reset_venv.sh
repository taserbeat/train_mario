#!/bin/bash

echo Now removing virtualenv...
rm -r .venv
echo Done removing virtualenv!

echo Now creating new virtualenv...
python -m venv .venv
echo Done creating new virtualenv!

shopt -s expand_aliases
source .venv/bin/activate

echo Now updating pip...
python -m pip install -U pip
echo Done updating pip!

echo Now pip installing...
pip install -r requirements.txt
echo Done pip installing!