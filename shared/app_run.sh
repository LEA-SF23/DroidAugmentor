#!/bin/bash

echo "=============================================================="
echo "Running app with parameters: $*"
echo "=============================================================="

cd /droidaugmentor/
pipenv run python main.py  $*

