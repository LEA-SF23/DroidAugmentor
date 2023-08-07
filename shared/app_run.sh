#!/bin/bash

cd /droidaugmentor/DroidAugmentor
git pull

echo "=============================================================="
echo "Running app with parameters: $*"
echo "=============================================================="

pipenv run python main.py  $*

