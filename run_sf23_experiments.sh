#!/bin/bash

pipenv install -r requirements.txt

pipenv run python3 run_campaign.py -c sf23_1l_256,sf23_1l_1024,sf23_1l_4096

