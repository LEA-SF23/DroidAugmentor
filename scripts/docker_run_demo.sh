#!/bin/bash
DIR=$(readlink -f shared)
sudo docker run -it --name=droidaugmentor-$RANDOM -v $DIR:/droidaugmentor/shared -e DISPLAY=unix$DISPLAY sf23/droidaugmentor:latest bash 
