#!/bin/bash
sudo docker run -it --name=droidaumentor-$RANDOM -e DISPLAY=unix$DISPLAY droidaugmentor/tensorflow 
