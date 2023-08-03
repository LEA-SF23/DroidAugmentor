#!/bin/bash
sudo docker run -it --name=droidaumentor-$RANDOM -e DISPLAY=unix$DISPLAY sf23/droidaugmentor:latest
