#!/bin/bash
[ $1 ] && [ -d $1 ] || {
	echo "Usage: $0 <code_directory>"
	echo " example: $0 ."
	exit
}
sudo docker run -it --name=droidaugmentor-$RANDOM -v $(readlink -f $1):/droidaugmentor/ -e DISPLAY=unix$DISPLAY droidaugmentor/tensorflow 
