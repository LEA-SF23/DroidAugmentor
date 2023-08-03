#!/bin/bash
[ $1 ] && [ -d $1 ] || {
	echo "Usage: $0 <code_directory>"
	echo " example: $0 ."
	exit
}
#sudo docker run -it --name=droidaugmentor-$RANDOM -v $(readlink -f $1):/droidaugmentor/shared -e DISPLAY=unix$DISPLAY sf23/droidaugmentor:latest python3 /droidaugmentor/setup/main.py --xx --yy --zz --kk ls ls ls
sudo docker run -it --name=droidaugmentor-$RANDOM -v $(readlink -f $1):/droidaugmentor/shared -e DISPLAY=unix$DISPLAY sf23/droidaugmentor:latest bash 
