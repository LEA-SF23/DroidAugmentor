#!/bin/bash

for i in $(sudo docker ps -a | awk '{print $1}' | grep -v CONTAINER)
do 
	sudo docker rm -f $i
done
