#!/bin/bash
DIR=$(readlink -f shared)
sudo docker image pull sf23/droidaugmentor:latest
sudo docker run -it --name=droidaugmentor-$RANDOM -v $DIR:/droidaugmentor/shared -e DISPLAY=unix$DISPLAY sf23/droidaugmentor:latest /droidaugmentor/shared/app_run.sh --verbosity 20 --output_dir /droidaugmentor/shared/outputs --input_dataset /droidaugmentor/shared/datasets/drebin215_small_64Malwares_64Benign.csv --dense_layer_sizes_g 256 --dense_layer_sizes_d 256 --number_epochs 1000 --training_algorithm Adam
ls shared/outputs/
