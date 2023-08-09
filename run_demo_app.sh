#!/bin/bash
pipenv run python3 main.py --verbosity 20 --output_dir outputs --input_dataset shared/datasets/drebin215_small_64Malwares_64Benign.csv --dense_layer_sizes_g 256 --dense_layer_sizes_d 256 --number_epochs 1000 --training_algorithm Adam
