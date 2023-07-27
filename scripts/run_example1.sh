#!/bin/bash
[ -d outputs ] || { mkdir outputs; }
pipenv run python generation.py --input_dataset  datasets/drebin215_small_64Malwares_64Benign.csv --data_type float32 --num_samples_class_malware 64 --num_samples_class_benign 64 --number_epochs 1000 --classifier knn --k_fold 5 --latent_dimension 128 --output_dataset outputs/synthetic_dataset_drebin215_small_64Malwares_64Benign.csv 
