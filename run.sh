#! /bin/bash

#Generate graph data for each dataset
python vgae_preprocessing.py DATASET

#Reproduce the experimental results
python train.py MODEL_NAME DATASET