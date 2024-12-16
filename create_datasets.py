#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:03:24 2024

@author: adriana
"""

# ··············································································
# ENVIRONMENT
# ··············································································
#  · Environment type: conda
#  · Environment specifications listed in spec-file-env-detection.txt
#  · To create a clone of the environment use: conda create --name <name_envir> --file spec-file-env-detection.txt
#  · Activate the environment: conda activate <name_envir>
# ··············································································

# ··············································································
# CODE DESCRIPTION
# ··············································································
# Pipeline to create training, validation and test datasets
# ··············································································

# ··············································································
# USED VERSIONS
# ··············································································
# python 3.11.4
# CUDA 12.2
# CUDA driver 535.86.05
#
#  · Verify tensorflow GPU requirements (https://www.tensorflow.org/install/pip#linux)
# ··············································································

# ··············································································
# LIBRARIES 
# ··············································································
import os
import numpy as np

import detection_functions as df
# ··············································································

# ··············································································
# GLOBAL VARIABLES
# ··············································································
# Configuration
seed = 0
np.random.seed(seed)

# Dataset name
dataset_name = 'dataset_name' # name of the dataset to split  
scorer = None 

# Paths
main_path = 'PATH' # project path 
tfr_path = os.path.join(main_path, 'TFRecords', dataset_name)
datasets_path = os.path.join(main_path, 'Datasets', dataset_name)
if not os.path.exists(datasets_path): os.makedirs(datasets_path) # make directory
# ··············································································


# ··············································································
# MAIN CODE
# ··············································································
# List TFR files:
print('· Creating datasets...')
files = df.list_files(tfr_path, scorer)

# Split data into train, validation and test datasets (80-20 default)
train, test = df.split_dataset(files, seed)
train, validation = df.split_dataset(train, seed)

# Save datasets
df.save_datasets(datasets_path, train, validation, test, scorer)