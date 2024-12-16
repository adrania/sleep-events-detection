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
# Pipeline to evaluate a deep-learning model
# It saves the corresponding test keras and sklearn metrics
# ··············································································

# ··············································································
# USED VERSIONS
# ·············································································· 
# python 3.11.4
# CUDA 12.2
# CUDA driver 535.86.05
#
# tensorflow 2.12.0
#  · Verify tensorflow GPU requirements (https://www.tensorflow.org/install/pip#linux)
# ··············································································

# ··············································································
# LIBRARIES 
# ··············································································
import os
import keras
import datetime
import numpy as np 
import tensorflow as tf
import random as python_random
import tensorflow_addons as tfa

import detection_functions as df

# Configuration
tf.debugging.set_log_device_placement(False)
print('Num GPUs Availables: ', len(tf.config.list_physical_devices('GPU')))
# ··············································································


# ··············································································
# GLOBAL VARIABLES
# ··············································································
# Configuration
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)
python_random.seed(seed)

date_time = datetime.datetime.now()

# Dataset and model names
dataset_folder_name = 'HMCP_c8_e5_stage_arousal' # database name 
dataset_name = 'full_dataset_scorer1.npy' # dataset name 
model_name = 'm46_SHHS_c8_e5_1events_03-22-2024_13-46-23' # model name

# ·························
# Paths
main_path = 'path' # project path 

models_path = os.path.join(main_path, 'Models')
logs_path = os.path.join(main_path, 'Logs')
tfr_path = os.path.join(main_path, 'TFRecords', dataset_folder_name)
datasets_path = os.path.join(main_path, 'Datasets', dataset_folder_name)

input_channels = int(dataset_folder_name.split('_')[1].replace('c','')) # input channels
seq_len = int(dataset_folder_name.split('_')[2].replace('e','')) # sequence length
# ·························

# Variables 
resp_variations = 'partial' # None to include all respiratory subcategories
window = 30 # sequence length (N)
hz = 100 # sampling rate
n_samples = window * hz # number of samples
split = 5 # time distribution split
th = 0.5 # ausence/presence threshold 
batch_size = 100

n_events = len(dataset_folder_name.split('_')[3:]) 
n_resp_clas = df.get_num_resp_events(resp_variations) 
n_arousal_clas = 1 
n_stages_clas = 5 
location_positions = 2
# ··············································································


# ··············································································
# MAIN CODE
# ··············································································
# Load datasets
print('· Loading dataset: ', os.path.join(datasets_path, dataset_name))
data = np.load(os.path.join(datasets_path, dataset_name))

# Create dataset generator
print('· Creating dataset generator')
dataset = df.tfr_dataset_generator(data, len(data), batch_size, split, input_channels, n_samples, seed)

# Load model
print('· Loading model: ', model_name)
model = keras.models.load_model(os.path.join(models_path, model_name))

# Predict
print('· Making predictions...')
predictions = model.predict(dataset)
# ··············································································

# ··············································································
# KERAS METRICS
# ··············································································
print('\n· Keras model evaluation...')
keras_metrics = model.evaluate(dataset, verbose=0, return_dict=True)

df.print_evaluation(keras_metrics, dataset_name)
# ··············································································

# ··············································································
# SKLEARN METRICS
# ··············································································
sklearn_metrics = df.calculate_metrics(model, predictions, dataset, n_stages_clas, n_resp_clas, th)

# ·························
# MEAN ABSOLUTE ERROR (sklearn)
# ·························
# Categorize predictions
ind, cat_preds, true_labels = df.categorize_predictions(model, dataset, predictions, n_stages_clas, n_resp_clas, th)

# Specific MAE
df.calculate_specific_mae(dataset_folder_name, n_events, true_labels, cat_preds, ind)

# ··············································································
# CLEAR SESSION
# ··············································································
tf.keras.backend.clear_session() # resets all state generated by Keras
# ··············································································