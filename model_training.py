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
# Pipeline to build a deep-learning model for sleep events detection and hypnogram contruction
# This script saves the model and its training history in a log file
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import random as python_random
import tensorflow_addons as tfa 
 
import detection_functions as df
import net

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
dataset_folder_name = 'SHHS_c4_e5_stage_arousal_resp' # dataset used for training 
scorer = None
m = '1' # model name 

# ·························
# Paths 
main_path = 'path' # project path 

models_path = os.path.join(main_path, 'Models')
logs_path = os.path.join(main_path, 'Logs')
tfr_path = os.path.join(main_path, 'TFRecords', dataset_folder_name)
datasets_path = os.path.join(main_path, 'Datasets', dataset_folder_name)

input_channels = int(dataset_folder_name.split('_')[1].replace('c','')) # amount of channels used
seq_len = int(dataset_folder_name.split('_')[2].replace('e','')) # sequence length
# ·························
    
# Variables 
resp_variations = 'partial' # None to include all respiratory subcategories
window = 30 # sequence length (N) 
hz = 100 # sampling rate (hz)
n_samples = window * hz # number of samples per window
split = 5 # time distribution split
th = 0.5 # presence/ausence threshold
 
n_resp_clas = df.get_num_resp_events(resp_variations) # number of respiratory categories
n_arousal_clas = 1 # number of arousal classes (presence/ausence)
n_stages_clas = 5 # number of sleep stages 
location_positions = 2 # location positionsn (onset, offset)
# ··············································································

# ··············································································
# MODEL CONFIGURATION
# ··············································································
filters = 8
lstm_units = 1000
lr = 0.001 
momentum = 0.9 
patience = 5 
epochs = 100
batch_size = 100

BATCH_NORM = False
DROP_OUT = True

n_events, loss, metrics = df.get_model_configuration(dataset_folder_name, n_stages_clas, n_resp_clas)

optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)

model_name = "m{0}_{1}_c{2}_e{3}_{4}ev_{5}".format(m, dataset_folder_name.split('_')[0], input_channels, split, n_events, date_time.strftime("%m-%d-%Y_%H-%M-%S"))
csv_name = "logs_m{0}_{1}_c{2}_e{3}_{4}ev_{5}".format(m, dataset_folder_name.split('_')[0], input_channels, split, n_events, date_time.strftime("%m-%d-%Y_%H-%M-%S"))


print('·····························')
print('\x1b[1;33;47mModel name: {0}\x1b[0m '.format(model_name))
print('\x1b[1;37;49m· Model configuration:\x1b[0m \nbatch size = {0}, filters = {1}, units = {2}, lr = {3}, momentum = {4}, patience = {5}, epochs = {6}, batch_norm = {7}, drop_out = {8}, optimizer = {9}, loss = {10}'.format(
    batch_size, filters, lstm_units, lr, momentum, patience, epochs, BATCH_NORM, DROP_OUT, optimizer._name, loss.items()))
print('\x1b[1;37;49m· Data configuration:\x1b[0m \ndatabase = {0}, scorer = {1}, channels = {2}, split = {3}, num_samples = {4}, th = {5}, num_events = {6}, num_resp_clas = {7}, metrics = {8}'.format(
    dataset_folder_name.split('_')[0], scorer, input_channels, split, n_samples, th, n_events, n_resp_clas, metrics.items()))
print('·····························\n')
# ··············································································

# ··············································································
# LOAD DATASETS
# ··············································································
# Load datasets
print('· Loading datasets...')
train, validation, test = df.load_datasets(datasets_path, scorer)

# Dataset generators
print('· Creating datasets generators...')
train_dataset, feature = df.tfr_dataset_generator(train, len(train), batch_size, split, input_channels, n_samples, seed, catched = True, shuffle = True, feature = True)
val_dataset = df.tfr_dataset_generator(validation, len(validation), batch_size, split, input_channels, n_samples, seed, catched = True)
test_dataset = df.tfr_dataset_generator(test, len(test), batch_size, split, input_channels, n_samples, seed)
# ··············································································

# ··············································································
# MODEL TRAINING
# ··············································································
print('· Creating keras model...\n')
cnn = net.create_cnn_functional(feature, input_channels, n_samples, n_stages_clas, n_resp_clas, n_arousal_clas, location_positions, filters, BATCH_NORM, DROP_OUT, cnn_model=False)
model = net.create_lstm_functional(cnn, feature, split, input_channels, n_samples, n_stages_clas, n_resp_clas, n_arousal_clas, location_positions, lstm_units)

model.summary()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics) 

# ··················
# TRAIN
# ··················
print('\n· Starting trainig...')
history = model.fit(train_dataset, epochs=epochs, batch_size=batch_size, validation_data=val_dataset, callbacks=early_stop, verbose=1) 
# ··················

# ··················
# MODEL SAVING 
# ··················
print('\n· Saving model...')
os.chdir(models_path)
model.save(model_name); 
print('Model saved!')
# ··················

# ··············································································
# LOGS AND METRICS SAVING 
# ··············································································
print('\n· Saving logs...')
os.chdir(logs_path)

logs_df = pd.DataFrame.from_dict(history.history)
logs_df.to_csv(csv_name) 

with open(csv_name, 'a', newline='') as file: 
    file.write('\nModel name: {0}'.format(model_name))
    file.write('\nModel configuration: \nbatch size = {0}, filters = {1}, units = {2}, lr = {3}, momentum = {4}, patience = {5}, epochs = {6}, batch_norm = {7}, drop_out = {8}, optimizer = {9}, loss = {10}'.format(
    batch_size, filters, lstm_units, lr, momentum, patience, epochs, BATCH_NORM, DROP_OUT, optimizer._name, loss.items()))
    file.write('\nData configuration: \ndatabase = {0}, scorer = {1}, channels = {2}, th = {3}, num_stages_units = {4}, num_events = {5}, num_resp_clas = {6}, num_event_clas = {7}, num_positions = {8}, metrics = {9}'.format(
    dataset_folder_name.split('_')[0], scorer, input_channels, th, n_stages_clas, n_events, n_resp_clas, n_arousal_clas, location_positions, metrics.items()))

print('*** Done ***')
# ··············································································

# ··············································································
# CLEAR SESSION
# ··············································································
tf.keras.backend.clear_session() # resets all state generated by Keras
# ··············································································