#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:46:04 2024

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
# Pipeline to preprocess EDF+ files, sleep signals and annotations. 
# It builds TFRecords for each database contained in a directory. 
#
# EDF+ format information: https://www.edfplus.info/
#
# Pipeline: 
#   · Read PSG and select signals
#   · Derivations construction and signal resampling (hz)
#   · Read annotations and annotations resampling (hz)
#   · Cut PSG to analyze only the TIB period 
#   · Split TIB PSG into windows with or without overlapping
#   · Link each window to its position in the entire PSG
#   · Link each window to its annotations → centroid calculation
#   · Normalize windows and annotations
#   · Build output vector and write TFR
#
# ··············································································
#
# Input channels are selected using ["c4","c6" or "c8"] depending on users preference
#   · channels_name = ['eeg1', 'eeg2', 'eog', 'emg'] → c4
#   · channels_name = ['eeg1', 'eeg2', 'eog', 'emg', 'saturation', 'airflow'] → c6
#   · channels_name = ['eeg1', 'eeg2', 'eog', 'emg', 'saturation', 'airflow', 'thores', 'abdores'] → c8
#
# Window: sequence length (N)
# δ: contextual information for each window (before and after) → extension = True
# Output vector: [p, c, w, h] → "p" presence/ausence, "c" class, "w" event onset, "h" event duration
#
# Amount and type of events to be detected (E) → selected with the variable "events_list"
#   · events_list = []                  → sleep stages
#   · events_list = ['resp']            → respiratory events and sleep stages
#   · events_list = ['arousal']         → EEG arousals and sleep stages
#   · events_list = ['arousal', 'resp'] → EEG arousals, respiratory events and sleep stages
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

# Database name
db_name = 'SHHS'

# Paths
main_path = 'path' # project path  

db_path = os.path.join(main_path, 'Databases', db_name) # database folder 
tfr_path = os.path.join(main_path, 'TFRecords') # TFR folder 
if not os.path.exists(tfr_path): os.makedirs(tfr_path)

# Config file
config_file = os.path.join(main_path, 'config.xml')

# Variables
window = 30 # input signal length
window_context_before = 60 # window context before (δ) (default 0)
window_context_after = 60 # window context after (δ) (default 0)
seq_len = window + window_context_before + window_context_after # total sequence length
overlap = 0 # window overlapping in seconds (i.e 10s → 1000 samples of overlapping)
hz = 100 # sample frequence 

input_channels = df.switch_channels('c4') # input channels
events_list = ['arousal', 'resp'] # amount and type of events to be detected
resp_variations = 'partial' # if events_list = ['arousal'] this parameter should be None 
#   · "None": consider all categories for the respiratory events 
#   · "partial": consider subcategories for the respiratory events ['NAN', 'Apnea', 'Hypopnea']
# ··············································································


# ··············································································
# MAIN CODE
# ··············································································
for dir, folders, files in os.walk(db_path): 

    # ······················
    # BLOCK 1 : PSGs
    # ······················
    if dir.endswith('PSGs'):
        counter = 0 
        
        # Create generic folder to save all TFRs
        db_tfr_folder_name = db_name + '_c' + str(len(input_channels)) + '_e' + str(seq_len) + '_stage'
        
        for name in events_list:
            tfr_folder_name = tfr_folder_name + '_' + name
        
        db_tfr_path = os.path.join(tfr_path, db_tfr_folder_name) 
        if not os.path.exists(db_tfr_path): os.makedirs(db_tfr_path)

        print('······················')
        print('\x1b[1;37;49mDATABASE: \x1b[0m', db_name)

        for file in files:
            os.chdir(dir) 

            signal_id = df.get_signal_id(db_name, file)
            print('\nProcessing signal: ', signal_id) 

            # Create specific folder to save each PSG TFRs
            psg_tfr_folder = os.path.join(db_tfr_path, signal_id)
            if not os.path.exists(psg_tfr_folder): os.makedirs(psg_tfr_folder)

            print('· Getting derivations...')  
            selected_signals = df.get_signals(file, db_name, input_channels, config_file)

            # Resampling signals (hz)
            print('· Resampling signals...')
            resampled_signals = df.resample_signals(selected_signals, hz)

            # ······················
            # BLOCK 2: Annotations
            # ······················
            for di, fold, fil in os.walk(db_path):               
                if (di.endswith('Annotations')):
                    os.chdir(di) 

                    for fi in fil:
                        if signal_id in fi:
                            print('· Getting annotations  -  file: {0}'.format(fi))
                            print('· Resampling annotations...')

                            scorer_id = df.get_scorer_id(db_name, fi)

                            # Resampling annotations
                            resampled_annotations = df.resample_annotations(fi, hz) 
                            
                            # ······················
                            # BLOCK 3: TIB period
                            # ······················
                            offset, onset = df.get_lights_index(resampled_annotations, index=3) # lights on and lights off index 
                            eval_signal_len = offset - onset 

                            # Cut the PSG
                            print('· Cutting the psg...')
                            idx, slices = df.get_slices(resampled_signals, window, hz, overlap)

                            # Remove windows not in TIB
                            print('· Applying light filter...')
                            filtered_idx, filtered_slices = df.apply_light_filter(idx, slices, onset, offset)
                            n_slices, n_channels, n_samples = np.shape(filtered_slices)

                            # Window annotations
                            print('· Getting slices annotations...')
                            slices_annotations = df.get_slices_annotations(filtered_idx, resampled_annotations)

                            # ······················
                            # BLOCK 4: Extend and normalize
                            # ······················
                            print('· Getting context for each window...')
                            idx, filtered_slices = df.extend_windows(filtered_idx, resampled_signals, window, hz, window_context_before, window_context_after)

                            # Normalize signals
                            print('· Applying normalization...')
                            norm_slices = df.normalize_signals(filtered_slices)

                            # Normalize annotations
                            norm_annotations = df.normalize_annotations(slices_annotations, filtered_idx, n_samples)                           

                            # ······················
                            # BLOCK 5: Vectorization
                            # ······················
                            print('· Building output vectors...')
                            outputs, events_names, idx_reg_pos = df.vectorize_events(norm_annotations, events_list, resp_variations)
                            
                            # ······················
                            # BLOCK 6: TFR
                            # ······················
                            print('· Creating TFRecords...')
                            df.create_tfr(norm_slices, outputs, events_list, psg_tfr_folder, db_name, signal_id, resp_variations, scorer_id)
                            
                            counter += 1
               
        # Print information
        print('\x1b[1;37;49m\n**************\x1b[0m')
        print('\x1b[1;37;49m{0} {1} database files have been written\x1b[0m'.format(counter, db_name))
        print('··········')
        print('Parameters used: ')
        print('· Channels: ', input_channels)
        print('· Detected events: ', events_names)
        print('· Frequency: ', hz)
        print('· Overlap: ', overlap)
        print('· Window: ', window)
        print('· Extension: {0}\tseconds_before: {1}\tseconds_after: {2}'.format(seq_len, window_context_before, window_context_after))
        print('··········')
        print(' Annotation example: ', norm_annotations[0])
        print(' Output example: {0}\t len: {1}'.format(outputs[0], len(outputs[0])))
        print('\x1b[1;37;49m**************\x1b[0m')
# ··············································································