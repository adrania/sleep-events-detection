#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:36:04 2024

@author: adriana
"""

# ··············································································
# DESCRIPTION 
# ··············································································
# Generic functions 
# ··············································································


# ··············································································
# USED VERSIONS
# ··············································································
# python 3.11.4
# CUDA 12.2
# CUDA driver 535.86.05
#
# tensorflow 2.12.0
#
#  · Verify tensorflow GPU requirements (https://www.tensorflow.org/install/pip#linux)
#  · Follow https://github.com/holgern/pyedflib instructions
#      · For conda users apply: conda install -c conda-forge pyedflib
# ··············································································


# ··············································································
# LIBRARIES 
# ··············································································
import os
import math
import keras
import numbers
import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
import pyedflib as edflib
from itertools import chain 
from pyedflib import highlevel
import tensorflow_addons as tfa
from scipy.signal import resample
import xml.etree.ElementTree as ET
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
# ··············································································


# ··············································································
# FILES MANAGING
# ··············································································
def find_single_file_path(filename, path):
    '''Search some file in a directory.    
    Returns the file directory.'''

    for root, dirs, files in os.walk(path):
        if filename in files:
            return os.path.join(root, filename)

def list_files (path, scorer=None):
    '''Group files.
    Return a list of files .'''

    list_of_files = []
    for dir, folder, files in tf.io.gfile.walk(path):
        for fi in files:
            if (scorer != None) and (scorer+'_' in fi):
                list_of_files.append(os.path.join(dir, fi))
            if scorer == None:
                list_of_files.append(os.path.join(dir, fi))
    return list_of_files

def char_replace(array, old, new):
    '''Change characters of a string.'''
    
    new_array = np.char.replace(array, old, new)
    return new_array
# ··············································································


# ··············································································
# SWITCH FUNCTIONS
# ··············································································
def switch_events (type, resp_variations=None):
    ''' Switch the names contained in a predefined list to its default names. 
        · type: event name
        · resp_variations: amount of respiratory categories to be differenciated. 
            · if "partial": the different categories are grouped into three subcategories.
            · if "None": all the categories are distinguished.'''
    
    if resp_variations == 'partial':
        return {'arousal': ['EEG arousal'],
                'resp':['NAN', 'Apnea', 'Hypopnea'], 
                'desat':['SpO2 desaturation'],
                'limb':['Limb movement'],
                'stage': ['Sleep stage W', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R']}.get(type) 
    else: 
        return {'arousal': ['EEG arousal'],
                'resp':['NAN', 'Apnea', 'Obstructive apnea', 'Central apnea', 'Mixed apnea', 'Hypopnea', 'Obstructive hypopnea', 'Central hypopnea'], 
                'desat':['SpO2 desaturation'],
                'limb':['Limb movement'], 
                'stage': ['Sleep stage W', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R']}.get(type)

def switch_channels (type):
    '''Switch the channels to be used to their original names.'''

    return {'c4':['eeg1', 'eeg2', 'eog', 'emg'], 
            'c5':['eeg1', 'eeg2', 'eog', 'emg', 'saturation'], 
            'c6':['eeg1', 'eeg2', 'eog', 'emg', 'saturation', 'airflow'], 
            'c8':['eeg1', 'eeg2', 'eog', 'emg', 'saturation', 'airflow', 'abdores', 'thores']}.get(type)
# ··············································································


# ··············································································
# DATA PROCESSING 
#   · Signals extraction
#   · Signals and annotations resampling
#   · Lights off and lights on extraction and signals filtration
#   · Windowing (before lights on and lights off filtering)
#   · Link annotations to windows
#   · Signals and annotations normalization
#   · Create output vectors for each window:
#     [event presence/absence, sleep stages classification, init position, end position]
# ··············································································
def get_signals (edf_file, db_name, channels_name, config_file):
    '''Returns a list with channels, samples rates and signals. 
    Args:
        · edf_file: signal to be processed
        · db_name: name of the database 
        · channels_name: name of input channels → list format
        · config_file: db configuration file, montages and signal settings'''

    # ··· Read and get channels in edf file
    reader = edflib.EdfReader(edf_file)
    channels = reader.getSignalLabels()

    # ··· Get xml tree
    tree = ET.parse(config_file)
    root = tree.getroot()
    
    # ··· Tree iteration
    for montage in root:

        if montage[0].text == db_name:
            arr = [] 
            counter = 0
            list_of_warnings = []

            for channel in channels_name: 
                dev_name = montage.find('channelmapping').find(channel).get('label') 
                list_of_channels = montage.find('channelmapping').find(channel).findall('channel')
                n_dev = len(list_of_channels)
                signals = []

                for dev in range(n_dev):
                    sign = list_of_channels[dev].attrib['sign'] 
                    signal = reader.readSignal(channels.index(list_of_channels[dev].attrib['label'])) 
                    sr = reader.samples_in_datarecord(channels.index(list_of_channels[dev].attrib['label']))/reader.datarecord_duration 
                    if sign == 'minus':
                        signal = np.negative(signal)
                    signals.append(signal)
                if n_dev == 1:
                    derivation = signals[0] 
                else: 
                    derivation = sum(signals)
                    
                # ::: RAISE WARNING :::
                if all(x == 0 for x in derivation) == True:
                    counter += 1
                    list_of_warnings.append([edf_file, dev_name])
                    print('\x1b[1;37;43m'+' *** WARNING *** '+'\x1b[0m'+'   This derivation contains only zero values.') 

                arr.append([dev_name, sr, derivation])
    reader.close()
    
    return arr

def resample_signals (signals_array, hz):
    '''Resample signals in a preselected frequency (hz). Returns a numpy.array with signals samples.
    Args:
        · signals_array: signals array from "get_signals()" function. Format: [channel, sample_rate, signal]
        · hz: resample frequency'''

    signals = []
    for channel in signals_array:
        rate = channel[1]
        if rate != hz:
            factor = hz / rate
            r_sig = resample(channel[2], int(np.trunc(len(channel[2])*factor)))
            signals.append(r_sig)
        else:
            factor = 1
            signals.append(channel[2]) 

    return np.array(signals)

def resample_annotations (annotations_path, hz):
    '''Resample annotations to vinculate index to sample.'''

    _, __, annot = highlevel.read_edf(annotations_path)
    annot = annot["annotations"]
    r_annot = []
 
    for a in annot:       
        # ·· Resample annotations
        start = int(a[0] * hz) 
        end = int(start + (a[1] * hz ))
        duration = int(a[1] * hz)
        if ((end-start) % 3000 != 0) and ('stage' in a[2]):
            print('*** Warning ***')
        r_annot.append([start, end, duration, a[2]])

    return r_annot

def get_lights_index (annotations):
    '''Get lights on and lights off from annotations.'''
    
    # ································
    def get_index_annot(annotations):
        '''Lights on and lights off annotations index'''

        for a in annotations[0]:
            if not isinstance(a, numbers.Number):
                index = annotations[0].index(a)
        return index
    # ································
    
    lights_on = None
    lights_off = None
    dicc = {}
    annot_index = get_index_annot(annotations)
    duration_index = annot_index - 1

    for a in annotations: 

        if a[annot_index].startswith('Lights on'):
            lights_on = a[0] 
        if a[annot_index].startswith('Lights off'):
            lights_off = a[0]

    if (lights_on == None) and (lights_off == None):
        for a in annotations:
            if 'sleep stage' in a[annot_index].lower():
                dicc[a[0]] = a[duration_index] 
        
        lights_on = max(dicc.keys())
        lights_on = lights_on + dicc[lights_on] 

        lights_off = min(dicc.keys())
    
    return lights_on, lights_off

def apply_light_filter (idx, slices, onset, offset):
    '''Remove lights off and lights on period in slices.'''
    
    init_mask = idx[:, 0] < onset
    end_mask = idx[:, 1] > offset
    mask = np.logical_not(np.logical_or(init_mask, end_mask))

    idx_filtered = idx[mask]
    slices_filtered = slices[mask]

    return idx_filtered, slices_filtered

# ·················
# SLICING
# ·················
def get_slices (signals, window, hz, overlap=0):
    '''Get windows and indexes with or without overlapping.
    Args:
        signals: signals derived from resample_signals function
        hz: resample frequency
        window: window size
        overlap: overlapping among slices.'''

    n_channels, n_samples = np.shape(signals)

    window_size = window * hz 
    overlap_size = overlap * hz 

    step_size = math.floor(window_size - overlap_size)

    n_windows = math.floor((n_samples - window_size) / step_size)

    idx = np.zeros((n_windows, 2), dtype = int) 
    slices = np.zeros((n_windows, n_channels, window_size))

    for i in range(n_windows):
        start = i * step_size 
        end = start + window_size - 1 
        idx[i] = [start, end]

        for j in range(n_channels):
            slices[i][j] = signals[j][start:end+1] 
    
    return idx, slices

def extend_windows (default_index, signals, window, hz, seconds_before=0, seconds_after=0): 
    '''Include contextual information for each window.'''

    n_windows, _ = np.shape(default_index)
    n_channels, n_samples = np.shape(signals) 
    
    window_samples = window * hz
    previous_samples = seconds_before * hz
    subsequent_samples = seconds_after * hz
    window_size = window_samples + previous_samples + subsequent_samples

    idx = np.zeros((n_windows, 2), dtype = int) 
    slices = np.zeros((n_windows, n_channels, window_size))

    for n in range(n_windows):
        start = default_index[n][0] - previous_samples # extended window start index
        end = default_index[n][1] + subsequent_samples # extended window end index
        idx[n] = [start, end] # extended window indexes
        
        for j in range(n_channels):
            if start < 0: 
                start_position = abs(start)  
                slices[n][j][start_position:] = signals[j][0:end+1]

            elif end > n_samples:
                aux = signals[j][start:n_samples+1]
                slices[n][j][:len(aux)] = aux
            else:
                slices[n][j] = signals[j][start:end+1]

    return idx, slices

def get_slices_annotations (idx, annotations):
    ''' Link annotations to slices. Takes into account complete contained annotations, partial contained annotations
    and extended contained annotations. Returns a slice number key dicc with its corresponding annotations.
    Postprocessing is done at the end to remove duplicated or wrong annotations.
    Args:
        · idx: init and end slice indexes
        · annotations: annotations edf file'''
 
    # ··········································
    def process_annotations (dicc, idx):
        '''Remove duplicated annotations and selects one sleep stage per slice.
        Args:
            dicc: slice number and annotations
            idx: slice init and end indexes'''

        for key, value in dicc.items():
            stages = [] 
            for i in value.copy(): # annotation in slice 
                centroid = int(np.ceil(i[0] + (i[1]-i[0])/2))
                if centroid not in range(idx[key][0], idx[key][1]) and ('stage' not in i[3]): 
                    value.remove(i) # remove annotations which centroid is not in window range
                if 'stage' in i[3]: # i[3] = annotation title
                    stages.append(i)  
            if len(stages) > 1: # if there are more than 1 sleep stage annotations in the slice
            # we need to select one sleep stage per window slice, some of them have two stages linked
                for item in stages:
                    if item[0] in range(idx[key][0], idx[key][1]):
                        size = idx[key][1] - item[0]
                        i_size = (idx[key][1] - idx[key][0]) - size
                        if size < i_size:
                            value.remove(item)
                    elif item[0] not in range(idx[key][0], idx[key][1]):
                        size = item[1] - idx[key][0]
                        i_size = (idx[key][1] - idx[key][0]) - size
                        if size <= i_size:
                            value.remove(item)
                    elif (item[1] == idx[key][0]) or (item[0] == idx[key][1]): # if annotation end equals init slice, remove (belongs to other window)
                        value.remove(item)
        return dicc 
    # ··········································

    dicc = {}
    for num, sig in enumerate(idx):
        init = idx[num, 0]
        end = idx[num, 1]
        dicc[num] = [] 
        for annot in annotations: 
            if annot[0] >= init and annot[1] <= end:
                dicc[num].append(annot) 
            if (annot[0] < init and annot[1] in range(init,end)) or (annot[1] > end and annot[0] in range(init,end)):
                dicc[num].append(annot)  
            if (annot[0] < init and annot[1] > end) and (a in b for a, b in zip(list(range(init,end)), list(range(annot[0], annot[1])))):
                dicc[num].append(annot) 
    
    dicc = process_annotations(dicc, idx)  
    return dicc

# ·················
# NORMALIZATION
# ·················
def normalize_signals (signals):
    ''' Normalize signals along channels. '''

    norm_slices = []
    scaler = StandardScaler()

    for window in signals:
        norm_w_T = scaler.fit_transform(window.T) # scales along the features axis (column-wise)
        # transpose the window to scale along the channels
        norm_window = norm_w_T.T # rescale our window
        norm_slices.append(norm_window)

    return norm_slices

def normalize_annotations (annotations, idx, n_samples):
    '''Normalize annotations using windows indexes as reference.
    zi = (xi – min(x)) / (max(x) – min(x)) where: 
        zi: The ith normalized value in the dataset
        xi: The ith value in the dataset
        min(x): The minimum value in the dataset
        max(x): The maximum value in the dataset
    Args:
        · annotations: annotations per slice
        · idx: windows start and end indexes
        · n_samples: amount of samples in each window'''
    
    norm_annot = {} 

    for key, annotations in annotations.items():
        norm_annot[key] = []
        for i in annotations:
            norm_start = (i[0] - idx[key][0]) / n_samples # annotation start
            norm_end = (i[1] - idx[key][0]) / n_samples # annotation end
            norm_dur = i[2] / n_samples # annotation duration
            norm_annot[key].append([norm_start, norm_end, norm_dur, i[3]])

    return norm_annot

# ·················
# VECTORIZATION
# ·················
def get_stages_classification (event, events_list=None):
    '''Returns event type classification'''
    
    # ·· Default sleep stages list
    if events_list == None:
        events_list = ['W', '1', '2', '3', 'R']

    clas = []
    event_location = events_list.index(event)
    for i in range(len(events_list)):
        events_vector = np.zeros(len(events_list), dtype = int)
        events_vector[i] = 1 
        clas.append(events_vector) 
        
    return clas[event_location]

def get_events_position (events_list):
    '''Returns a list of 2 positions for each element (init and end)'''
    positions = []
    for i in range(len(events_list)):
        idx = [i*2 + j for j in range(2)]
        positions.append(idx)
    return positions

def get_vector_len (events_list, resp_variations=None):
    '''Calculate how many classification positions are needed to create the classification vector'''
    vector_len = 0

    if resp_variations == 'partial':
        variations = 3 # apnea, hypopnea, nan 
    else: 
        variations = 8 # 7 subcategories + nan

    for event in events_list:
        if event == 'resp': # only resp events have more positions than presence/ausence
            vector_len += variations # one for apnea, one for hypopnea, one for nan
        else:
            vector_len += 1 # presence

    vector_len += 5 # add 5 positions for sleep stages
    return vector_len

def link_events (events_list, resp_variations=None):
    '''Return two lists, one with events real names and other with each corresponding regression positions'''

    idx_reg_pos = [] 
    events_names = []

    for event in events_list: 
        e_name = switch_events(event, resp_variations)
        events_names.append(e_name) 

    for num, eve in enumerate(events_names):
        for e in eve:
            idx_reg_pos.append(num)

    events_names = [item for sublist in events_names for item in sublist]
    
    return idx_reg_pos, events_names 

def vectorize_events (annotations, events_list, resp_variations=None): 
    '''Get event vectors by window.
    Args:
        annotations: annotations by window, dictionary format
        events_list: events to localize, list format
        resp_variations: amount of respiratory events 'partial' or None
    Returns output vectors, event_names and position indexes'''

    vector_len_clas = get_vector_len(events_list, resp_variations) # get vector length based on the number of respiratory variations considered
    idx_reg_pos, events_names = link_events(events_list, resp_variations) 

    dicc = {}

    for key, value in annotations.items():
        vector_clas = np.zeros(vector_len_clas, dtype = int) 
        vector_reg = np.zeros((2 * len(events_list)), dtype = float) 
        vector_reg_pos = get_events_position(events_list) 

        value = sorted(value, key=lambda annot: annot[2], reverse=False) 

        for annot in value:
            for event, pos_reg in zip(events_names, idx_reg_pos):
                if (event in annot[3]) or (event.lower() in annot[3]):
                    vector_clas[events_names.index(event)] = 1 
                    vector_reg[vector_reg_pos[pos_reg][0]] = annot[0] 
                    vector_reg[vector_reg_pos[pos_reg][1]] = annot[1] 

            if 'stage' in annot[3]: 
                vector_clas[-5:] = get_stages_classification(annot[3][-1]) 

        if 'NAN' in events_names:
            nan_position = events_names.index('NAN')
            resp_array = vector_clas[nan_position:-5] 
            if resp_array.any() == 1:
                pass
            else:
                vector_clas[nan_position] = 1

        dicc[key] = np.concatenate((vector_clas, vector_reg), axis = None)

    return dicc, events_names, idx_reg_pos
# ··············································································


# ··············································································
# CREATE TFRs
# ··············································································
def create_tfr (signals, annotations, events_list, path, db_name, signal_id, resp_variations=None, scorer_id=None):
    '''Creates TFR files from a list of signals and an annotations dictionary.
    Path is needed to save files in a preselected directory.'''

    def create_example(signal, annotation, events_list, resp_variations): 
        '''Builds tensorflow TFR example using bytes default format.'''
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        feature = {}
        
        events_names = [switch_events(e, resp_variations) for e in events_list] 
        events_names = [item for sublist in events_names for item in sublist]
        
        # Signal feature
        signal_feature = _bytes_feature(tf.io.serialize_tensor(signal))
        feature["signal"] = signal_feature

        # Stages feature
        ini_stage = len(events_names) 
        end_stage = ini_stage + 5 
        stage = np.asarray(annotation[ini_stage:end_stage], dtype='int64')
        stage_feature = _bytes_feature(tf.io.serialize_tensor(stage))
        feature["stage_label"] = stage_feature 
 
        step = 0 

        for idx, event in enumerate(events_list):
            if event == 'resp':
                num_resp_var = len(switch_events(event, resp_variations))
                if idx == 0:
                    eve = np.asarray(annotation[idx:num_resp_var], dtype='int64') 
                else:
                    eve = np.asarray(annotation[idx:num_resp_var+1], dtype='int64') 
            else:
                eve = np.asarray(annotation[idx], dtype='int64')   

            # Event feature 
            event_feature = _bytes_feature(tf.io.serialize_tensor(eve))
            
            # Event position 
            ini_position = len(events_names) + 5 + idx + step           
            end_position = ini_position + 2 
            position = np.asarray(annotation[ini_position:end_position], dtype='float64') 
            position_feature = _bytes_feature(tf.io.serialize_tensor(position))

            feature[event+"_label"] = event_feature
            feature[event+"_position"] = position_feature
            step += 1 

        return tf.train.Example(features=tf.train.Features(feature=feature))

    for key, value in annotations.items():
        example = create_example(signals[key], value, events_list, resp_variations) 
        if scorer_id != None:
            filename = os.path.join(path, (db_name + '_' + signal_id + '_' + scorer_id + '_' + str(key)) + '.tfrecord')  
        else:
            filename = os.path.join(path, (db_name + '_' + signal_id + '_' + str(key)) + '.tfrecord')
        tf.io.TFRecordWriter(filename).write(example.SerializeToString()) 
# ··············································································


# ··············································································
# DATASETS 
#   · Split data into train, validation and test datasets
#   · Create tfr dataset generators
# ··············································································
def split_dataset (files, seed, ts_size=0.2, shuffle=True):
    '''Gives train and test datasets shuffled with same random_state=seed,
    proportion 80-20 taken into account classes distribution with stratify parameter.''' 

    train, test = sklearn.model_selection.train_test_split(files, test_size=ts_size, shuffle=shuffle, random_state=seed) 
    
    return train, test

def save_datasets (path, train, validation, test, scorer=None):
    '''Save datasets.
        · path: directory for saving
        · train, validation, test: datasets
        · scorer: if we consider a specific annotator.'''    

    if scorer != None:
        np.save(os.path.join(path, 'train_dataset_'+scorer+'.npy'), train)
        np.save(os.path.join(path, 'validation_dataset_'+scorer+'.npy'), validation)
        np.save(os.path.join(path, 'test_dataset_'+scorer+'.npy'), test)
    else:
        np.save(os.path.join(path, 'train_dataset.npy'), train)
        np.save(os.path.join(path, 'validation_dataset.npy'), validation)
        np.save(os.path.join(path, 'test_dataset.npy'), test)

def load_datasets(path, scorer=None):
    '''Load datasets to train, validate and test a model. 
        · paht: directory 
        · scorer: if we consider a specific annotator.'''

    if scorer != None:
        train = np.load(os.path.join(path, 'train_dataset_'+scorer+'.npy'))
        validation = np.load(os.path.join(path, 'validation_dataset_'+scorer+'.npy'))
        test = np.load(os.path.join(path, 'test_dataset_'+scorer+'.npy'))
    else: 
        train = np.load(os.path.join(path, 'train_dataset.npy'))
        validation = np.load(os.path.join(path, 'validation_dataset.npy'))
        test = np.load(os.path.join(path, 'test_dataset.npy'))
    
    return train, validation, test            

# ·················
# DATASETS GENERATORS
# ·················
def parse_tfr(feature_description):
    '''Parse TFR and decodify depending on the feature description.
    Returns a dictionary with inputs (x) and outputs (y).'''

    def parse_example(record):
        example = tf.io.parse_single_example(record, feature_description)
        input_dicc = {}
        output_dicc = {}

        for key in feature_description: 
            if key == 'signal': 
                example[key] = tf.io.parse_tensor(example[key], out_type = tf.float64) 
                input_dicc[key] = example[key]
            if 'label' in key: 
                example[key] = tf.io.parse_tensor(example[key], out_type = tf.int64) 
                output_dicc[key] = example[key]
            if 'position' in key: 
                example[key] = tf.io.parse_tensor(example[key], out_type = tf.float64) 
                example[key] = tf.ensure_shape(example[key], (2,))
                output_dicc[key] = example[key]
        
        return input_dicc, output_dicc

    return parse_example

def get_feature_description(dataset):
    '''Returns the features description of a dataset.'''
    
    feature_description = {}
    keys_list = []

    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
    for key, items in example.features.feature.items():
        keys_list.append(key)
    for key in keys_list:
        feature_description[key] = tf.io.FixedLenFeature([], tf.string)
    
    return feature_description

def split_data(signal, split):
    '''Split signal for the time distribution block.'''

    signal = np.array(np.split(signal, split, axis=1))
    return signal

def process_data(signal, labels, split, num_input_channels, num_samples):
    '''Process data to generate the correct dimension for the LSTM.'''

    signal['signal'] = tf.py_function(split_data, inp=[signal['signal'], split], Tout=tf.float16)
    signal['signal'] = tf.ensure_shape(signal['signal'], (split, num_input_channels, num_samples)) 

    return (signal, labels)

def process_data_tensor(signal, split, num_input_channels, num_samples):
    
    signal = tf.py_function(split_data, inp=[signal, split], Tout=tf.float16)
    signal = tf.ensure_shape(signal, (split, num_input_channels, num_samples)) 

    return signal

def tfr_dataset_generator(data, buffer_size, batch_size, split, num_input_channels, num_samples, seed, catched=False, shuffle=False, feature=False):
   '''TFR dataset generator''' 

   dataset = tf.data.TFRecordDataset(data) # convert to tfr dataset
   feature_description = get_feature_description(dataset) # get tfr features description
   dataset = dataset.map(parse_tfr(feature_description), num_parallel_calls=tf.data.AUTOTUNE) # decode samples
   dataset = dataset.map(lambda signal, labels: process_data(signal, labels, split, num_input_channels, num_samples), num_parallel_calls=tf.data.AUTOTUNE) # split signals
   
   if catched:
       dataset = dataset.cache()

   if shuffle:
      dataset = dataset.shuffle(buffer_size, seed, reshuffle_each_iteration=True)
   
   dataset = dataset.batch(batch_size)
   dataset = dataset.prefetch(tf.data.AUTOTUNE)

   if feature:
       return dataset, feature_description
   else:
       return dataset
   
def tensor_dataset_generator(data, batch_size, split, num_input_channels, num_samples, catched=False):
     
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(lambda signal: process_data_tensor(signal, split, num_input_channels, num_samples), num_parallel_calls=tf.data.AUTOTUNE)
    if catched:
       dataset = dataset.cache()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
# ··············································································


# ··············································································
# ACCESS CONFIGURATION
# ··············································································
def get_model_configuration(data_name, num_stages_clas, num_resp_clas=None):
    '''Get metrics and loss used to train the model automatically depending on the events to be detected.'''

    ty = data_name.split('_') # data_name contains the name of the events to predict
    loss, metrics = {}, {} # build two dictionaries

    if len(ty) == 4:
        num_events = 0 
    else:
        num_events = len(ty[4:]) # number of events in data_name  

    # Define metrics and loss used
    for idx, name in enumerate(ty):
        
        if (idx == 0) or (idx == 1) or (idx == 2): 
            continue
        
        if name == 'stage': # we separate stages from events due to different loss (Categorical vs Binary)
            loss[name+"_label"] = keras.losses.CategoricalCrossentropy() 
            metrics[name+"_label"] = [keras.metrics.MeanAbsoluteError(), tfa.metrics.CohenKappa(num_classes=num_stages_clas, name='Cohen Kappa')] 
        
        else:
            if name == 'resp':
                loss[name+"_label"] = keras.losses.CategoricalCrossentropy()
                loss[name+"_position"] = keras.losses.MeanSquaredError()
                metrics[name+"_label"] = [keras.metrics.MeanAbsoluteError(), tfa.metrics.CohenKappa(num_classes=num_resp_clas, name='Cohen Kappa')]
                metrics[name+"_position"] = [keras.metrics.MeanSquaredError()]
            
            else:
                loss[name+"_label"] = keras.losses.BinaryCrossentropy()
                loss[name+"_position"] = keras.losses.MeanSquaredError()
                metrics[name+"_label"] = [keras.metrics.MeanAbsoluteError(), tfa.metrics.CohenKappa(num_classes=2, name='Cohen Kappa')]
                metrics[name+"_position"] = [keras.metrics.MeanSquaredError()]
    
    return num_events, loss, metrics

def get_num_resp_events(resp_variations=None):
    if resp_variations == 'partial':
        num_resp_clas = 3
    elif resp_variations == None:
        num_resp_clas = None
    else:
        num_resp_clas = 8
    
    return num_resp_clas

# ·················
# PSG CONFIGURATION
# ·················
def get_psg_information(edf_file):
    
    # -- Open edf reader
    reader = edflib.EdfReader(edf_file)

    file_duration = reader.file_duration
    start_time = reader.getStartdatetime()

    reader.close()

    return file_duration, start_time

def get_list_of_derivations (db_name, channels_name, config_file):

    # ··· Get xml tree
    tree = ET.parse(config_file)
    root = tree.getroot()
    list_of_derivations = []

    # ··· Tree iteration
    for montage in root:

        if montage[0].text == db_name: # select same montage as db_name
            for channel in channels_name: 
                dev_name = montage.find('channelmapping').find(channel).get('label')
                list_of_derivations.append(dev_name)
    return list_of_derivations

def get_signal_id(db_name, file):

    if db_name == 'HMCP':
        signal_id = file[0:3]
    if db_name == 'SHHS':
        file_names = file.split('.') 
        signal_id = file_names[0] 

    return signal_id

def get_scorer_id (db_name, file):

    if db_name == 'HMCP':
        scorer_idx = file.find('score') 
        extension_idx = file.find('edf')
        scorer_id = file[scorer_idx:extension_idx-1] 
    if db_name == 'SHHS':
        scorer_id = None

    return scorer_id
# ··············································································


# ··············································································
# MODEL EVALUATION
# ··············································································
def extract_predicted_labels (predictions, ty, num_classes=None, index=None, th=None):
    
    if index != None:
        predictions = predictions[index]

    # ···············
    if ty == 'sigmoid':
        boolean = np.array(tf.greater(predictions, th)) 
        int_labels = boolean.astype(int)
        predicted_labels = np.concatenate(int_labels, axis=0)
    # ···············
    if ty == 'softmax':
        max_pred = np.argmax(predictions, axis=1)
        predicted_labels = tf.keras.utils.to_categorical(max_pred, num_classes=num_classes)
    # ···············
        
    return predicted_labels

def extract_true_labels (dataset, index):
    '''Extract true labels from dataset'''
    true_labels = []

    for x, y in dataset:
        true_labels.append(y[index])
    
    true_labels = np.concatenate(true_labels, axis=0)

    return true_labels

def categorize_predictions (model, dataset, predictions, num_stages_clas=None, num_resp_clas=None, th=None):

    categorical_predictions = []
    true_labels =[] 
    indexes = {} 
    
    if len(model.output_names) == 1:
        categorical_predictions = extract_predicted_labels(predictions, ty='softmax')
        true_labels = extract_true_labels(dataset, model.output_names[0])
        
    else:
        for idx, out in enumerate(model.output_names):

            # Multi-classification
            if "label" in out:
                if ("stage" in out) or ("resp" in out):
                    # Select de number of classes 
                    if "stage" in out:
                        num_clas = num_stages_clas
                    else:
                        num_clas = num_resp_clas
                    # Extract categorical labels
                    pred_label = extract_predicted_labels(predictions, ty='softmax', num_classes=num_clas, index=idx)
                    true_label = extract_true_labels(dataset, out)
                    # Append transformed labels 
                    categorical_predictions.append(pred_label)
                    true_labels.append(true_label)
                
                # Binary classification
                else:
                    # Extract categorical labels
                    pred_label = extract_predicted_labels(predictions, ty='sigmoid', index=idx, th=th)
                    true_label = extract_true_labels(dataset, out)
                    # Transform arousal label array into an array of arrays [[]] to provide the same structure as resp and stages
                    b = []
                    v = []
                    for t, p in zip(true_label, pred_label):
                        b.append([t])
                        v.append([p])
                    # Append transformed labels
                    categorical_predictions.append(np.asarray(v))
                    true_labels.append(np.asarray(b))
            
            # Regression 
            if "position" in out:
                true_label = extract_true_labels(dataset, out)
                # Append labels
                categorical_predictions.append(predictions[idx])
                true_labels.append(true_label)
            
            indexes[out] = idx # model outputs indexes
    return indexes, categorical_predictions, true_labels 

# ·················
def get_sklearn_metrics (true_labels, pred_labels):
    '''Calculate accuracy, precision, recall, f1score y global mae using sklearn.'''
    
    acc = sklearn.metrics.accuracy_score(true_labels, pred_labels)
    precision = sklearn.metrics.precision_score(true_labels, pred_labels, average='macro', zero_division=np.nan)
    recall = sklearn.metrics.recall_score(true_labels, pred_labels, average='macro', zero_division=np.nan)
    f1_score = sklearn.metrics.f1_score(true_labels, pred_labels, average='macro', zero_division=np.nan)
    global_mae = sklearn.metrics.mean_absolute_error(true_labels, pred_labels, multioutput='uniform_average')
    print('accuracy: {0}\t precision: {1}\t recall: {2}\t f1_score: {3}\t global_mae: {4}'. format(acc, precision, recall, f1_score, global_mae))

    return {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1_score': f1_score, 'Global_MAE': global_mae}

def calculate_metrics (model, predictions, test_dataset, num_stages_clas, num_resp_clas, th):

    metrics = {}

    if len(model.output_names) == 1:
        pred_label = extract_predicted_labels(predictions, ty='softmax')
        true_label = extract_true_labels(test_dataset, model.output_names[0])
        
        print('\nSklearn metrics for {0} predictions: '.format(model.output_names[0]))
        metric = get_sklearn_metrics(true_label, pred_label)
        metrics[model.output_names[0]] = metric
    else:
        for idx, out in enumerate(model.output_names):
            print('\nOutput_name: ', out)
            if "label" in out:
                if ("stage" in out) or ("resp" in out):
                    print('* Softmax classification *')
                    if "stage" in out:
                        num_clas = num_stages_clas
                    else:
                        num_clas = num_resp_clas
                    print('Num classes: ', num_clas)
                    pred_label = extract_predicted_labels(predictions, ty='softmax', num_classes=num_clas, index=idx)
                    true_label = extract_true_labels(test_dataset, out)
                else:
                    print('* Sigmoid classification *')
                    pred_label = extract_predicted_labels(predictions, 'sigmoid', index=idx, th=th)
                    true_label = extract_true_labels(test_dataset, out)
                print('\nSklearn metrics for \x1b[1;37;49m{0}\x1b[0m predictions: '.format(out))
                metric = get_sklearn_metrics(true_label, pred_label)
                metrics[out] = metric
    return metrics

def calculate_specific_mae (dataset, n_events, true_labels, cat_preds, ind):

    if n_events == 1: 
        y_pred = cat_preds
        y_true = true_labels
        stages_mae_c = sklearn.metrics.mean_absolute_error(y_true, y_pred)
        print(' · Stages MAE: ', stages_mae_c)

    elif n_events == 2: # (stages + resp)
        if dataset.split('_')[-1] == 'resp': 
            y_pred = np.hstack((cat_preds[ind['resp_label']], cat_preds[ind['stage_label']], cat_preds[ind['resp_position']]))
            y_true = np.hstack((true_labels[ind['resp_label']], true_labels[ind['stage_label']], true_labels[ind['resp_position']]))
            # Array positions
            stage_id = np.array([3, 4, 5, 6, 7])
            resp_id_c = np.array([0, 1, 2])
            resp_id_bw = np.array([8, 9])
            resp_id_cbw = np.hstack((resp_id_c, resp_id_bw))
            # Remove position predictions if non-presence
            for array in y_pred:
                if array[resp_id_c[0]] == 1:
                    array[resp_id_bw] = 0
            # MAE metrics
            stages_mae_c = sklearn.metrics.mean_absolute_error(y_true[:,stage_id], y_pred[:,stage_id])
            resp_mae_c = sklearn.metrics.mean_absolute_error(y_true[:,resp_id_c], y_pred[:,resp_id_c])
            resp_mae_bw = sklearn.metrics.mean_absolute_error(y_true[:,resp_id_bw], y_pred[:,resp_id_bw])
            resp_mae_cbw = sklearn.metrics.mean_absolute_error(y_true[:,resp_id_cbw], y_pred[:,resp_id_cbw])
            print(' · Stages MAE: ', stages_mae_c)
            print(' · Respiratory MAE c: ', resp_mae_c)
            print(' · Respiratory MAE bw: ', resp_mae_bw)
            print(' · Respiratory MAE cbw: ', resp_mae_cbw)
        
        if dataset.split('_')[-1] == 'arousal': # (stages + arousal)
            y_pred = np.hstack((cat_preds[ind['arousal_label']], cat_preds[ind['stage_label']], cat_preds[ind['arousal_position']]))
            y_true = np.hstack((true_labels[ind['arousal_label']], true_labels[ind['stage_label']], true_labels[ind['arousal_position']]))
            # Array positions
            stage_id = np.array([1, 2, 3, 4, 5])
            arousal_id_c = np.array([0])
            arousal_id_bw = np.array([6, 7])
            arousal_id_cbw = np.hstack((arousal_id_c, arousal_id_bw))
            # Remove position predictions if non-presence
            for array in y_pred:
                if array[arousal_id_c] == 0:
                    array[arousal_id_bw] = 0
            # MAE metrics
            stages_mae_c = sklearn.metrics.mean_absolute_error(y_true[:,stage_id], y_pred[:,stage_id])
            arousal_mae_c = sklearn.metrics.mean_absolute_error(y_true[:,arousal_id_c], y_pred[:,arousal_id_c])
            arousal_mae_bw = sklearn.metrics.mean_absolute_error(y_true[:,arousal_id_bw], y_pred[:,arousal_id_bw])
            arousal_mae_cbw = sklearn.metrics.mean_absolute_error(y_true[:,arousal_id_cbw], y_pred[:,arousal_id_cbw])
            print(' · Stages MAE: ', stages_mae_c)
            print(' · Arousal MAE c: ', arousal_mae_c)
            print(' · Arousal MAE bw: ', arousal_mae_bw)
            print(' · Arousal MAE cbw: ', arousal_mae_cbw)

    elif n_events == 3: # (stages + arousal + resp)
        y_pred = np.hstack((cat_preds[ind['arousal_label']], cat_preds[ind['resp_label']], cat_preds[ind['stage_label']], cat_preds[ind['arousal_position']], cat_preds[ind['resp_position']]))
        y_true = np.hstack((true_labels[ind['arousal_label']], true_labels[ind['resp_label']], true_labels[ind['stage_label']], true_labels[ind['arousal_position']], true_labels[ind['resp_position']]))
        # Array positions
        stage_id = np.array([4, 5, 6, 7, 8])
        arousal_id_c = np.array([0])
        arousal_id_bw = np.array([9, 10])
        arousal_id_cbw = np.hstack((arousal_id_c, arousal_id_bw))
        resp_id_c = np.array([1, 2, 3])
        resp_id_bw = np.array([11, 12])
        resp_id_cbw = np.hstack((resp_id_c, resp_id_bw))
        # Remove position predictions if non-presence
        for array in y_pred:
            if array[arousal_id_c] == 0:
                array[arousal_id_bw] = 0
            if array[resp_id_c[0]] == 1:
                array[resp_id_bw] = 0
        # MAE metrics
        stages_mae_c = sklearn.metrics.mean_absolute_error(y_true[:,stage_id], y_pred[:,stage_id])
        arousal_mae_c = sklearn.metrics.mean_absolute_error(y_true[:,arousal_id_c], y_pred[:,arousal_id_c])
        arousal_mae_bw = sklearn.metrics.mean_absolute_error(y_true[:,arousal_id_bw], y_pred[:,arousal_id_bw])
        arousal_mae_cbw = sklearn.metrics.mean_absolute_error(y_true[:,arousal_id_cbw], y_pred[:,arousal_id_cbw])

        resp_mae_c = sklearn.metrics.mean_absolute_error(y_true[:,resp_id_c], y_pred[:,resp_id_c])
        resp_mae_bw = sklearn.metrics.mean_absolute_error(y_true[:,resp_id_bw], y_pred[:,resp_id_bw])
        resp_mae_cbw = sklearn.metrics.mean_absolute_error(y_true[:,resp_id_cbw], y_pred[:,resp_id_cbw])
        print(' · Stages MAE: ', stages_mae_c)
        print(' · Arousal MAE c: ', arousal_mae_c)
        print(' · Arousal MAE bw: ', arousal_mae_bw)
        print(' · Arousal MAE cbw: ', arousal_mae_cbw)
        print(' · Respiratory MAE c: ', resp_mae_c)
        print(' · Respiratory MAE bw: ', resp_mae_bw)
        print(' · Respiratory MAE cbw: ', resp_mae_cbw)

    print('Indexes: ', ind)
    print('Example predictions stack: ', y_pred[0])
    print('Example predictions with bw processed: ', y_pred[0])
    print('Example true labels stack: ', y_true[0])

    # Global MAE
    global_mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    print('\nGlobal Mean absolute error: ', global_mae)
# ·················
    
def print_evaluation(dicc, name):
    print('·································')
    print('{0} results'.format(name))
    for key, value in dicc.items():
        print('{0}:  {1}'.format(key, value))

def save_metrics_to_csv (metrics_name, tr_results, val_results, ts_results, sklearn_metrics):
    dicc = defaultdict(list)

    # · Keras metrics
    for key, value in chain(ts_results.items(), tr_results.items(), val_results.items()):
        dicc[key].append(value)
    # · Sklearn metrics
    for key, value in sklearn_metrics.items():
        for k, v in value.items():
            dicc[key+'_'+k.lower()].append(v)

    metrics_df = pd.DataFrame.from_dict(dicc, orient='index') 
    metrics_df.columns = ['test', 'train', 'validation']
    metrics_df.to_csv(metrics_name)
# ··············································································