#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:36:04 2024

@author: adriana
"""

# ··············································································
# DESCRIPTION 
# ··············································································
# Neural network architectures
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
import tensorflow as tf
import keras
# ··············································································

# ··············································································
# Keras functional architecture
# ··············································································
def create_cnn_functional(feature, num_input_channels, num_samples, num_stages_clas, num_resp_clas, num_event_clas, num_positions, filters, BATCH_NORM, DROP_OUT, cnn_model=False):
    '''Create convolutional neural network with three operational blocks.'''

    # Input tensor
    input_tensor = keras.Input(shape = (num_input_channels, num_samples, 1), name = 'signal') 

    # Convolutional block 
    n = 3 
    for block in range(1, n+1):
        if block == 1:
            conv1D = keras.layers.Conv1D(filters, kernel_size=100, strides=1, padding='same', trainable=True)(input_tensor)
        if block > 1:
            filters = filters * 2
            conv1D = keras.layers.Conv1D(filters, kernel_size=100, strides=1, padding='same', trainable=True)(pooling)

        activation = keras.layers.Activation('relu', trainable=True)(conv1D)
        if BATCH_NORM:
            batch_norm = keras.layers.BatchNormalization(trainable=True)(activation)
            pooling = keras.layers.AveragePooling2D(pool_size=[1,2], strides=[1,2], trainable=True)(batch_norm)            
        else:
            pooling = keras.layers.AveragePooling2D(pool_size=[1,2], strides=[1,2], trainable=True)(activation)
    
    # Output block 
    flatten = keras.layers.Flatten(trainable=True)(pooling)
    dense = keras.layers.Dense(50, activation='relu', trainable=True)(flatten)
    if DROP_OUT:
        drop_out = tf.keras.layers.Dropout(0.5)(dense)
        hidden = drop_out
    else:
        hidden = dense
    
    # Create cnn model  
    cnn = keras.Model(inputs=input_tensor, outputs=hidden)
    return cnn

def create_lstm_functional(cnn, feature, split, num_input_channels, num_samples, num_stages_clas, num_resp_clas, num_event_clas, num_positions, lstm_units):
    '''Create LSTM.'''

    # LSTM structure
    input_tensor =  keras.Input(shape = (split, num_input_channels, num_samples, 1), name = 'signal')
    time_distrib = keras.layers.TimeDistributed(cnn)(input_tensor)
    lstm_layer = keras.layers.LSTM(units=lstm_units, return_sequences=False)(time_distrib)
    
    # Output tensors
    output_tensors = []
    for key in feature: 
        if "label" in key:
            if "stage" in key:
                out_layer = keras.layers.Dense(num_stages_clas, activation = 'softmax', name=key)(lstm_layer) 
                output_tensors.append(out_layer)
            else:
                if "resp" in key:
                    out_layer = keras.layers.Dense(num_resp_clas, activation = 'softmax', name=key)(lstm_layer)
                    output_tensors.append(out_layer)
                else:
                    out_layer = keras.layers.Dense(num_event_clas, activation = 'sigmoid', name=key)(lstm_layer) 
                    output_tensors.append(out_layer) 
        
        if "position" in key:
            out_layer = keras.layers.Dense(num_positions, activation = 'linear', name=key)(lstm_layer) 
            output_tensors.append(out_layer)

    model = keras.Model(inputs=input_tensor, outputs=output_tensors)
    return model