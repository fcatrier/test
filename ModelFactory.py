#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

import os
import sys

cur_dir = os.getcwd()
if cur_dir == 'C:\\Users\\T0042310\\MyApp\\miniconda3':
    sys.path.append('C:\\Users\\T0042310\\Documents\\Perso\\Py\\TF')
    py_dir = 'C:\\Users\\T0042310\\Documents\\Perso\\Py'
else:
    sys.path.append('E:\\Py\\pythonProject')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v7.6.5 for CUDA 10.1\\bin')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v8.0.3.33 for CUDA 10.1\\bin')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin')
    py_dir = 'E:\\Py'



import arbo

import keras
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
import numpy


model_factory_params = {
    'model_architecture' : '',
    #
    'conv1D_block1_filters' : 0,
    'conv1D_block1_kernel_size' : 0,
    'conv1D_block1_MaxPooling1D_pool_size' : 0,
    #
    'config_GRU_LSTM_units' : 0,
    #
    'config_Dense_units' : 0,
    'config_Dense_units2' : 0,
    #
    'dropout_rate' : 0.0,
    'optimizer_name' : '',
    'optimizer_modif_learning_rate' : 0.0}

class CModelFactory:
    #
    __model_architecture = None
    #
    __input_features = -1
    __input_timesteps = -1
    __output_shape = -1
    #
    __model_dropout_rate = -1
    #
    __conv1D_block1_filters = -1
    __conv1D_block1_kernel_size = -1
    __conv1D_block1_MaxPooling1D_pool_size = -1
    #
    __config_GRU_LSTM_units = -1
    #
    __config_Dense_units = -1
    __config_Dense_units2 = -1
    #
    __optimizer_name = 'adam'
    __optimizer_modif_learning_rate = 1.0
    #
    __model_count_params = -1
    __X_train_params = -1
    #
    def __init__(self):
        self.__input_features = -1
        self.__input_timesteps = -1
        self.__output_shape = -1
        self.__dropout = -1
        #
        self.__conv1D_block1_filters = -1
        self.__conv1D_block1_kernel_size = -1
        self.__conv1D_block1_MaxPooling1D_pool_size = -1
        #
        self.__config_GRU_LSTM_units = -1
        #
        self.__config_Dense_units = -1
        self.__config_Dense_units2 = -1

    def set_params_inout(self, input_features, input_timesteps, output_shape, train_samples):
        self.__input_features = input_features
        self.__input_timesteps = input_timesteps
        self.__output_shape = output_shape
        self.__X_train_params = train_samples * input_features * input_timesteps

    def set_dropout_rate(self,model_dropout_rate):
        self.__model_dropout_rate = model_dropout_rate

    def set_params_Conv1D(self, conv1D_block1_filters, conv1D_block1_kernel_size, conv1D_block1_MaxPooling1D_pool_size):
        self.__conv1D_block1_filters = conv1D_block1_filters
        self.__conv1D_block1_kernel_size = conv1D_block1_kernel_size
        self.__conv1D_block1_MaxPooling1D_pool_size = conv1D_block1_MaxPooling1D_pool_size

    def set_params_GRU_LSTM(self, config_GRU_LSTM_units):
        self.__config_GRU_LSTM_units = config_GRU_LSTM_units

    def set_params_Dense(self, config_Dense_units, config_Dense_units2=-1):
        self.__config_Dense_units = config_Dense_units
        self.__config_Dense_units2 = config_Dense_units2

    def set_params_optimizer(self, optimizer_name, optimizer_modif_learning_rate):
        self.__optimizer_name = optimizer_name
        self.__optimizer_modif_learning_rate =optimizer_modif_learning_rate

    def __create_model_Dense_Dense(self):
        #
        model = keras.Sequential()
        #
        # entrée du modèle
        model.add(keras.Input(shape=(self.__input_timesteps, self.__input_features)))
        model.add(keras.layers.Flatten())
        #
        model.add(keras.layers.Dense(self.__config_Dense_units, activation='relu'))
        model.add(keras.layers.Dropout(self.__model_dropout_rate))
        model.add(keras.layers.Flatten())
        #
        model.add(keras.layers.Dense(self.__config_Dense_units2, activation='relu'))
        model.add(keras.layers.Dropout(self.__model_dropout_rate))
        #
        # sortie des classes
        #
        model.add(keras.layers.Dense(self.__output_shape, activation='softmax'))
        return model

    def __create_model_LSTM_Dense(self):
        #
        model = keras.Sequential()
        #
        # entrée du LSTM
        #
        model.add(keras.layers.LSTM(self.__config_GRU_LSTM_units, return_sequences=True,
                                    input_shape=(self.__input_timesteps, self.__input_features)))
        model.add(keras.layers.Dropout(self.__model_dropout_rate))
        #
        # ajout d'une couche Flatten intermédiaire pour ne pas avoir à gérer des soucis de
        # taille de données (=> à partir d'ici on est en 1D)
        #
        model.add(keras.layers.Flatten())
        #
        model.add(keras.layers.Dense(self.__config_Dense_units, activation='relu'))
        model.add(keras.layers.Dropout(self.__model_dropout_rate))
        #
        # sortie des classes
        #
        model.add(keras.layers.Dense(self.__output_shape, activation='softmax'))
        return model

    def __create_model_Conv1D_Dense(self):
        #
        model = keras.Sequential()
        #
        model.add(Conv1D(filters=self.__conv1D_block1_filters, kernel_size=self.__conv1D_block1_kernel_size,
                         activation='relu',
                         input_shape=(self.__input_timesteps, self.__input_features)))
        model.add(Dropout(self.__model_dropout_rate))
        #
        if self.__conv1D_block1_MaxPooling1D_pool_size != 0:
            model.add(MaxPooling1D(self.__conv1D_block1_MaxPooling1D_pool_size))
            model.add(Dropout(self.__model_dropout_rate))
        #
        # ajout d'une couche Flatten intermédiaire pour ne pas avoir à gérer des soucis de
        # taille de données (=> à partir d'ici on est en 1D)
        #
        model.add(keras.layers.Flatten())
        #
        model.add(keras.layers.Dense(self.__config_Dense_units, activation='relu'))
        model.add(keras.layers.Dropout(self.__model_dropout_rate))
        #
        # sortie des classes
        #
        model.add(keras.layers.Dense(self.__output_shape, activation='softmax'))
        return model

    def __create_optimizer(self):
        if self.__optimizer_name == 'sgd':  # sgd
            learning_rate = 0.01 * self.__optimizer_modif_learning_rate
            return keras.optimizers.SGD(learning_rate)
        elif self.__optimizer_name == 'adam':  # adam
            learning_rate = 0.001 * self.__optimizer_modif_learning_rate
            return keras.optimizers.Adam(learning_rate)
        else:
            raise ValueError('Unknown optimizer_choice')

    def create_compile_model(self, model_architecture):
        #
        model = None
        if model_architecture == 'Conv1D_Dense':
            model = self.__create_model_Conv1D_Dense()
        elif model_architecture == 'Dense_Dense':
            model = self.__create_model_Dense_Dense()
        elif model_architecture == 'LSTM_Dense':
            model = self.__create_model_LSTM_Dense()
        else:
            raise ValueError('Unknown model name')
        #
        self.__model_architecture = model_architecture
        #
        optimizer = self.__create_optimizer()
        #
        model.compile(optimizer=self.__create_optimizer(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        #
        self.__model_count_params = model.count_params()
        #
        return model

    def save_model(self, dataset_name, dir_npy, idx_run_loop):
        #
        hist_model_architecture = []
        hist_input_features = []
        hist_input_timesteps = []
        hist_output_shape = []
        hist_model_dropout_rate = []
        hist_conv1D_block1_filters = []
        hist_conv1D_block1_kernel_size = []
        hist_conv1D_block1_MaxPooling1D_pool_size = []
        hist_config_GRU_LSTM_units = []
        hist_config_Dense_units = []
        hist_config_Dense_units2 = []
        hist_optimizer_name = []
        hist_optimizer_modif_learning_rate = []
        hist_model_count_params = []
        hist_X_train_params = []
        #
        hist_model_architecture.append(self.__model_architecture)
        hist_input_features.append(self.__input_features)
        hist_input_timesteps.append(self.__input_timesteps)
        hist_output_shape.append(self.__output_shape)
        hist_model_dropout_rate.append(self.__model_dropout_rate)
        hist_conv1D_block1_filters.append(self.__conv1D_block1_filters)
        hist_conv1D_block1_kernel_size.append(self.__conv1D_block1_kernel_size)
        hist_conv1D_block1_MaxPooling1D_pool_size.append(self.__conv1D_block1_MaxPooling1D_pool_size)
        hist_config_GRU_LSTM_units.append(self.__config_GRU_LSTM_units)
        hist_config_Dense_units.append(self.__config_Dense_units)
        hist_config_Dense_units2.append(self.__config_Dense_units2)
        hist_optimizer_name.append(self.__optimizer_name)
        hist_optimizer_modif_learning_rate.append(self.__optimizer_modif_learning_rate)
        hist_model_count_params.append(self.__model_count_params)
        hist_X_train_params.append(self.__X_train_params)
        #
        path = arbo.get_study_dir(py_dir, dataset_name) + dir_npy + '\\' + str(idx_run_loop)
        #
        numpy.save(path + '_hist_model_architecture', hist_model_architecture)
        numpy.save(path + '_hist_input_features', hist_input_features)
        numpy.save(path + '_hist_input_timesteps', hist_input_timesteps)
        numpy.save(path + '_hist_output_shape', hist_output_shape)
        numpy.save(path + '_hist_model_dropout_rate', hist_model_dropout_rate)
        numpy.save(path + '_hist_conv1D_block1_filters', hist_conv1D_block1_filters)
        numpy.save(path + '_hist_conv1D_block1_kernel_size', hist_conv1D_block1_kernel_size)
        numpy.save(path + '_hist_conv1D_block1_MaxPooling1D_pool_size', hist_conv1D_block1_MaxPooling1D_pool_size)
        numpy.save(path + '_hist_config_GRU_LSTM_units', hist_config_GRU_LSTM_units)
        numpy.save(path + '_hist_config_Dense_units', hist_config_Dense_units)
        numpy.save(path + '_hist_config_Dense_units2', hist_config_Dense_units2)
        numpy.save(path + '_hist_optimizer_name', hist_optimizer_name)
        numpy.save(path + '_hist_optimizer_modif_learning_rate', hist_optimizer_modif_learning_rate)
        numpy.save(path + '_hist_model_count_params', hist_model_count_params)
        numpy.save(path + '_hist_X_train_params', hist_X_train_params)
