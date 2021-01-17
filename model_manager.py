#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

import os
import sys

cur_dir = os.getcwd()
if cur_dir == 'C:\\Users\\T0042310\\MyApp\\miniconda3':
    sys.path.append('C:\\Users\\T0042310\\Documents\\Perso\\Py\\TF')
    py_dir = 'C:\\Users\\T0042310\\Documents\\Perso\\Py'
elif cur_dir == 'C:\\Users\\Frédéri\\PycharmProjects\\pythonProject':
    py_dir = 'C:\\Users\\Frédéri\\Py'
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


class cmodel_manager:
    #
    __model_dict = dict ([
        #
        ('model_architecture', 'None'),
        #
        ('conv1D_block1_filters', -1),
        ('conv1D_block1_kernel_size', -1),
        ('conv1D_block1_MaxPooling1D_pool_size', -1),
        #
        ('config_GRU_LSTM_units', -1),
        #
        ('config_Dense_units', -1),
        ('config_Dense_units2', -1),
        #
        ('dropout_rate', 0.0),
        ('optimizer_name', 'None'),
        ('optimizer_modif_learning_rate', 0.0),
        #
        ('input_features', -1),
        ('input_timesteps', -1),
        ('output_shape', -1),
        ('model_count_params', -1),
        ('X_train_params', -1),
        #
        ('fit_batch_size', 32),
        ('fit_epochs_max', 500),
        ('fit_earlystopping_patience', 100)
    ])
    #
    def __init__(self):
        pass
    #
    def get_properties(self):
        return self.__model_dict
    #
    def update_properties(self,model_dict):
        self.__model_dict = model_dict
    #
    def __create_model_Dense_Dense(self):
        #
        model = keras.Sequential()
        #
        # entrée du modèle
        model.add(keras.Input(shape=(self.__model_dict['input_timesteps'], self.__model_dict['input_features'])))
        model.add(keras.layers.Flatten())
        #
        model.add(keras.layers.Dense(self.__model_dict['config_Dense_units'], activation='relu'))
        model.add(keras.layers.Dropout(self.__model_dict['dropout_rate']))
        model.add(keras.layers.Flatten())
        #
        model.add(keras.layers.Dense(self.__model_dict['config_Dense_units2'], activation='relu'))
        model.add(keras.layers.Dropout(self.__model_dict['dropout_rate']))
        #
        # sortie des classes
        #
        model.add(keras.layers.Dense(self.__model_dict['output_shape'], activation='softmax'))
        return model
    #
    def __create_model_LSTM_Dense(self):
        #
        model = keras.Sequential()
        #
        # entrée du LSTM
        #
        model.add(keras.layers.LSTM(self.__model_dict['config_GRU_LSTM_units'], return_sequences=True,
                                    input_shape=(self.__model_dict['nput_timesteps'],
                                                 self.__model_dict['input_features'])))
        model.add(keras.layers.Dropout(self.__model_dict['dropout_rate']))
        #
        # ajout d'une couche Flatten intermédiaire pour ne pas avoir à gérer des soucis de
        # taille de données (=> à partir d'ici on est en 1D)
        #
        model.add(keras.layers.Flatten())
        #
        model.add(keras.layers.Dense(self.__model_dict['config_Dense_units'], activation='relu'))
        model.add(keras.layers.Dropout(self.__model_dict['dropout_rate']))
        #
        # sortie des classes
        #
        model.add(keras.layers.Dense(self.__model_dict['output_shape'], activation='softmax'))
        return model
    #
    def __create_model_Conv1D_Dense(self):
        #
        model = keras.Sequential()
        #
        model.add(Conv1D(filters=self.__model_dict['conv1D_block1_filters'],
                         kernel_size=self.__model_dict['conv1D_block1_kernel_size'],
                         activation='relu',
                         input_shape=(self.__model_dict['input_timesteps'],
                                      self.__model_dict['input_features'])))
        model.add(Dropout(self.__model_dict['dropout_rate']))
        #
        if self.__model_dict['conv1D_block1_MaxPooling1D_pool_size'] != 0:
            model.add(MaxPooling1D(self.__model_dict['conv1D_block1_MaxPooling1D_pool_size']))
            model.add(Dropout(self.__model_dict['dropout_rate']))
        #
        # ajout d'une couche Flatten intermédiaire pour ne pas avoir à gérer des soucis de
        # taille de données (=> à partir d'ici on est en 1D)
        #
        model.add(keras.layers.Flatten())
        #
        model.add(keras.layers.Dense(self.__model_dict['config_Dense_units'], activation='relu'))
        model.add(keras.layers.Dropout(self.__model_dict['dropout_rate']))
        #
        # sortie des classes
        #
        model.add(keras.layers.Dense(self.__model_dict['output_shape'], activation='softmax'))
        return model
    #
    def __create_optimizer(self):
        if self.__model_dict['optimizer_name'] == 'sgd':  # sgd
            learning_rate = 0.01 * self.__model_dict['optimizer_modif_learning_rate']
            return keras.optimizers.SGD(learning_rate)
        elif self.__model_dict['optimizer_name'] == 'adam':  # adam
            learning_rate = 0.001 * self.__model_dict['optimizer_modif_learning_rate']
            return keras.optimizers.Adam(learning_rate)
        else:
            raise ValueError('Unknown optimizer_choice')
    #
    def create_compile_model(self):
        #
        model = None
        if self.__model_dict['model_architecture'] == 'Conv1D_Dense':
            model = self.__create_model_Conv1D_Dense()
        elif  self.__model_dict['model_architecture'] == 'Dense_Dense':
            model = self.__create_model_Dense_Dense()
        elif  self.__model_dict['model_architecture'] == 'LSTM_Dense':
            model = self.__create_model_LSTM_Dense()
        else:
            raise ValueError('Unknown model name')
        #
        model.compile(optimizer=self.__create_optimizer(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        #
        self.__model_dict['model_count_params'] = model.count_params()
        #
        return model
    #
    # def fit(self):
    #     #
    #     model = self.create_compile_model()
    #     #
    #     callback = keras.callbacks.EarlyStopping(monitor='val_accuracy',
    #                                              patience=self.__model_dict['fit_earlystopping_patience'],
    #                                              restore_best_weights=True)
    #     history = model.fit(np_X_train, df_y_Nd_train,
    #                         self.__model_dict['fit_batch_size'],
    #                         shuffle=False,
    #                         epochs=self.__model_dict['model_fit_epochs_max'],
    #                         callbacks=[callback], verbose=1,
    #                         validation_data=(np_X_val, df_y_Nd_val))
    #     #
    #     train_loss = round(min(history.history['loss']), 3)
    #     val_loss = round(min(history.history['val_loss']), 3)
    #     train_accuracy = round(max(history.history['accuracy']), 2)
    #     val_accuracy = round(max(history.history['val_accuracy']), 2)
    #
    def save(self, npy_path_prefix):
        #
        # for key       in params_dict.keys():
        # for key_value in params_dict.values():
        for key, key_value in self.__model_dict.items():
            tmp = []
            tmp.append(key_value)
            numpy.save(npy_path_prefix + '_hist_' + key + '.npy',  tmp)
    #
    def get_param_list(self):
        param_list = []
        for key, key_value in self.__model_dict.items():
            param_list.append(key)
        return param_list
