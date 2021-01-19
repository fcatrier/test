#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

# -----------------------------------------------------------------------------
# Gestion de l'historique des apprentissages
# -----------------------------------------------------------------------------

import os
import sys
import pandas
import numpy


cur_dir = os.getcwd()
if cur_dir == 'C:\\Users\\T0042310\\MyApp\\miniconda3':
    sys.path.append('C:\\Users\\T0042310\\Documents\\Perso\\Py\\pythonProject\\test-master')
    py_dir = 'C:\\Users\\T0042310\\Documents\\Perso\\Py'
elif cur_dir == 'C:\\Users\\Frédéri\\PycharmProjects\\pythonProject':
    py_dir = 'C:\\Users\\Frédéri\\Py'
else:
    sys.path.append('E:\\Py\\pythonProject')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v7.6.5 for CUDA 10.1\\bin')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v8.0.3.33 for CUDA 10.1\\bin')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin')
    py_dir = 'E:\\Py'


import step2_dataset_prepare_target_data as step2
import step3_dataset_prepare_learning_input_data as step3
from model_manager import ModelManager
import learn_evaluate_results
import utils


def new_npy_idx(npy_path):
    idx_max = 0
    for root, dirs, files in os.walk(npy_path):
        for file in files:
            try:
                idx = int(file.split('_')[0])
            except:
                idx = 0
            #
            if idx > idx_max:
                idx_max = idx
    #
    idx_run = idx_max + 1
    return idx_run


def get_all_params():
    all_params = []
    for param in step2.get_param_list():
        all_params.append(param)
    for param in step3.get_param_list():
        all_params.append(param)
    #
    _modelManager = ModelManager()
    for param in utils.dict_get_param_list(_modelManager.get_properties()):
        all_params.append(param)
    #
    for metric in utils.dict_get_param_list(learn_evaluate_results.learning_metrics_template):
        all_params.append(metric)
    #
    for base_params in utils.dict_get_param_list(learn_evaluate_results.post_learning_metrics_template):
        for postfix in('val','test1','test2'):
            current = base_params
            current += '_'
            current += postfix
            all_params.append(current)
    #
    for obsolete_metrics in utils.dict_get_param_list(learn_evaluate_results.obsolete_metrics_for_backward_compatibility):
        all_params.append(obsolete_metrics)
    #
    return all_params


def npy_results(npy_path, start_idx=0):
    #
    df = pandas.DataFrame()
    #
    id_max_loop = new_npy_idx(npy_path)
    #
    param_list = get_all_params()
    #
    for idx_df_filename in range(start_idx, id_max_loop):
        pandas_idx = len(df)
        for param in param_list:
            try:
                hist = numpy.load(npy_path + '\\' + str(idx_df_filename) + '_hist_' + param + '.npy')
                df.loc[pandas_idx, param] = hist[0]
            except:
                # print("missing :",param,"for idx :",pandas_idx)
                pass
    #
    print(df.columns)
    print(df.head())
    try:
        #df = df.sort_values(by=['val_accuracy'], ascending=False)
        df.to_excel(npy_path + '\\' + 'df_results.xlsx')
    except Exception as exc:
        print("exception de type ", exc.__class__)
        # affiche exception de type  exceptions.ZeroDivisionError
        print("message", exc)
        # affiche le message associé à l'exception
        pass
    #
    return df
