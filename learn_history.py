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
    sys.path.append('C:\\Users\\T0042310\\Documents\\Perso\\Py\\TF')
    py_dir = 'C:\\Users\\T0042310\\Documents\\Perso\\Py'
else:
    sys.path.append('E:\\Py\\pythonProject')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v7.6.5 for CUDA 10.1\\bin')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v8.0.3.33 for CUDA 10.1\\bin')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin')
    py_dir = 'E:\\Py'


import arbo
import step3_dataset_prepare_learning_input_data as step3


def new_npy_idx(dataset_name, dir_npy):
    idx_max = 0
    for root, dirs, files in os.walk(arbo.get_study_dir(py_dir, dataset_name) + dir_npy):
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

def run_loop_historize( # global parameters
                       dataset_name,
                       dir_npy,
                       idx_run_loop,
                       #
                       model_fit_batch_size,
                       model_fit_epochs_max,
                       model_fit_earlystopping_patience,
                       #
                       # training parameters
                       train_loss,
                       val_loss,
                       train_accuracy
                       val_accuracy):
    #
    hist_model_fit_batch_size = []
    hist_model_fit_epochs_max = []
    hist_model_fit_earlystopping_patience = []
    #
    hist_train_loss = []
    hist_val_loss = []
    hist_train_accuracy = []
    hist_val_accuracy = []
    #
    hist_model_fit_batch_size.append(model_fit_batch_size)
    hist_model_fit_epochs_max.append(model_fit_epochs_max)
    hist_model_fit_earlystopping_patience.append(model_fit_earlystopping_patience)
    #
    hist_train_loss.append(train_loss)
    hist_val_loss.append(val_loss)
    hist_train_accuracy.append(train_accuracy)
    hist_val_accuracy.append(val_accuracy)
    #
    path = arbo.get_study_dir(py_dir, dataset_name) + dir_npy + '\\' + str(idx_run_loop)
    #
    numpy.save(path + '_hist_model_fit_batch_size',             hist_model_fit_batch_size)
    numpy.save(path + '_hist_model_fit_epochs_max',             hist_model_fit_epochs_max)
    numpy.save(path + '_hist_model_fit_earlystopping_patience', hist_model_fit_earlystopping_patience)
    #
    numpy.save(path + '_hist_train_loss.npy',                   hist_train_loss)
    numpy.save(path + '_hist_val_loss.npy',                     hist_val_loss)
    numpy.save(path + '_hist_train_accuracy.npy',               hist_train_accuracy)
    numpy.save(path + '_hist_val_accuracy.npy',                 hist_val_accuracy)
    #


def npy_results(dataset_name, dir_npy, start_idx=0):
    #
    df = pandas.DataFrame()
    #
    id_max_loop = new_npy_idx(dataset_name, dir_npy)
    #
    param_list = []
    #
    _model_manager = model_manager()
    model_manager_param_list = _model_manager.get_param_list()
    for each_param in model_manager_param_list :
        param_list.append(each_param)
    #
    param_list = [
        'acc',
        'val_acc',
        'loss',
        'val_loss',
        'epochs_sum',
        'cm',
        'result',
        'res_eval_result_atr',
        'step2_target_class_col_name',
        'step2_profondeur_analyse',
        'step2_target_period',
        'step2_symbol_for_target',
        'step2_targets_classes_count',
        'step2_targetLongShort',
        'step2_ratio_coupure',
        'step3_recouvrement',
        'step3_samples_by_class',
        'step3_tests_by_class',
        'step3_time_depth',
        'step3_columns',
        'train_loss',
        'val_loss',
        'train_accuracy',
        'val_accuracy',
        'val_acc',
        'res_eval_result_atr_val',
        'cm_val',
        'pc_resultat_val',
        'test1_acc',
        'res_eval_result_atr_test1',
        'cm_test1',
        'pc_resultat_test1',
        'test2_acc',
        'res_eval_result_atr_test2',
        'cm_test2',
        'pc_resultat_test2']
    #
    for idx_df_filename in range(start_idx, id_max_loop):
        pandas_idx = len(df)
        for param in param_list:
            try:
                hist = numpy.load(arbo.get_study_dir(py_dir, dataset_name) + dir_npy + '\\' + str(
                    idx_df_filename) + '_hist_' + param + '.npy')
                df.loc[pandas_idx, param] = hist[0]
            except:
                # print("missing :",param,"for idx :",pandas_idx)
                pass
    #
    df = df.sort_values(by=['result'], ascending=False)
    try:
        df.to_excel(arbo.get_study_dir(py_dir, dataset_name) + dir_npy + '\\' + 'df_results.xlsx')
    except:
        pass
    #
    return df
