#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

import os
import sys
import pandas
import numpy
import sklearn
from sklearn.utils import shuffle
import keras


cur_dir = os.getcwd()
if cur_dir == 'C:\\Users\\T0042310\\MyApp\\miniconda3':
    sys.path.append('C:\\Users\\T0042310\\Documents\\Perso\\Py\\pythonProject\\test-master')
    py_dir = 'C:\\Users\\T0042310\\Documents\\Perso\\Py'
else:
    sys.path.append('E:\\Py\\pythonProject')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v7.6.5 for CUDA 10.1\\bin')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cuDNN\\cuDNN v8.0.3.33 for CUDA 10.1\\bin')
    sys.path.append('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin')
    py_dir = 'E:\\Py'


import arbo


# -----------------------------------------------------------------------------
# 1_dataset_prepare_raw_data
# -----------------------------------------------------------------------------

import step1_dataset_prepare_raw_data as step1

_dataset_name = 'work'

Usa500_dfH1, Usa500_dfM30, Usa500_dfM15, Usa500_dfM5, \
UsaInd_dfH1, UsaInd_dfM30, UsaInd_dfM15, UsaInd_dfM5, \
UsaTec_dfH1, UsaTec_dfM30, UsaTec_dfM15, UsaTec_dfM5, \
Ger30_dfH1, Ger30_dfM30, Ger30_dfM15, Ger30_dfM5, \
EURUSD_dfH1, EURUSD_dfM30, EURUSD_dfM15, EURUSD_dfM5, \
USDJPY_dfH1, USDJPY_dfM30, USDJPY_dfM15, USDJPY_dfM5, \
GOLD_dfH1, GOLD_dfM30, GOLD_dfM15, GOLD_dfM5, \
LCrude_dfH1, LCrude_dfM30, LCrude_dfM15, LCrude_dfM5, \
big_H1, big_M30, big_M15, big_M5 = step1.load_prepared_data(_dataset_name)

data_all_df_bigs = {'H1': [big_H1],
                    'M30': [big_M30],
                    'M15': [big_M15],
                    'M5': [big_M5]}
_all_df_bigs = pandas.DataFrame(data_all_df_bigs)

# -----------------------------------------------------------------------------
# step2_dataset_prepare_target_data
# step3_dataset_prepare_learning_input_data
# -----------------------------------------------------------------------------

import step2_dataset_prepare_target_data as step2
import step3_dataset_prepare_learning_input_data as step3

step2.step2_params['step2_target_class_col_name'] = 'target_class'
step2.step2_params['step2_profondeur_analyse'] = 3
step2.step2_params['step2_target_period'] = 'M15'

# paramètres spécifiques à 'generate_big_define_target'
step2.step2_params['step2_symbol_for_target'] = 'UsaInd'
step2.step2_params['step2_targets_classes_count'] = 3
step2.step2_params['step2_symbol_spread'] = 2.5
# step2.step2_params['step2_targetLongShort'] = 20.0
# step2.step2_params['step2_ratio_coupure'] = 1.3
# step2.step2_params['step2_use_ATR'] = False
step2.step2_params['step2_targetLongShort'] = 0.95
step2.step2_params['step2_ratio_coupure'] = 1.1
step2.step2_params['step2_use_ATR'] = True

output_step2_data_with_target = step2.prepare_target_data_with_define_target(
    _all_df_bigs,
    step2.step2_params['step2_profondeur_analyse'],
    step2.step2_params['step2_target_period'],
    step2.step2_params['step2_symbol_for_target'],
    step2.step2_params['step2_targetLongShort'],
    step2.step2_params['step2_ratio_coupure'],
    step2.step2_params['step2_targets_classes_count'],
    step2.step2_params['step2_target_class_col_name'],
    step2.step2_params['step2_use_ATR'])

#
# imports from this project
#
import learn_evaluate_results
import model_manager
from model_manager import model_manager

import learn_history


def create_step3_data(step3_params):
    #
    learning_data = learning_data_template.copy()
    #
    idx_generate_learning_data = step3_params['step3_idx_start']    # incrémenté à chaque appel à step3.generate_learning_data_dynamic3
    resOK_test2, idx_generate_learning_data, np_X_test2, df_y_Nd_test2, df_y_1d_test2, df_atr_test2 = step3.generate_learning_data_dynamic3(
        output_step2_data_with_target, idx_generate_learning_data,
        step3_params['step3_recouvrement'], step3_params['step3_tests_by_class'], step3_params['step3_time_depth'],
        step3_params['step3_column_names_to_scale'], step3_params['step3_column_names_not_to_scale'])
    #
    learning_data_base_test2 = learning_data_base_template.copy()
    learning_data_base_test2['np_X'] = np_X_test2
    learning_data_base_test2['df_y_Nd'] = df_y_Nd_test2
    learning_data_base_test2['df_y_1d'] = df_y_1d_test2
    learning_data_base_test2['df_atr'] = df_atr_test2
    learning_data['test2'] = learning_data_base_test2
    #
    if resOK_test2 == False :
        raise RuntimeError('Error during step3.generate_learning_data_dynamic3 for test2')
    #
    resOK_test1, idx_generate_learning_data, np_X_test1, df_y_Nd_test1, df_y_1d_test1, df_atr_test1 = step3.generate_learning_data_dynamic3(
        output_step2_data_with_target, idx_generate_learning_data,
        step3_params['step3_recouvrement'], step3_params['step3_tests_by_class'], step3_params['step3_time_depth'],
        step3_params['step3_column_names_to_scale'], step3_params['step3_column_names_not_to_scale'])
    #
    learning_data_base_test1 = learning_data_base_template.copy()
    learning_data_base_test1['np_X'] = np_X_test1
    learning_data_base_test1['df_y_Nd'] = df_y_Nd_test1
    learning_data_base_test1['df_y_1d'] = df_y_1d_test1
    learning_data_base_test1['df_atr'] = df_atr_test1
    learning_data['test1'] = learning_data_base_test1
    #
    if resOK_test1 == False :
        raise RuntimeError('Error during step3.generate_learning_data_dynamic3 for test1')
    #
    resOK_val, idx_generate_learning_data, np_X_val, df_y_Nd_val, df_y_1d_val, df_atr_val = step3.generate_learning_data_dynamic3(
        output_step2_data_with_target, idx_generate_learning_data,
        step3_params['step3_recouvrement'], step3_params['step3_tests_by_class'], step3_params['step3_time_depth'],
        step3_params['step3_column_names_to_scale'], step3_params['step3_column_names_not_to_scale'])
    #
    np_X_val, df_y_1d_val, df_y_Nd_val = sklearn.utils.shuffle(np_X_val, df_y_1d_val, df_y_Nd_val)
    #
    learning_data_base_val = learning_data_base_template.copy()
    learning_data_base_val['np_X'] = np_X_val
    learning_data_base_val['df_y_Nd'] = df_y_Nd_val
    learning_data_base_val['df_y_1d'] = df_y_1d_val
    learning_data_base_val['df_atr'] = df_atr_val
    learning_data['val'] = learning_data_base_val
    #
    if resOK_val == False :
        raise RuntimeError('Error during step3.generate_learning_data_dynamic3 for val')
    #
    resOK_train, idx_generate_learning_data, np_X_train, df_y_Nd_train, df_y_1d_train, df_atr_train = step3.generate_learning_data_dynamic3(
        output_step2_data_with_target, idx_generate_learning_data,
        step3_params['step3_recouvrement'], step3_params['step3_tests_by_class'], step3_params['step3_time_depth'],
        step3_params['step3_column_names_to_scale'], step3_params['step3_column_names_not_to_scale'])
    #
    np_X_train, df_y_1d_train, df_y_Nd_train = sklearn.utils.shuffle(np_X_train, df_y_1d_train, df_y_Nd_train)
    #
    learning_data_base_train = learning_data_base_template.copy()
    learning_data_base_train['np_X'] = np_X_train
    learning_data_base_train['df_y_Nd'] = df_y_Nd_train
    learning_data_base_train['df_y_1d'] = df_y_1d_train
    learning_data_base_train['df_atr'] = df_atr_train
    learning_data['train'] = learning_data_base_train
    #
    if resOK_train == False :
        raise RuntimeError('Error during step3.generate_learning_data_dynamic3 for train')
    #
    return learning_data



def post_learning_metrics(model, learning_data, train_val_test):
    #
    np_X = learning_data[train_val_test]['np_X']
    df_y_1d = learning_data[train_val_test]['df_y_1d']
    df_atr= learning_data[train_val_test]['df_atr']
    #
    y_pred_raw = model.predict(np_X)  # avec les poids sortie modèle
    df_y_pred = pandas.DataFrame(numpy.argmax(y_pred_raw, axis=-1))  # avec les poids forcés à 0/1
    df_y_pred.index = df_y_1d.index
    #
    acc, res_eval_result_atr, cm, pc_resultat = learn_evaluate_results.evaluate_atr(
        df_y_pred, df_y_1d, df_atr,
        step2.step2_params['step2_symbol_spread'], step2.step2_params['step2_ratio_coupure'])
    print("--- results analysis for ",train_val_test)
    print('res_eval_result_atr = ', res_eval_result_atr, '\tpc_resultat =', pc_resultat)
    print('acc = ', acc)
    print(cm)
    learn_evaluate_results.cm_metrics(cm)
    print("---")
    #
    result = post_learning_metrics_template.copy()
    result['acc'] = acc
    result['res_eval_result_atr'] = res_eval_result_atr
    result['cm'] = cm
    result['pc_resultat'] = pc_resultat
    return result


def learn_from_step3(step3_params, _modelManager, loops_count=1):
    #
    idx_run_loop = learn_history.new_npy_idx(_dataset_name, _dir_npy)
    #
    try:
        learning_data = create_step3_data(step3_params)
    except:
        print("step3.generate_learning_data_dynamic3 failed. STOP")
        return
    #
    # Setting of last model manager properties that need information about data generation
    #
    input_features = learning_data['train']['np_X'].shape[2]
    input_timesteps = learning_data['train']['np_X'].shape[1]
    output_shape = learning_data['train']['df_y_Nd'].shape[1]
    train_params = learning_data['train']['np_X'].shape[0]
    print("input_features=", input_features)
    print("input_timesteps=", input_timesteps)
    print("output_shape=", output_shape)
    print("train_params=", train_params)
    #
    _mm_dict = _modelManager.get_properties()
    #
    _mm_dict['input_features'] = input_features
    _mm_dict['input_timesteps'] = input_timesteps
    _mm_dict['output_shape'] = output_shape
    _mm_dict['train_params'] = train_params
    #
    _modelManager.update_properties(_mm_dict)
    #
    for i in range(0, loops_count):
        print("")
        print("-----------------------------------------------------------")
        print("Repeat #", i + 1, " /", loops_count)
        print("-----------------------------------------------------------")
        print("")
        #
        model = _modelManager.create_compile_model()
        #
        if (i == 0):  # affiché uniquement au premier passage pour désaturer l'affichage
            print(model.summary())
            print("model.count_params()=", model.count_params())
            print("train_params : ", train_params)
        #
        callback = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                 patience=_mm_dict['fit_earlystopping_patience'],
                                                 restore_best_weights=True)
        history = model.fit(learning_data['train']['np_X'],
                            learning_data['train']['df_y_Nd'],
                            _mm_dict['fit_batch_size'],
                            shuffle=False,
                            epochs=_mm_dict['fit_epochs_max'],
                            callbacks=[callback],
                            verbose=1,
                            validation_data=(learning_data['val']['np_X'],
                                             learning_data['val']['df_y_Nd']))
        #
        train_loss = round(min(history.history['loss']), 3)
        val_loss = round(min(history.history['val_loss']), 3)
        train_accuracy = round(max(history.history['accuracy']), 2)
        val_accuracy = round(max(history.history['val_accuracy']), 2)
        print("train_loss     = ", train_loss)
        print("val_loss       = ", val_loss)
        print("train_accuracy = ", train_accuracy)
        print("val_accuracy   = ", val_accuracy)
        print("---")
        #
        post_learning_metrics_val   = post_learning_metrics(model, learning_data, 'val')
        post_learning_metrics_test1 = post_learning_metrics(model, learning_data, 'test1')
        post_learning_metrics_test2 = post_learning_metrics(model, learning_data, 'test2')
        #
        path = npy_path_with_prefix(dataset_name, dir_npy, idx_run_loop)
        #
        dictionary_save(path, model_manager.get_properties())
        dictionary_save(path, step2.step2_params)
        dictionary_save(path, step3.step3_params)
        dictionary_save(path, post_learning_metrics_val,   'val')
        dictionary_save(path, post_learning_metrics_test1, 'train1')
        dictionary_save(path, post_learning_metrics_test2, 'train2')
        #
        idx_run_loop += 1
        #
        del model
    #


#
# Ici point d'entrée pour ajuster les paramètres ou coder les boucles de variations
#
def execute():
    #
    _modelManager = ModelManager()
    #
    # step3 parameters : unchanged during loop
    #
    step3.step3_params['step3_column_names_to_scale'] = []
    step3.step3_params['step3_column_names_not_to_scale'] = [
        'UsaInd_M15_time_slot',
        'UsaInd_M15_pRSI_3', 'UsaInd_M15_pRSI_5', 'UsaInd_M15_pRSI_8', 'UsaInd_M15_pRSI_13', 'UsaInd_M15_pRSI_21']
    step3.step3_params['step3_tests_by_class'] = 66
    step3.step3_params['step3_idx_start'] = 0  # step3_idx_start = int(random.random()*1000)
    #
    for step3_recouvrement in (8, 13, 21, 34, 55, 89, 144, 233):   #(2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233):
        for step3_samples_by_class in (330, 660):    #(330, 660, 990, 1320, 1650, 1980):
            #
            # step3 parameters : modified by this loop
            #
            step3.step3_params['step3_recouvrement'] = step3_recouvrement  # 1 / proportion recouvrement
            step3.step3_params['step3_time_depth'] = step3_recouvrement
            step3.step3_params['step3_samples_by_class'] = step3_samples_by_class
            #
            # Model and learning parameters : unchanged during loop
            #
            _mm_dict = _modelManager.get_properties()
            #
            _mm_dict['model_architecture'] = 'Conv1D_Dense'
            _mm_dict['conv1D_block1_MaxPooling1D_pool_size'] = 2
            _mm_dict['config_GRU_LSTM_units'] = 128
            _mm_dict['config_Dense_units'] = 96
            _mm_dict['dropout_rate'] = 0.5
            _mm_dict['optimizer_name'] = 'adam'
            _mm_dict['optimizer_modif_learning_rate'] = 0.75
            #
            # learning parameters (fit) TODO
            model_fit_batch_size = 32
            model_fit_epochs_max = 500
            model_fit_earlystopping_patience = 100
            #
            _modelManager.update_properties(_mm_dict)
            #
            for conv1D_block1_filters in (55,89,144,233,377,610,987):
                for conv1D_block1_kernel_size in (2,3,5):
                    #
                    # Model and learning parameters : modified by this loop
                    #
                    _mm_dict = _modelManager.get_properties()
                    #
                    _mm_dict['conv1D_block1_filters'] = conv1D_block1_filters
                    _mm_dict['conv1D_block1_kernel_size'] = conv1D_block1_kernel_size
                    #
                    _modelManager.update_properties(_mm_dict)
                    #
                    learn_from_step3(step3.step3_params, _modelManager)

