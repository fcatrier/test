#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

import os
import sys
import keras

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


import pandas
import numpy

import sklearn
from sklearn.utils import shuffle

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
import ModelFactory
from ModelFactory import CModelFactory

import learn_history

# learning history
_dir_npy = '\\npy_current'


def learn_from_step3(step3_params, model_factory_params, loops_count=1):
    #
    idx_run_loop = learn_history.new_npy_idx(_dataset_name, _dir_npy)
    #
    idx_generate_learning_data = step3_params['step3_idx_start']    # incrémenté à chaque appel à step3.generate_learning_data_dynamic3
    resOK_train, idx_generate_learning_data, np_X_test2, df_y_Nd_test2, df_y_1d_test2, df_atr_test2 = step3.generate_learning_data_dynamic3(
        output_step2_data_with_target, idx_generate_learning_data,
        step3_params['step3_recouvrement'], step3_params['step3_tests_by_class'], step3_params['step3_time_depth'],
        step3_params['step3_column_names_to_scale'], step3_params['step3_column_names_not_to_scale'])
    resOK_val, idx_generate_learning_data, np_X_test1, df_y_Nd_test1, df_y_1d_test1, df_atr_test1 = step3.generate_learning_data_dynamic3(
        output_step2_data_with_target, idx_generate_learning_data,
        step3_params['step3_recouvrement'], step3_params['step3_tests_by_class'], step3_params['step3_time_depth'],
        step3_params['step3_column_names_to_scale'], step3_params['step3_column_names_not_to_scale'])
    resOK_test1, idx_generate_learning_data, np_X_val, df_y_Nd_val, df_y_1d_val, df_atr_val = step3.generate_learning_data_dynamic3(
        output_step2_data_with_target, idx_generate_learning_data,
        step3_params['step3_recouvrement'], step3_params['step3_tests_by_class'], step3_params['step3_time_depth'],
        step3_params['step3_column_names_to_scale'], step3_params['step3_column_names_not_to_scale'])
    resOK_test2, idx_generate_learning_data, np_X_train, df_y_Nd_train, df_y_1d_train, df_atr_train = step3.generate_learning_data_dynamic3(
        output_step2_data_with_target, idx_generate_learning_data,
        step3_params['step3_recouvrement'], step3_params['step3_tests_by_class'], step3_params['step3_time_depth'],
        step3_params['step3_column_names_to_scale'], step3_params['step3_column_names_not_to_scale'])
    #
    if ((resOK_train == False) or (resOK_val == False) or (resOK_test1 == False) or (resOK_test2 == False)):
        print("step3.generate_learning_data_dynamic3 failed. STOP")
        return
    #
    # bizarrement si on enlève ce shuffle il n'y a plus de convergence, bien qu'un nouveau shuffle soit fait plus tard par model.fit
    # TODO à mettre en paramètre ici et rééessayer de l'enlever...
    np_X_train, df_y_1d_train, df_y_Nd_train = sklearn.utils.shuffle(np_X_train, df_y_1d_train, df_y_Nd_train)
    np_X_val, df_y_1d_val, df_y_Nd_val = sklearn.utils.shuffle(np_X_val, df_y_1d_val, df_y_Nd_val)
    #
    input_features = np_X_train.shape[2]
    input_timesteps = np_X_train.shape[1]
    output_shape = df_y_Nd_train.shape[1]
    print("input_features=", input_features)
    print("input_timesteps=", input_timesteps)
    print("output_shape=", output_shape)
    train_samples = np_X_train.shape[0]
    print("train_samples=", train_samples)
    #
    for i in range(0, loops_count):
        print("")
        print("-----------------------------------------------------------")
        print("Repeat #", i + 1, " /", loops_count)
        print("-----------------------------------------------------------")
        print("")
        model, model_factory = define_model(model_factory_params, input_features, input_timesteps, output_shape, train_samples)
        #
        if (i == 0):  # affiché uniquement au premier passage pour désaturer l'affichage
            print(model.summary())
            print("model.count_params()=", model.count_params())
            print("np_X_train.shape[0]*np_X_train.shape[1]*np_X_train.shape[2] (X_train_params) : ", train_samples*input_timesteps*input_features)
        #
        callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=model_fit_earlystopping_patience, restore_best_weights=True)
        history = model.fit(np_X_train, df_y_Nd_train, model_fit_batch_size, shuffle=False, epochs=model_fit_epochs_max,
                            callbacks=[callback], verbose=1,
                            validation_data=(np_X_val, df_y_Nd_val))
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
        y_pred_raw = model.predict(np_X_val)  # avec les poids sortie modèle
        df_y_pred_val = pandas.DataFrame(numpy.argmax(y_pred_raw, axis=-1))  # avec les poids forcés à 0/1
        df_y_pred_val.index = df_y_1d_val.index
        #
        val_acc, res_eval_result_atr_val, cm_val, pc_resultat_val = learn_evaluate_results.evaluate_atr(
            df_y_pred_val, df_y_1d_val, df_atr_val,
            step2.step2_params['step2_symbol_spread'], step2.step2_params['step2_ratio_coupure'])
        print('==>> res_eval_result_atr_val = ', res_eval_result_atr_val, '\tpc_resultat_val =', pc_resultat_val)
        print('val_acc = ', val_acc)
        print(cm_val)
        learn_evaluate_results.cm_metrics(cm_val)
        print("---")
        #
        df_y_pred1 = pandas.DataFrame(numpy.argmax(model.predict(np_X_test1), axis=-1))
        df_y_pred1.index = df_y_1d_test1.index
        #
        test1_acc, res_eval_result_atr_test1, cm_test1, pc_resultat_test1 = learn_evaluate_results.evaluate_atr(
            df_y_pred1, df_y_1d_test1, df_atr_test1,
            step2.step2_params['step2_symbol_spread'], step2.step2_params['step2_ratio_coupure'])
        print('==>> res_eval_result_atr_test1 = ', res_eval_result_atr_test1, '\tpc_resultat_test1 =',
              pc_resultat_test1)
        print('test1_acc = ', test1_acc)
        print(cm_test1)
        learn_evaluate_results.cm_metrics(cm_test1)
        print("---")
        #
        df_y_pred2 = pandas.DataFrame(numpy.argmax(model.predict(np_X_test2), axis=-1))
        df_y_pred2.index = df_y_1d_test2.index
        #
        test2_acc, res_eval_result_atr_test2, cm_test2, pc_resultat_test2 = learn_evaluate_results.evaluate_atr(
            df_y_pred2, df_y_1d_test2, df_atr_test2,
            step2.step2_params['step2_symbol_spread'], step2.step2_params['step2_ratio_coupure'])
        print('==>> res_eval_result_atr_test2 = ', res_eval_result_atr_test2, '\tpc_resultat_test2 =',
              pc_resultat_test2)
        print('test2_acc= ', test2_acc)
        print(cm_test2)
        learn_evaluate_results.cm_metrics(cm_test2)
        print("---")
        #
        model_factory.save_model(_dataset_name, _dir_npy, idx_run_loop)
        step2.step2_save(_dataset_name, _dir_npy, idx_run_loop, step2.step2_params)
        step3.step3_save(_dataset_name, _dir_npy, idx_run_loop, step3_params)
        #
        learn_history.run_loop_historize(_dataset_name,
                                         _dir_npy,
                                         idx_run_loop,
                                         #
                                         model_fit_batch_size,
                                         model_fit_epochs_max,
                                         model_fit_earlystopping_patience,
                                         #
                                         train_loss,
                                         val_loss,
                                         train_accuracy,
                                         val_accuracy, res_eval_result_atr_val, cm_val, pc_resultat_val,
                                         test1_acc, res_eval_result_atr_test1, cm_test1, pc_resultat_test1,
                                         test2_acc, res_eval_result_atr_test2, cm_test2, pc_resultat_test2)
        idx_run_loop += 1
        #
        del model  # attention en supprimant le modèle on perd l'apprentissage ici
    #

def define_model(model_factory_params, input_features, input_timesteps, output_shape, train_samples):
    model_factory = CModelFactory()
    #
    model_factory.set_params_Conv1D(model_factory_params['conv1D_block1_filters'],
                                    model_factory_params['conv1D_block1_kernel_size'],
                                    model_factory_params['conv1D_block1_MaxPooling1D_pool_size'])
    model_factory.set_params_GRU_LSTM(model_factory_params['config_GRU_LSTM_units'])
    model_factory.set_params_Dense(model_factory_params['config_Dense_units'])
    #
    model_factory.set_params_optimizer(model_factory_params['optimizer_name'],
                                       model_factory_params['optimizer_modif_learning_rate'])
    model_factory.set_dropout_rate(model_factory_params['dropout_rate'])
    model_factory.set_params_inout(input_features, input_timesteps, output_shape, train_samples)
    #
    model = model_factory.create_compile_model(model_factory_params['model_architecture'])
    return model, model_factory


# learning parameters (fit) TODO
model_fit_batch_size = 32
model_fit_epochs_max = 500
model_fit_earlystopping_patience = 100

#
# Ici point d'entrée pour ajuster les paramètres ou coder les boucles de variations
#
def execute():
    #
    # step3 parameters
    #
    step3.step3_params['step3_column_names_to_scale'] = []
    step3.step3_params['step3_column_names_not_to_scale'] = [
        'UsaInd_M15_time_slot',
        'UsaInd_M15_pRSI_3', 'UsaInd_M15_pRSI_5', 'UsaInd_M15_pRSI_8', 'UsaInd_M15_pRSI_13', 'UsaInd_M15_pRSI_21']
    for step3_recouvrement in (8, 13, 21, 34, 55, 89, 144, 233):   #(2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233):
        step3.step3_params['step3_recouvrement'] = step3_recouvrement  # 1 / proportion recouvrement
        step3.step3_params['step3_tests_by_class'] = 66
        step3.step3_params['step3_time_depth'] = step3.step3_params['step3_recouvrement']
        step3.step3_params['step3_idx_start'] = 0
        # step3_idx_start = int(random.random()*1000)
        for step3_samples_by_class in (330, 660):    #(330, 660, 990, 1320, 1650, 1980):
            step3.step3_params['step3_samples_by_class'] = step3_samples_by_class
            #
            #
            # Model and learning parameters / TODO plus en variables globales
            #
            #
            # Model and learning parameters / TODO plus en variables globales
            #
            ModelFactory.model_factory_params['model_architecture'] = 'Conv1D_Dense'
            #
            ModelFactory.model_factory_params['conv1D_block1_filters'] = 0
            ModelFactory.model_factory_params['conv1D_block1_kernel_size'] = 6
            ModelFactory.model_factory_params['conv1D_block1_MaxPooling1D_pool_size'] = 2
            #
            ModelFactory.model_factory_params['config_GRU_LSTM_units'] = 128
            #
            ModelFactory.model_factory_params['config_Dense_units'] = 96
            ModelFactory.model_factory_params['config_Dense_units2'] = 0
            #
            ModelFactory.model_factory_params['optimizer_name'] = 0
            ModelFactory.model_factory_params['optimizer_modif_learning_rate'] = 0
            #
            ModelFactory.model_factory_params['dropout_rate'] = 0.5
            ModelFactory.model_factory_params['optimizer_name'] = 'adam'
            ModelFactory.model_factory_params['optimizer_modif_learning_rate'] = 0.75
            #
            for conv1D_block1_filters in (55,89,144,233,377,610,987):
                ModelFactory.model_factory_params['conv1D_block1_filters'] = conv1D_block1_filters
                for conv1D_block1_kernel_size in (2,3,5):
                    ModelFactory.model_factory_params['conv1D_block1_kernel_size'] = conv1D_block1_kernel_size
                    learn_from_step3(step3.step3_params, ModelFactory.model_factory_params)

