#
# Copyright (c) 2020 by Frederi CATRIER - All rights reserved.
#

# -----------------------------------------------------------------------------
# Méthodes pour l'analyse des résultats post learning
# -----------------------------------------------------------------------------

import pandas
import numpy
from sklearn.metrics import confusion_matrix


def evaluate_not_atr(df_y_pred, df_y_1d_test, df_atr_test, target_long_short, step2_symbol_spread, step2_ratio_coupure):
    cm, test_acc, dfMerged = compute_cm(df_y_pred, df_y_1d_test, df_atr_test)
    res = 0.0
    try:
        gain = target_long_short
        #
        true_1_predicted_1 = cm[0, 0]
        true_1_predicted_2 = cm[0, 1]
        true_1_predicted_3 = cm[0, 2]
        #
        true_2_predicted_1 = cm[1, 0]
        true_2_predicted_2 = cm[1, 1]
        true_2_predicted_3 = cm[1, 2]
        #
        true_3_predicted_1 = cm[2, 0]
        true_3_predicted_2 = cm[2, 1]
        true_3_predicted_3 = cm[2, 2]
        #
        res += true_1_predicted_1 * (+gain - step2_symbol_spread)
        res += true_2_predicted_1 * (-gain * step2_ratio_coupure - step2_symbol_spread)
        res += true_3_predicted_1 * (-step2_symbol_spread)
        #
        res += true_1_predicted_2 * (-gain * step2_ratio_coupure - step2_symbol_spread)
        res += true_2_predicted_2 * (+gain - step2_symbol_spread)
        res += true_3_predicted_2 * (-step2_symbol_spread)
        #
        res += true_1_predicted_3 * 0.0  # predicted 3 = no action
        res += true_2_predicted_3 * 0.0  # predicted 3 = no action
        res += true_3_predicted_3 * 0.0  # predicted 3 = no action
        #
    except:
        pass
    return test_acc, res, cm


def evaluate_atr(df_y_pred, df_y_1d_test, df_atr_test, step2_symbol_spread, step2_ratio_coupure):
    cm, test_acc, dfMerged = compute_cm(df_y_pred, df_y_1d_test, df_atr_test)
    #
    dfMerged_y_pred_0 = dfMerged[(dfMerged['y_pred'] == 0)]
    dfMerged_y_pred_1 = dfMerged[(dfMerged['y_pred'] == 1)]
    dfMerged_y_pred_2 = dfMerged[(dfMerged['y_pred'] == 2)]
    #
    dfMerged_y_pred_0_true_0 = dfMerged_y_pred_0[(dfMerged_y_pred_0['y_true'] == 0)]
    dfMerged_y_pred_0_true_1 = dfMerged_y_pred_0[(dfMerged_y_pred_0['y_true'] == 1)]
    dfMerged_y_pred_0_true_2 = dfMerged_y_pred_0[(dfMerged_y_pred_0['y_true'] == 2)]
    #
    dfMerged_y_pred_1_true_0 = dfMerged_y_pred_1[(dfMerged_y_pred_1['y_true'] == 0)]
    dfMerged_y_pred_1_true_1 = dfMerged_y_pred_1[(dfMerged_y_pred_1['y_true'] == 1)]
    dfMerged_y_pred_1_true_2 = dfMerged_y_pred_1[(dfMerged_y_pred_1['y_true'] == 2)]
    #
    # gains = atr
    #
    gains = dfMerged_y_pred_0_true_0['atr_test'].sum() + dfMerged_y_pred_1_true_1['atr_test'].sum()
    #
    # perte = atr * step2_ratio_coupure
    #     Nota : les prédictions 0 ou 1 et réel 2 sont comptabilisées neutres
    #            (coût = spread, inclus dans coût total spread)
    #
    pertes = (dfMerged_y_pred_0_true_1['atr_test'].sum() + dfMerged_y_pred_1_true_0[
        'atr_test'].sum()) * step2_ratio_coupure
    #
    # cout_total_spread = step2_symbol_spread * nb_operations
    nb_operations = dfMerged_y_pred_0.shape[0] + dfMerged_y_pred_1.shape[0]
    cout_total_spread = step2_symbol_spread * nb_operations
    #
    # resultat
    #
    resultat = gains - pertes - cout_total_spread
    max_possible_resultat = dfMerged['atr_test'].sum() - step2_symbol_spread * cm.sum()
    min_possible_resultat = - dfMerged['atr_test'].sum() * step2_ratio_coupure - step2_symbol_spread * cm.sum()
    pc_resultat = 100.0 * (resultat - min_possible_resultat) / (max_possible_resultat - min_possible_resultat)
    #
    return test_acc, int(resultat), cm, int(pc_resultat)


def compute_cm(df_y_pred, df_y_1d_test, df_atr_test):
    df_y_1d_test_translated_0_2 = df_y_1d_test[0] - 1
    dfMerged = pandas.concat([df_y_pred, df_y_1d_test_translated_0_2, df_atr_test], join='inner', axis=1)
    dfMerged.columns = ['y_pred', 'y_true', 'atr_test']
    #
    y_true = numpy.array(df_y_1d_test_translated_0_2.values)
    y_pred = numpy.array(df_y_pred.values)
    cm = confusion_matrix(y_true, y_pred)
    #
    # numpy.trace(cm) = somme sur la diagonale (prédictions correctes)
    test_acc = round(numpy.trace(cm) / numpy.sum(cm), 2)
    #
    return cm, test_acc, dfMerged


def cm_metrics(cm):
    #
    # sommes par colonnes : analyse des prédictions
    #
    pred_1_count = cm[:, 0].sum()
    pred_2_count = cm[:, 1].sum()
    pred_3_count = cm[:, 2].sum()
    arr_for_std = []
    arr_for_std.append(100.0 * pred_1_count / cm[:].sum())
    arr_for_std.append(100.0 * pred_3_count / cm[:].sum())
    arr_for_std.append(100.0 * pred_3_count / cm[:].sum())
    print("pred_1 =", round(pred_1_count / (pred_1_count + pred_2_count + pred_3_count), 2),
          "\tpred_2 =", round(pred_2_count / (pred_1_count + pred_2_count + pred_3_count), 2),
          "\tpred_3 =", round(pred_3_count / (pred_1_count + pred_2_count + pred_3_count), 2),
          "\tecart-type pred 1/2/3 =", int(numpy.std(arr_for_std)))
    # #
    # # sommes par lignes : analyse des valeurs réelles (ici doit faire step3_tests_by_class)
    # #
    # true_1_count = cm[0].sum()
    # true_2_count = cm[1].sum()
    # true_3_count = cm[2].sum()
    # #
    # # analyse détaillée de chaque cas
    # #
    # true_1_predicted_1 = cm[0,0]
    # true_1_predicted_2 = cm[0,1]
    # true_1_predicted_3 = cm[0,2]
    # #
    # true_2_predicted_1 = cm[1,0]
    # true_2_predicted_2 = cm[1,1]
    # true_2_predicted_3 = cm[1,2]
    # #
    # true_3_predicted_1 = cm[2,0]
    # true_3_predicted_2 = cm[2,1]
    # true_3_predicted_3 = cm[2,2]
    # #
    # pc_predicted_true_class_1 = 0.0
    # pc_predicted_true_class_2 = 0.0
    # pc_predicted_true_class_3 = 0.0
    # ratio_OK_vs_KO_true_1_predicted_2 = 0.0
    # ratio_OK_vs_KO_true_2_predicted_1 = 0.0
    # try:
    # pc_predicted_true_class_1 = round(pred_1_count/true_1_count,2)
    # pc_predicted_true_class_2 = round(pred_2_count/true_2_count,2)
    # pc_predicted_true_class_3 = round(pred_3_count/true_3_count,2)
    # #
    # ratio_OK_vs_KO_true_1_predicted_2 = round(true_1_predicted_1/true_1_predicted_2,2)
    # ratio_OK_vs_KO_true_2_predicted_1 = round(true_2_predicted_2/true_2_predicted_1,2)
    # except:
    # pass
    # #
    # print("   pc_predicted_true_class_1 =",pc_predicted_true_class_1)
    # print("   pc_predicted_true_class_2 =",pc_predicted_true_class_2)
    # print("   pc_predicted_true_class_3 =",pc_predicted_true_class_3)
    # print("   ratio_OK_vs_KO_true_1_predicted_2 =",ratio_OK_vs_KO_true_1_predicted_2, \
    # "(erreur true_1_predicted_2=",true_1_predicted_2, \
    # " vs ok true_1_predicted_1=", true_1_predicted_1,")")
    # print("   ratio_OK_vs_KO_true_2_predicted_1 =",ratio_OK_vs_KO_true_2_predicted_1, \
    # "(erreur true_2_predicted_1=",true_2_predicted_1, \
    # " vs ok true_2_predicted_2=", true_2_predicted_2,")")
    # #
