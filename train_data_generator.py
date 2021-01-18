import os
import sys


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


import arbo



class TrainDataGeneratorBase:
    # -------------------------------------------------------------------------
    # Attributes
    # -------------------------------------------------------------------------
    __df_raw_data = None                # = output of loader
    __df_raw_data_with_target = None    # = output of trigger
    __df_learning_data = None           # = output of norm data ?
    # -------------------------------------------------------------------------
    # Members functions
    # -------------------------------------------------------------------------
    def set_raw_data(self, raw_data):
        self.__df_raw_data = raw_data
    #
    def get_raw_data(self):
        return self.__df_raw_data
    #
    def set_raw_data_with_target(self, raw_data_with_target):
        self.__df_raw_data_with_target
    #
    def get_raw_data_with_target(self):
        return self.__df_raw_data_with_target
    #
    def get_learning_data(self):
        return self.__df_learning_data
    #
    # Set __df_raw_data from outside this class
    #
    def compute_raw_data(self):
        raise NotImplementedError("Please Implement this method")
    #
    # Add if not present and set field 'target' to  __df_raw_data => __df_raw_data_with_target
    #
    def compute_target(self):
        raise NotImplementedError("Please Implement this method")
    #
    # Generate __df_learning_data for networks with Flat entry
    #
    def compute_flat_learning_data(samples_by_class,
                                   column_names_to_scale,
                                   column_names_not_to_scale):
        raise NotImplementedError("Please Implement this method")
    #
    # Generate __df_learning_data for networks with LSTM/GRU or Conv1D entry 
    #
    def compute_learning_data(samples_by_class,
                              time_depth,
                              column_names_to_scale,
                              column_names_not_to_scale):
        raise NotImplementedError("Please Implement this method")




class FCTrainDataGenerator(TrainDataGeneratorBase):
    #
    # Set __df_raw_data from outside this class
    #
    def compute_raw_data(self):
        #
        import pandas
        import step1_dataset_prepare_raw_data as step1
        #
        Usa500_dfH1, Usa500_dfM30, Usa500_dfM15, Usa500_dfM5, \
        UsaInd_dfH1, UsaInd_dfM30, UsaInd_dfM15, UsaInd_dfM5, \
        UsaTec_dfH1, UsaTec_dfM30, UsaTec_dfM15, UsaTec_dfM5, \
        Ger30_dfH1, Ger30_dfM30, Ger30_dfM15, Ger30_dfM5, \
        EURUSD_dfH1, EURUSD_dfM30, EURUSD_dfM15, EURUSD_dfM5, \
        USDJPY_dfH1, USDJPY_dfM30, USDJPY_dfM15, USDJPY_dfM5, \
        GOLD_dfH1, GOLD_dfM30, GOLD_dfM15, GOLD_dfM5, \
        LCrude_dfH1, LCrude_dfM30, LCrude_dfM15, LCrude_dfM5, \
        big_H1, big_M30, big_M15, big_M5 = step1.load_prepared_data(_dataset_name)
        #
        data_all_df_bigs = {'H1': [big_H1],
                            'M30': [big_M30],
                            'M15': [big_M15],
                            'M5': [big_M5]}
        _all_df_bigs = pandas.DataFrame(data_all_df_bigs)
        #
        super().set_raw_data(_all_df_bigs)
    #
    # Add if not present and set field 'target' to  __df_raw_data => __df_raw_data_with_target
    #
    def compute_target(self, step2_params):
        #
        import step2_dataset_prepare_target_data as step2
        #
        output_step2_data_with_target = step2.prepare_target_data_with_define_target(
            self.get_raw_data(),
            step2_params['step2_profondeur_analyse'],
            step2_params['step2_target_period'],
            step2_params['step2_symbol_for_target'],
            step2_params['step2_targetLongShort'],
            step2_params['step2_ratio_coupure'],
            step2_params['step2_targets_classes_count'],
            step2_params['step2_target_class_col_name'],
            step2_params['step2_use_ATR'])
        #
        print(step2_params['step2_profondeur_analyse'])
        print(output_step2_data_with_target)
        super().set_raw_data_with_target(output_step2_data_with_target)
    #
    def create_step3_data(self,step3_params):
        #
        learning_data = learning_data_template.copy()
        #
        output_step2_data_with_target = self.get_raw_data_with_target()
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





#
# Use
#

_dataset_name = 'work'

fcg = FCTrainDataGenerator()
fcg.compute_raw_data()
fcg.get_raw_data()



step2_params = {
    'step2_target_class_col_name' : [],
    'step2_profondeur_analyse' : 0,
    'step2_target_period' : 0,
    'step2_symbol_for_target' : '',
    'step2_symbol_spread' : 0.0,
    'step2_targets_classes_count' : 0,
    'step2_targetLongShort' : 0.0,
    'step2_ratio_coupure' : 0.0,
    'step2_use_ATR' : True}

step2_params['step2_target_class_col_name'] = 'target_class'
step2_params['step2_profondeur_analyse'] = 3
step2_params['step2_target_period'] = 'M15'

step2_params['step2_symbol_for_target'] = 'UsaInd'
step2_params['step2_targets_classes_count'] = 3
step2_params['step2_symbol_spread'] = 2.5

step2_params['step2_targetLongShort'] = 0.95
step2_params['step2_ratio_coupure'] = 1.1
step2_params['step2_use_ATR'] = True

fcg.compute_target(step2_params)
fcg.get_raw_data_with_target()


