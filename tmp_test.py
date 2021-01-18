
import os
os.system('cls')

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
    ('X_train_params', -1)
])


class TrainDataGeneratorBase:
    # -------------------------------------------------------------------------
    # Attributes
    # -------------------------------------------------------------------------
    __df_raw_data = 1                # = output of loader
    __df_raw_data_with_target = None    # = output of trigger
    __df_learning_data = None           # = output of norm data ?
    # -------------------------------------------------------------------------
    # Members functions
    # -------------------------------------------------------------------------
    def get_raw_data(self):
        return self.__df_raw_data
    #
    def set_raw_data(self, raw_data):
        self.__df_raw_data = raw_data
    #



class FCTrainDataGenerator(TrainDataGeneratorBase):
    #
    # Set __df_raw_data from outside this class
    #
    def set_raw_data(self, raw_data):
        #
        super().set_raw_data(raw_data+1)




#
# Use
#


fcg = FCTrainDataGenerator()
fcg.get_raw_data()
fcg.set_raw_data(2)
fcg.get_raw_data()



fcg.define_target()



fcg.get_raw_data_with_target()
fcg.get_learning_data()

