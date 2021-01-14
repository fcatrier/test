# -----------------------------------------------------------------------------
# Définition des chemins
# -----------------------------------------------------------------------------

# dataset_name     = 'M5M15-0720'
# dataset_name     = 'M1-0720 - Copie'
# dataset_name     = 'flower_photos'
# dataset_name     = 'M5M1-0720 - MA'

# py_dir='C:\\Users\\T0042310\\Documents\\Perso\\Py'
# py_dir='E:\\Py'

# datasets_dir     = py_dir + '\\Datasets'
# dataset_root_dir = datasets_dir + '\\' +dataset_name
# source_data_dir  = dataset_root_dir + '\\SourceData'
# full_data_dir    = dataset_root_dir + '\\FullData'
# tmp_generate_dir = full_data_dir + '\\tmp_generate'
# study_dir        = dataset_root_dir + '\\Study'
# train_dir        = study_dir + '\\train'
# val_dir          = study_dir + '\\val'
# test_dir         = study_dir + '\\test'
# run_dir_train    = study_dir + '\\run_dir_train'
# run_dir_val      = study_dir + '\\run_dir_val'
# model_dir        = study_dir + '\\model'

def get_datasets_dir(py_dir):
    return py_dir + '\\Datasets'


def get_dataset_root_dir(py_dir, dataset_name):
    return get_datasets_dir(py_dir) + '\\' + dataset_name


def get_source_data_dir(py_dir, dataset_name):
    return get_dataset_root_dir(py_dir, dataset_name) + '\\SourceData'


def get_full_data_dir(py_dir, dataset_name):
    return get_dataset_root_dir(py_dir, dataset_name) + '\\FullData'


def get_tmp_generate_dir(py_dir, dataset_name):
    return get_full_data_dir(py_dir, dataset_name) + '\\tmp_generate'


def get_study_dir(py_dir, dataset_name):
    return get_dataset_root_dir(py_dir, dataset_name) + '\\Study'


def get_train_dir(py_dir, dataset_name):
    return get_study_dir(py_dir, dataset_name) + '\\train'


def get_val_dir(py_dir, dataset_name):
    return get_study_dir(py_dir, dataset_name) + '\\val'


def get_test_dir(py_dir, dataset_name):
    return get_study_dir(py_dir, dataset_name) + '\\test'


def get_run_dir_train(py_dir, dataset_name):
    return get_study_dir(py_dir, dataset_name) + '\\run_dir_train'


def get_run_dir_val(py_dir, dataset_name):
    return get_study_dir(py_dir, dataset_name) + '\\run_dir_val'


def get_model_dir(py_dir, dataset_name):
    return get_study_dir(py_dir, dataset_name) + '\\model'


def get_learning_files_database_dir(py_dir, dataset_name):
    return get_study_dir(py_dir, dataset_name) + '\\learning_files_database'

# TU (à appeler d'un autre fichier)
# print(arbo.get_datasets_dir(py_dir))
# print(arbo.get_dataset_root_dir(py_dir,dataset_name))
# print(arbo.get_source_data_dir(py_dir,dataset_name))
# print(arbo.get_full_data_dir(py_dir,dataset_name))
# print(arbo.get_tmp_generate_dir(py_dir,dataset_name))
# print(arbo.get_study_dir(py_dir,dataset_name))
# print(arbo.get_train_dir(py_dir,dataset_name))
# print(arbo.get_val_dir(py_dir,dataset_name))
# print(arbo.get_test_dir(py_dir,dataset_name))
# print(arbo.get_run_dir_train(py_dir,dataset_name))
# print(arbo.get_run_dir_val(py_dir,dataset_name))
# print(arbo.get_model_dir(py_dir,dataset_name))
