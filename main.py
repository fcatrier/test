#
# Copyright (c) 2020 by Frederi CATRIER - All rights reserved.
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


# import new1
# if __name__ == '__main__':
    # new1.execute()

import arbo
import learn_history

_dataset_name = 'work'
_dir_npy = '\\npy_current'

if __name__ == '__main__':
    npy_path = arbo.get_study_dir(py_dir, _dataset_name) + _dir_npy
    df = learn_history.npy_results(npy_path)
    print(df.tail())