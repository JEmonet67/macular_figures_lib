import os
import pickle

import numpy as np

from src.data_manager.SpatialAnalyser import SpatialAnalyser

# Get data for test from relative path.
path_data_test = os.path.normpath(f"{os.getcwd()}/../data_test/data_manager/")

path_pyb_file_default = f"{path_data_test}/SpatialAnalyser/RC_RM_dSGpCP0033_barSpeed30dps_default_0f.pyb"
# Import of the default MacularDictArray to be compared with preprocessing.
with open(path_pyb_file_default, "rb") as file_default:
    macular_dict_array_default = pickle.load(file_default)
