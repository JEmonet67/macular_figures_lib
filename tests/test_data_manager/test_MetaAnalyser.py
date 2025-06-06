import os
import pickle

import numpy as np

from src.data_manager.MetaAnalyser import MetaAnalyser

# Get data for test from relative path.
path_data_test = os.path.normpath(f"{os.getcwd()}/../data_test/data_manager/")

# Import of a fully analyzed MacularAnalysisDataframes based on default multiple MacularDictArray.
with (open(f"{path_data_test}/MacularAnalysisDataframes/fully_analyzed_macular_analysis_dataframe.pyb", "rb")
      as file):
    macular_analysis_dataframes_default = pickle.load(file)


def test_normalization_computing():
    # Case with only integers.
    assert MetaAnalyser.normalization_computing(4, 2, 1) == 2

    # Case with an array in the numerator.
    assert np.array_equal(MetaAnalyser.normalization_computing(np.array([2, 4, 6]), 2, 3),
                          np.array([3, 6, 9]))

    # Case with an array in the denominator.
    assert np.array_equal(MetaAnalyser.normalization_computing(10, np.array([2, 5, 10]), 2.5),
                          np.array([12.5, 5, 2.5]))

    # Multiplication factor to be used for normalization.
    assert np.array_equal(MetaAnalyser.normalization_computing(np.array([10, 8, 5]), np.array([10, 8, 5]), 1),
                          np.array([1, 1, 1]))


def test_linear_fit_computing():
    # Initialisation of an index for fitting tests.
    index = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    # Initialisation of data corresponding to a single linear segment.
    data_one_segment = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    fitting_dict_one_segment = MetaAnalyser.linear_fit_computing(index, data_one_segment, 1)

    # Creating a correctly fitted dictionary for the single linear segment case.
    fitting_dict_one_segment_correct = {
        'slopes': [1.0],
        'index_prediction': np.array(
            [1., 1.141, 1.283, 1.424, 1.566, 1.707, 1.848, 1.99, 2.131, 2.273, 2.414, 2.556, 2.697, 2.838, 2.98, 3.121,
             3.263, 3.404, 3.545, 3.687, 3.828, 3.97, 4.111, 4.253, 4.394, 4.535, 4.677, 4.818, 4.96, 5.101, 5.242,
             5.384, 5.525, 5.667, 5.808, 5.949, 6.091, 6.232, 6.374, 6.515, 6.657, 6.798, 6.939, 7.081, 7.222, 7.364,
             7.505, 7.646, 7.788, 7.929, 8.071, 8.212, 8.354, 8.495, 8.636, 8.778, 8.919, 9.061, 9.202, 9.343, 9.485,
             9.626, 9.768, 9.909, 10.051, 10.192, 10.333, 10.475, 10.616, 10.758, 10.899, 11.04, 11.182, 11.323, 11.465,
             11.606, 11.747, 11.889, 12.03, 12.172, 12.313, 12.455, 12.596, 12.737, 12.879, 13.02, 13.162, 13.303,
             13.444, 13.586, 13.727, 13.869, 14.01, 14.152, 14.293, 14.434, 14.576, 14.717, 14.859, 15.]),
        'data_prediction': np.array(
            [1., 1.141, 1.283, 1.424, 1.566, 1.707, 1.848, 1.99, 2.131, 2.273, 2.414, 2.556, 2.697, 2.838, 2.98,
             3.121, 3.263, 3.404, 3.545, 3.687, 3.828, 3.97, 4.111, 4.253, 4.394, 4.535, 4.677, 4.818, 4.96, 5.101,
             5.242, 5.384, 5.525, 5.667, 5.808, 5.949, 6.091, 6.232, 6.374, 6.515, 6.657, 6.798, 6.939, 7.081, 7.222,
             7.364, 7.505, 7.646, 7.788, 7.929, 8.071, 8.212, 8.354, 8.495, 8.636, 8.778, 8.919, 9.061, 9.202, 9.343,
             9.485, 9.626, 9.768, 9.909, 10.051, 10.192, 10.333, 10.475, 10.616, 10.758, 10.899, 11.04, 11.182, 11.323,
             11.465, 11.606, 11.747, 11.889, 12.03, 12.172, 12.313, 12.455, 12.596, 12.737, 12.879, 13.02, 13.162,
             13.303, 13.444, 13.586, 13.727, 13.869, 14.01, 14.152, 14.293, 14.434, 14.576, 14.717, 14.859, 15.])}

    # Testing the case with a single linear segment.
    for key in fitting_dict_one_segment:
        if "prediction" in key:
            assert np.array_equal(fitting_dict_one_segment[key], fitting_dict_one_segment_correct[key])
        else:
            assert fitting_dict_one_segment[key] == fitting_dict_one_segment_correct[key]

    # Initialisation of data corresponding to 4 linear segments.
    data_multi_segment = np.array([1, 2, 3, 6, 9, 12, 15, 35, 55, 75, 95, 85, 75, 65, 55])
    fitting_dict_multi_segment = MetaAnalyser.linear_fit_computing(index, data_multi_segment, 4)

    # Creating a correctly fitted dictionary for the case with multiple linear segments.
    fitting_dict_multi_segment_correct = {
        'slopes': [1.0, 3.0, 20.0, -10.0],
        'index_prediction': np.array(
            [1., 1.141, 1.283, 1.424, 1.566, 1.707, 1.848, 1.99, 2.131, 2.273, 2.414, 2.556, 2.697, 2.838, 2.98, 3.121,
             3.263, 3.404, 3.545, 3.687, 3.828, 3.97, 4.111, 4.253, 4.394, 4.535, 4.677, 4.818, 4.96, 5.101, 5.242,
             5.384, 5.525, 5.667, 5.808, 5.949, 6.091, 6.232, 6.374, 6.515, 6.657, 6.798, 6.939, 7.081, 7.222, 7.364,
             7.505, 7.646, 7.788, 7.929, 8.071, 8.212, 8.354, 8.495, 8.636, 8.778, 8.919, 9.061, 9.202, 9.343, 9.485,
             9.626, 9.768, 9.909, 10.051, 10.192, 10.333, 10.475, 10.616, 10.758, 10.899, 11.04, 11.182, 11.323, 11.465,
             11.606, 11.747, 11.889, 12.03, 12.172, 12.313, 12.455, 12.596, 12.737, 12.879, 13.02, 13.162, 13.303,
             13.444, 13.586, 13.727, 13.869, 14.01, 14.152, 14.293, 14.434, 14.576, 14.717, 14.859, 15.]),
        'data_prediction': np.array(
            [1., 1.141, 1.283, 1.424, 1.566, 1.707, 1.848, 1.99, 2.131, 2.273, 2.414, 2.556, 2.697, 2.838, 2.98, 3.363,
             3.789, 4.212, 4.635, 5.061, 5.484, 5.91, 6.333, 6.759, 7.182, 7.605, 8.031, 8.454, 8.88, 9.303, 9.726,
             10.152, 10.575, 11.001, 11.424, 11.847, 12.273, 12.696, 13.122, 13.545, 13.971, 14.394, 14.817, 16.62,
             19.44, 22.28, 25.1, 27.92, 30.76, 33.58, 36.42, 39.24, 42.08, 44.9, 47.72, 50.56, 53.38, 56.22, 59.04,
             61.86, 64.7, 67.52, 70.36, 73.18, 76.02, 78.84, 81.66, 84.5, 87.32, 90.16, 92.98, 94.6, 93.18, 91.77,
             90.35, 88.94, 87.53, 86.11, 84.7, 83.28, 81.87, 80.45, 79.04, 77.63, 76.21, 74.8, 73.38, 71.97, 70.56,
             69.14, 67.73, 66.31, 64.9, 63.48, 62.07, 60.66, 59.24, 57.83, 56.41, 55.]),
        'inflexion_points_index': [3.0, 7.0, 11.0],
        'inflexion_points_data': [3.363, 16.62, 94.6]}

    # Testing the case with multiple linear segments.
    for key in fitting_dict_multi_segment:
        if "prediction" in key:
            assert np.array_equal(fitting_dict_multi_segment[key], fitting_dict_multi_segment_correct[key])
        else:
            assert fitting_dict_multi_segment[key] == fitting_dict_multi_segment_correct[key]

