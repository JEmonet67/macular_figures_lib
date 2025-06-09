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
    # Set the randomness of the fitting for testing.
    np.random.seed(2)

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
             13.303, 13.444, 13.586, 13.727, 13.869, 14.01, 14.152, 14.293, 14.434, 14.576, 14.717, 14.859, 15.]),
        'data_intercepts': np.array([0]),
        'index_intercepts': np.array([0])
    }

    # Testing the case with a single linear segment.
    for key in fitting_dict_one_segment:
        if "prediction" in key or "intercepts" in key:
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
        'inflection_points_index': [3.0, 7.0, 11.0],
        'inflection_points_data': [3.363, 16.62, 94.6],
        'data_intercepts': np.array([0, -6, -125, 205]),
        'index_intercepts': np.array([0, -2, -6.25, -20.5])
    }

    # Testing the case with multiple linear segments.
    for key in fitting_dict_multi_segment:
        if "prediction" in key or "intercepts" in key:
            assert np.array_equal(fitting_dict_multi_segment[key], fitting_dict_multi_segment_correct[key])
        else:
            assert fitting_dict_multi_segment[key] == fitting_dict_multi_segment_correct[key]


def test_mean_computing():
    assert MetaAnalyser.mean_computing(np.array([i for i in range(11)])) == 5


def test_statistic_binning():
    # Initialisation of the correct binned data array.
    correct_data_array = [54.281, 61.723, 66.685, 71.646, 76.608, 81.57, 86.531, 91.493, 96.455, 101.416, 106.378,
                          111.339, 116.301, 121.263, 126.224, 131.186, 136.147, 141.109, 146.071, 151.032, 158.475,
                          165.917, 173.36, 180.802, 188.245, 195.687, 203.129, 210.572, 218.014, 225.457, 232.899,
                          240.341, 245.303, 252.746, 260.188, 267.63, 275.073, 282.515, 289.958, 297.4, 304.842,
                          312.285, 319.727, 327.17, 334.612, 342.054, 349.497, 356.94, 364.382, 371.824, 379.267,
                          386.709, 394.152, 399.113, 406.556, 413.998, 421.44, 428.883, 436.325, 443.768, 451.21,
                          458.653, 466.095, 473.537, 480.98, 488.422, 495.864, 503.307, 510.75, 518.192, 525.634,
                          533.077, 540.519]

    # Initialisation of the correct binned index array.
    correct_index_array = [1.225, 1.446, 1.668, 1.889, 2.111, 2.332, 2.554, 2.775, 2.997, 3.218, 3.44, 3.661, 3.883,
                           4.105, 4.326, 4.548, 4.769, 4.991, 5.212, 5.434, 5.655, 5.877, 6.098, 6.32, 6.541, 6.763,
                           6.984, 7.206, 7.427, 7.649, 7.87, 8.092, 8.313, 8.535, 8.756, 8.978, 9.2, 9.421, 9.643,
                           9.864, 10.086, 10.307, 10.529, 10.75, 10.972, 11.193, 11.415, 11.636, 11.858, 12.079, 12.301,
                           12.522, 12.744, 12.965, 13.187, 13.408, 13.63, 13.851, 14.073, 14.294, 14.516, 14.738,
                           14.959, 15.181, 15.402, 15.624, 15.845, 16.067, 16.288, 16.51, 16.731, 16.953, 17.174]

    # Initialisation of the array data to bz binned.
    data_array = [51.8, 56.762, 61.723, 66.685, 71.646, 76.608, 81.57, 86.531, 91.493, 96.455, 101.416, 106.378,
                  111.339, 116.301, 121.263, 126.224, 131.186, 136.147, 141.109, 146.071, 151.032, 155.994, 160.956,
                  165.917, 170.879, 175.84, 180.802, 185.764, 190.725, 195.687, 200.648, 205.61, 210.572, 215.533,
                  220.495, 225.457, 230.418, 235.38, 240.341, 245.303, 250.265, 255.226, 260.188, 265.149, 270.111,
                  275.073, 280.034, 284.996, 289.958, 294.919, 299.881, 304.842, 309.804, 314.766, 319.727, 324.689,
                  329.651, 334.612, 339.574, 344.535, 349.497, 354.459, 359.42, 364.382, 369.343, 374.305, 379.267,
                  384.228, 389.19, 394.152, 399.113, 404.075, 409.036, 413.998, 418.96, 423.921, 428.883, 433.844,
                  438.806, 443.768, 448.729, 453.691, 458.653, 463.614, 468.576, 473.537, 478.499, 483.461, 488.422,
                  493.384, 498.345, 503.307, 508.269, 513.23, 518.192, 523.154, 528.115, 533.077, 538.038, 543.0]

    # Initialisation of the array index to be binned.
    index_array = [1.114, 1.328, 1.543, 1.758, 1.972, 2.187, 2.402, 2.616, 2.831, 3.046, 3.26, 3.475, 3.689, 3.904,
                   4.119, 4.333, 4.548, 4.763, 4.977, 5.192, 5.406, 5.576, 5.726, 5.876, 6.026, 6.176, 6.326, 6.477,
                   6.627, 6.777, 6.927, 7.077, 7.227, 7.377, 7.527, 7.677, 7.828, 7.978, 8.128, 8.278, 8.428, 8.578,
                   8.728, 8.878, 9.028, 9.179, 9.329, 9.479, 9.629, 9.779, 9.929, 10.079, 10.229, 10.379, 10.53, 10.68,
                   10.83, 10.98, 11.13, 11.28, 11.43, 11.58, 11.73, 11.881, 12.031, 12.181, 12.331, 12.481, 12.631,
                   12.781, 12.931, 13.081, 13.232, 13.382, 13.532, 13.682, 13.832, 13.982, 14.132, 14.282, 14.432,
                   14.583, 14.733, 14.883, 15.033, 15.183, 15.333, 15.483, 15.633, 15.783, 15.934, 16.084, 16.234,
                   16.384, 16.534, 16.684, 16.834, 16.984, 17.134, 17.285]

    # Binning data and index arrays.
    binned_index_array, binned_data_array = MetaAnalyser.statistic_binning(data_array, index_array, 73)

    # Testing that binning is working properly.
    assert np.array_equal(binned_index_array, correct_index_array)
    assert np.array_equal(binned_data_array, correct_data_array)


def test_subtraction_computing():
    # Case of int values.
    assert MetaAnalyser.subtraction_computing(8, [4, 3, 1]) == 0

    # Case of an array as the value to be subtracted and an int as the initial value.
    assert np.array_equal(MetaAnalyser.subtraction_computing(8, [np.array([1, 2, 3]), 3, 1]),
                          np.array([3, 2, 1]))

    # Case of two arrays to be subtracted.
    assert np.array_equal(MetaAnalyser.subtraction_computing(np.array([10, 20, 30]), [np.array([1, 2, 3])]),
                          np.array([9, 18, 27]))
