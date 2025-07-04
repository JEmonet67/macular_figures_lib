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


def test_activation_time_computing():
    # Import a 2D array of valid VSDI activation times.
    with open(f"{path_data_test}/SpatialAnalyser/activation_time_VSDI_array.pyb", "rb") as file:
        activation_time_array_correct = pickle.load(file)

    # Creation of a 2D activation time array for test.
    activation_time_array = SpatialAnalyser.activation_time_computing(macular_dict_array_default.data["VSDI"],
                                                                      macular_dict_array_default.index["temporal"],
                                                                      0.001)

    assert np.array_equal(activation_time_array, activation_time_array_correct)


def test_dynamic_threshold_computing():
    # Creation of a correct dynamic threshold array.
    dynamic_threshold_correct = np.array([[[0.03, 0.03, 0.03]], [[0.05, 0.05, 0.05]]])

    # Initialisation of an empty array to contain an array of data.
    data_array_test = np.empty((2, 1, 3))

    # Filling the data array.
    data_array_test[0][0] = np.array([1, 2, 3])
    data_array_test[1][0] = np.array([3, 4, 5])

    # Calculation of the dynamic threshold for the data array.
    dynamic_threshold = SpatialAnalyser.dynamic_threshold_computing(data_array_test, 0.01)

    # Verification of the correct functioning of the dynamic threshold calculation.
    assert np.array_equal(dynamic_threshold, dynamic_threshold_correct)


def test_latency_computing():
    # Import a 2D array of valid VSDI horizontal latency.
    with open(f"{path_data_test}/SpatialAnalyser/horizontal_latency_VSDI_array.pyb", "rb") as file:
        horizontal_latency_array_correct = pickle.load(file)

    # Creation of a 2D horizontal latency array for test.
    horizontal_latency_array = SpatialAnalyser.latency_computing(macular_dict_array_default.data["VSDI"],
                                                                 macular_dict_array_default.index["temporal_centered"],
                                                                 0.001, "horizontal")

    assert np.array_equal(horizontal_latency_array, horizontal_latency_array_correct)

    # Import a 2D array of valid VSDI vertical latency.
    with open(f"{path_data_test}/SpatialAnalyser/vertical_latency_VSDI_array.pyb", "rb") as file:
        vertical_latency_array_correct = pickle.load(file)

    # Creation of a 2D vertical latency array for test.
    vertical_latency_array = SpatialAnalyser.latency_computing(macular_dict_array_default.data["VSDI"],
                                                               macular_dict_array_default.index["temporal_centered"],
                                                               0.001, "vertical")

    assert np.array_equal(vertical_latency_array, vertical_latency_array_correct)


def test_time_to_peak_computing():
    # Import a 2D array of valid VSDI time to peak.
    with open(f"{path_data_test}/SpatialAnalyser/time_to_peak_VSDI_array.pyb", "rb") as file:
        time_to_peak_array_correct = pickle.load(file)

    # Creation of a 2D time to peak array for test.
    time_to_peak_array = SpatialAnalyser.time_to_peak_computing(macular_dict_array_default.data["VSDI"],
                                                                macular_dict_array_default.index["temporal_ms"])

    assert np.array_equal(time_to_peak_array, time_to_peak_array_correct)


def test_peak_delay_computing():
    # Import a 2D array of valid VSDI horizontal peak delay.
    with open(f"{path_data_test}/SpatialAnalyser/horizontal_peak_delay_VSDI_array.pyb", "rb") as file:
        horizontal_peak_delay_array_correct = pickle.load(file)

    # Creation of a 2D horizontal peak delay array for test.
    horizontal_peak_delay_array = SpatialAnalyser.peak_delay_computing(
        macular_dict_array_default.data["VSDI"],
        macular_dict_array_default.index["temporal_centered_ms"],
        "horizontal")

    assert np.array_equal(horizontal_peak_delay_array, horizontal_peak_delay_array_correct)

    # Import a 2D array of valid VSDI vertical latency.
    with open(f"{path_data_test}/SpatialAnalyser/vertical_peak_delay_VSDI_array.pyb", "rb") as file:
        vertical_peak_delay_array_correct = pickle.load(file)

    # Creation of a 2D vertical latency array for test.
    vertical_peak_delay_array = SpatialAnalyser.peak_delay_computing(
        macular_dict_array_default.data["VSDI"], macular_dict_array_default.index["temporal_centered"], "vertical")

    assert np.array_equal(vertical_peak_delay_array, vertical_peak_delay_array_correct)


def test_peak_amplitude_computing():
    # Import a 2D array of valid VSDI time to peak.
    with open(f"{path_data_test}/SpatialAnalyser/amplitude_VSDI_array.pyb", "rb") as file:
        amplitude_array_correct = pickle.load(file)

    # Creation of a 2D amplitude array for test.
    amplitude_array = SpatialAnalyser.peak_amplitude_computing(macular_dict_array_default.data["VSDI"])

    assert np.array_equal(amplitude_array, amplitude_array_correct)


def test_initial_amplitude_computing():
    # Initialisation of an empty array to contain an array of data.
    data_array_test = np.empty((2, 3, 3))

    # Filling the data array.
    data_array_test[0][0] = np.array([4, 2, 3])
    data_array_test[0][1] = np.array([3, 4, 5])
    data_array_test[0][2] = np.array([4, 2, 3])
    data_array_test[1][0] = np.array([3, 4, 5])
    data_array_test[1][1] = np.array([4, 2, 3])
    data_array_test[1][2] = np.array([3, 4, 5])

    # Creation of a 2D amplitude array for test.
    amplitude_array = SpatialAnalyser.initial_amplitude_computing(data_array_test)

    assert np.array_equal(amplitude_array, np.array([[4, 3, 4], [3, 4, 3]]))


def test_spatial_mean_computing():
    # Initialisation of an empty array to contain an array of data.
    data_array_test = np.empty((2, 3, 3))

    # Filling the data array.
    data_array_test[0][0] = np.array([4, 2, 3])
    data_array_test[0][1] = np.array([3, 4, 5])
    data_array_test[0][2] = np.array([4, 2, 3])
    data_array_test[1][0] = np.array([3, 4, 5])
    data_array_test[1][1] = np.array([4, 2, 3])
    data_array_test[1][2] = np.array([3, 4, 5])

    # Case of a spatial mean array with a 3D array along the first axis.
    spatial_mean_array = SpatialAnalyser.spatial_mean_computing(data_array_test, 0)
    assert np.array_equal(spatial_mean_array, np.array([3.3333, 3.6667]))

    # Case of a spatial mean array with a 3D array along the first axis.
    spatial_mean_array = SpatialAnalyser.spatial_mean_computing(data_array_test, 2)
    assert np.array_equal(spatial_mean_array, np.array([3.5, 3, 4]))

    # Case of a spatial mean array with a 2D array.
    spatial_mean_array = SpatialAnalyser.spatial_mean_computing(data_array_test[:, :, 0], 1)
    assert np.array_equal(spatial_mean_array, np.array([3.5, 3.5, 3.5]))

    # Case of a spatial mean array with a 2D array but with a wrong axis.
    try:
        SpatialAnalyser.spatial_mean_computing(data_array_test[:, :, 0], 2)
        assert False
    except IndexError:
        assert True