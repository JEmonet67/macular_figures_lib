import os
import pickle

import numpy as np

from src.data_manager.DataPreprocessor import DataPreprocessor

# Get data for test from relative path.
path_data_test = os.path.normpath(f"{os.getcwd()}/../data_test/data_manager/")

# Import of a reduced MacularDictArray control with only the 100 first rows.
path_pyb_file_head100 = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_copy_0f.pyb"
with open(path_pyb_file_head100, "rb") as file_head100:
    macular_dict_array_head100 = pickle.load(file_head100)

# Initialisation of the MacularDictArray for tests with the values of the reduced control MacularDictArray.
with open(path_pyb_file_head100, "rb") as file_test:
    macular_dict_array_test = pickle.load(file_test)

# Import of a reduced MacularDictArray control with only the 100 first rows and binning 0.0016s.
path_pyb_file_head100_binning = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_binning0,0016_0f.pyb"
with open(path_pyb_file_head100_binning, "rb") as file_head100_binning:
    macular_dict_array_head100_binning = pickle.load(file_head100_binning)

# Import of a reduced MacularDictArray control with only the 100 first rows and VSDI.
path_pyb_file_head100_VSDI = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_VSDI_0f.pyb"
with open(f"{path_pyb_file_head100_VSDI}", "rb") as file_VSDI:
    macular_dict_array_head100_VSDI = pickle.load(file_VSDI)

# Import of a reduced MacularDictArray control with only the 100 first rows and centered.
path_pyb_file_head100_centered = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_temporalCentered_0f.pyb"
with open(f"{path_pyb_file_head100_centered}", "rb") as file_centered:
    macular_dict_array_head100_centered = pickle.load(file_centered)

# Import of a SMS MacularDictArray with 1440Hz frame rate and 200°/s bar speed.
path_pyb_file_SMS_default = f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0134_barSpeed200dps_0f.pyb"
with open(f"{path_pyb_file_SMS_default}", "rb") as file_SMS:
    macular_dict_array_SMS = pickle.load(file_SMS)

preprocessor = DataPreprocessor()

# Dictionaries generation.
name_file_head100 = "RC_RM_dSGpCP0026_barSpeed6dps_head100_0f"
dict_simulation_head100 = {
    "path_csv": f"../data_test/data_manager/{name_file_head100}.csv",
    "path_pyb": f"../data_test/data_manager/{name_file_head100}.pyb",
    "n_cells_x": 83,
    "n_cells_y": 15,
    "dx": 0.225,
    "delta_t": 0.0167,
    "end": "max",
    "speed": 6,
    "size_bar": 0.67,
    "axis": "horizontal"
}
dict_preprocessing_default = {}


def test_array_edge_cropping():
    # Case without cropping the edges.
    cropped_imbricated_array = preprocessor.array_edge_cropping(macular_dict_array_head100.data[
                                                                    "FiringRate_GanglionGainControl"])
    assert cropped_imbricated_array.shape[0] == 15
    assert cropped_imbricated_array.shape[1] == 83
    assert cropped_imbricated_array.shape[2] == 99

    # Case of cropping on both spatial axis.
    cropped_array = preprocessor.array_edge_cropping(macular_dict_array_head100.data["FiringRate_GanglionGainControl"],
                                                     {"x_min_edge": 5, "x_max_edge": 2, "y_min_edge": 1,
                                                      "y_max_edge": 3})
    assert cropped_array.shape[0] == 11
    assert cropped_array.shape[1] == 76
    assert cropped_array.shape[2] == 99

    # Case of cropping on all axis.
    cropped_array = preprocessor.array_edge_cropping(macular_dict_array_head100.data["FiringRate_GanglionGainControl"],
                                                     {"x_min_edge": 5, "x_max_edge": 2, "y_min_edge": 1,
                                                      "y_max_edge": 3, "t_min_edge": 12, "t_max_edge": 17})
    assert cropped_array.shape[0] == 11
    assert cropped_array.shape[1] == 76
    assert cropped_array.shape[2] == 70


def test_crop_imbricated_array():
    # Definition of an imbricated array to be cropped.
    array_to_crop = np.array([[np.array([0.42, 0.8, 0.99]), np.array([0.72, 0.97, 0.75]), np.array([0., 0.31, 0.28]),
                               np.array([0.3, 0.69, 0.79])], [np.array([0.15, 0.88, 0.1]), np.array([0.09, 0.89, 0.45]),
                                                              np.array([0.19, 0.09, 0.91]),
                                                              np.array([0.35, 0.04, 0.29])],
                              [np.array([0.4, 0.17, 0.29]), np.array([0.54, 0.88, 0.13]), np.array([0.42, 0.1, 0.02]),
                               np.array([0.69, 0.42, 0.68])],
                              [np.array([0.2, 0.96, 0.21]), np.array([0.88, 0.53, 0.27]),
                               np.array([0.03, 0.69, 0.49]), np.array([0.67, 0.32, 0.05])],
                              [np.array([0.42, 0.69, 0.57]), np.array([0.56, 0.83, 0.15]), np.array([0.14, 0.02, 0.59]),
                               np.array([0.2, 0.75, 0.7])]])

    # Case of crop with fixed edges.
    cropped_array = preprocessor.crop_imbricated_array(array_to_crop,"fixed_edge", {})
    assert cropped_array.shape[0] == 5
    assert cropped_array.shape[1] == 4
    assert cropped_array.shape[2] == 3

    # Construction of a correctly cropped array with threshold.
    threshold_cropped_array_correct = np.array([
        [np.array([0.42, 0.8, 0.99]), np.array([0.72, 0.97, 0.75]), np.array([0.31, 0.28]),
         np.array([0.3, 0.69, 0.79])],
        [np.array([0.15, 0.88, 0.1]), np.array([0.89, 0.45]), np.array([0.19, 0.91]), np.array([0.35, 0.29])],
        [np.array([0.4, 0.17, 0.29]), np.array([0.54, 0.88, 0.13]), np.array([0.42, 0.1]),
         np.array([0.69, 0.42, 0.68])],
        [np.array([0.2, 0.96, 0.21]), np.array([0.88, 0.53, 0.27]), np.array([0.69, 0.49]), np.array([0.67, 0.32])],
        [np.array([0.42, 0.69, 0.57]), np.array([0.56, 0.83, 0.15]), np.array([0.14, 0.59]),
         np.array([0.2, 0.75, 0.7])]
    ], dtype=object)

    # Case of crop with a threshold.
    cropped_array = preprocessor.crop_imbricated_array(array_to_crop, "threshold",
                                                       {"threshold": 0.1, "axis": 0})

    for i in range(len(threshold_cropped_array_correct)):
        for j in range(len(threshold_cropped_array_correct[i])):
            assert np.array_equal(cropped_array[i][j], threshold_cropped_array_correct[i][j])

    # Construction of a correctly cropped array with threshold dependent on maximum values.
    threshold_cropped_array_correct = np.array([
        [np.array([4., 0.4]), np.array([0.15, 0.09, 0.19, 0.35]), np.array([0.4, 0.54, 0.42, 0.69]),
         np.array([0.2, 0.88, 0.67]), np.array([0.42, 0.56, 0.14, 0.2])]
    ], dtype=object)

    # Definition of a simple array to be cropped with a ratio threshold.
    array_to_crop = np.array([
        [[4, 0.4, 0.1, 0.3], [0.15, 0.09, 0.19, 0.35], [0.4, 0.54, 0.42, 0.69], [0.2, 0.88, 0.03, 0.67],
         [0.42, 0.56, 0.14, 0.2]]
    ])

    # Case of crop with a threshold based on a ratio of the maximum value.
    cropped_array = preprocessor.crop_imbricated_array(array_to_crop,"max_ratio_threshold",
                                                       {"ratio_threshold": 0.1, "axis": 2})
    for i in range(len(threshold_cropped_array_correct)):
        for j in range(len(threshold_cropped_array_correct[i])):
            assert np.array_equal(cropped_array[i][j], threshold_cropped_array_correct[i][j])


def test_fixed_edge_cropping():
    # Definition of an imbricated array to test cropping.
    imbricated_array = np.array([
        [np.array([0.42, 0.72, 0., 0.3]), np.array([0.15, 0.09, 0.19, 0.35]), np.array([0.4, 0.54, 0.42, 0.69]),
         np.array([0.2, 0.88, 0.03, 0.67]), np.array([0.42, 0.56, 0.14, 0.2])],
        [np.array([0.8, 0.97, 0.31, 0.69]), np.array([0.88, 0.89, 0.09, 0.04]), np.array([0.17, 0.88, 0.1, 0.42]),
         np.array([0.96, 0.53, 0.69, 0.32]), np.array([0.69, 0.83, 0.02, 0.75])],
        [np.array([0.99, 0.75, 0.28, 0.79]), np.array([0.1, 0.45, 0.91, 0.29]), np.array([0.29, 0.13, 0.02, 0.68]),
         np.array([0.21, 0.27, 0.49, 0.05]), np.array([0.57, 0.15, 0.59])]
    ], dtype=object)

    # Case of cropping edge on imbricated arrays witj null edges.
    cropped_array = preprocessor.fixed_edge_cropping(imbricated_array.copy())

    assert cropped_array.shape[0] == 3
    assert cropped_array.shape[1] == 5
    assert cropped_array[0, 0].shape[0] == 4

    # Case of cropping edge on imbricated arrays with non-null edges.
    cropped_array = preprocessor.fixed_edge_cropping(imbricated_array.copy(), {"edge_start": 1, "edge_end": 1})

    assert cropped_array.shape[0] == 3
    assert cropped_array.shape[1] == 5
    assert cropped_array[0, 0].shape[0] == 2


def test_threshold_cropping():
    # Definition of an imbricated array to test threshold cropping.
    imbricated_array_test = np.array([
        [np.array([0.42, 0.72, 0., 0.3]), np.array([0.15, 0.09, 0.19, 0.35]), np.array([0.4, 0.54, 0.42, 0.69]),
         np.array([0.2, 0.88, 0.03, 0.67]), np.array([0.42, 0.56, 0.14, 0.2])],
        [np.array([0.8, 0.97, 0.31, 0.69]), np.array([0.88, 0.89, 0.09, 0.04]), np.array([0.17, 0.88, 0.1, 0.42]),
         np.array([0.96, 0.53, 0.69, 0.32]), np.array([0.69, 0.83, 0.02, 0.75])],
        [np.array([0.99, 0.75, 0.28, 0.79]), np.array([0.1, 0.45, 0.91, 0.29]), np.array([0.29, 0.13, 0.02, 0.68]),
         np.array([0.21, 0.27, 0.49, 0.05]), np.array([0.57, 0.15, 0.59, 0.7])]
    ], dtype=object)

    # Creation of an imbricated array correctly cropped by threshold.
    threshold_cropped_array_correct = np.array([
        [np.array([0.42, 0.72, 0.3]), np.array([0.15, 0.19, 0.35]), np.array([0.4, 0.54, 0.42, 0.69]),
         np.array([0.2, 0.88, 0.67]),
         np.array([0.42, 0.56, 0.14, 0.2])],
        [np.array([0.8, 0.97, 0.31, 0.69]), np.array([0.88, 0.89]), np.array([0.17, 0.88, 0.1, 0.42]),
         np.array([0.96, 0.53, 0.69, 0.32]), np.array([0.69, 0.83, 0.75])],
        [np.array([0.99, 0.75, 0.28, 0.79]), np.array([0.1, 0.45, 0.91, 0.29]), np.array([0.29, 0.13, 0.68]),
         np.array([0.21, 0.27, 0.49]), np.array([0.57, 0.15, 0.59, 0.7])]
    ], dtype=object)

    # Case of cropping edge on imbricated arrays with threshold.
    cropped_array = preprocessor.threshold_cropping(imbricated_array_test, 0.1)

    for i in range(len(threshold_cropped_array_correct)):
        for j in range(len(threshold_cropped_array_correct[i])):
            assert np.array_equal(cropped_array[i][j], threshold_cropped_array_correct[i][j])


def test_max_ratio_threshold_cropping():
    # Definition of an imbricated array to test maximal ratio threshold cropping.
    imbricated_array_test = np.array([
        [np.array([4, 0.4, 0.1, 0.3]), np.array([0.15, 0.09, 0.19, 0.35]), np.array([0.4, 0.54, 0.42, 0.69]),
         np.array([0.2, 0.88, 0.03, 0.67]),
         np.array([0.42, 0.56, 0.14, 0.2])],
        [np.array([0.8, 0.97, 0.31, 0.69]), np.array([0.88, 0.89, 0.09, 0.04]), np.array([0.17, 0.88, 0.1, 0.42]),
         np.array([0.96, 0.53, 0.69, 0.32]),
         np.array([0.69, 0.83, 0.02, 0.75])]
    ])

    # Creation of an imbricated array correctly cropped by maximal ratio threshold.
    threshold_cropped_array_correct = np.array([
        [np.array([4., 0.4]), np.array([0.15, 0.09, 0.19, 0.35]), np.array([0.4, 0.54, 0.42, 0.69]),
         np.array([0.2, 0.88, 0.67]), np.array([0.42, 0.56, 0.14, 0.2])],
        [np.array([0.8, 0.97, 0.31, 0.69]), np.array([0.88, 0.89, 0.09]), np.array([0.17, 0.88, 0.1, 0.42]),
         np.array([0.96, 0.53, 0.69, 0.32]), np.array([0.69, 0.83, 0.75])]
    ], dtype=object)

    # Case of cropping edge on imbricated arrays with maximal ratio threshold.
    cropped_array = preprocessor.max_ratio_threshold_cropping(imbricated_array_test, 0.1)

    for i in range(len(threshold_cropped_array_correct)):
        for j in range(len(threshold_cropped_array_correct[i])):
            assert np.array_equal(cropped_array[i][j], threshold_cropped_array_correct[i][j])


def test_transform_3d_array_to_imbricated_arrays():
    # Set the randomness for testing.
    np.random.seed(1)

    # Test array to be converted to a imbricated array.
    array_to_transform = np.random.rand(*(1, 2, 3)).round(2)

    # Array correctly imbricated on its vertical axis.
    vertical_imbricated_array_correct = np.array([[np.array([0.42]), np.array([0.72]), np.array([0.0])],
                                                  [np.array([0.3]), np.array([0.15]), np.array([0.09])]])
    # Imbricating of the vertical axis of the test array.
    imbricated_array = DataPreprocessor.transform_3d_array_to_imbricated_arrays(array_to_transform, "vertical")

    # Verification of the vertically imbricated array.
    for i in range(len(vertical_imbricated_array_correct)):
        for j in range(len(vertical_imbricated_array_correct[i])):
            assert np.array_equal(imbricated_array[i][j], vertical_imbricated_array_correct[i][j])

    # Array correctly imbricated on its horizontal axis.
    horizontal_imbricated_array_correct = np.array([[np.array([0.42, 0.3]), np.array([0.72, 0.15]),
                                                     np.array([0.0, 0.09])]])
    # Imbricating of the horizontal axis of the test array.
    imbricated_array = DataPreprocessor.transform_3d_array_to_imbricated_arrays(array_to_transform, "horizontal")

    # Verification of the horizontally imbricated array.
    for i in range(len(horizontal_imbricated_array_correct)):
        for j in range(len(horizontal_imbricated_array_correct[i])):
            assert np.array_equal(imbricated_array[i][j], horizontal_imbricated_array_correct[i][j])

    # Array correctly imbricated on its temporal axis.
    temporal_imbricated_array_correct = np.array([[np.array([0.42, 0.72, 0.0]), np.array([0.3, 0.15, 0.09])]])
    # Imbricating of the temporal axis of the test array.
    imbricated_array = DataPreprocessor.transform_3d_array_to_imbricated_arrays(array_to_transform, "temporal")

    # Verification of the temporally imbricated array.
    for i in range(len(temporal_imbricated_array_correct)):
        for j in range(len(temporal_imbricated_array_correct[i])):
            assert np.array_equal(imbricated_array[i][j], temporal_imbricated_array_correct[i][j])


def test_imbricated_arrays_axis_averaging():
    # Import of a SMS MacularDictArray with 1440Hz frame rate and 200°/s bar speed with sections already averaged.
    path_pyb_file_SMS_default = f"{path_data_test}/RC_RM_dSGpCP0134_barSpeed200dps_mean_sectioned_0f.pyb"
    with open(f"{path_pyb_file_SMS_default}", "rb") as file_SMS:
        macular_dict_array_SMS_mean_sectioned = pickle.load(file_SMS)

    # Transform the test array into an imbricated array along the vertical axis.
    vertical_imbricated_array = DataPreprocessor.transform_3d_array_to_imbricated_arrays(
        macular_dict_array_SMS.data["VSDI"], "vertical")

    # Execution of the average function of the vertical section.
    vertical_mean_section = preprocessor.imbricated_arrays_axis_averaging(vertical_imbricated_array)

    # Test in the case of a vertical averaged section.
    assert np.array_equal(vertical_mean_section, macular_dict_array_SMS_mean_sectioned.data["vertical_mean_section"].round(4))

    # Transform the test array into an imbricated array along the horizontal axis.
    horizontal_imbricated_array = DataPreprocessor.transform_3d_array_to_imbricated_arrays(
        macular_dict_array_SMS.data["VSDI"], "horizontal")

    # Execution of the average function of the horizontal section.
    horizontal_mean_section = preprocessor.imbricated_arrays_axis_averaging(horizontal_imbricated_array)

    # Test in the case of a horizontally averaged section.
    assert np.array_equal(horizontal_mean_section, macular_dict_array_SMS_mean_sectioned.data["horizontal_mean_section"].round(4))

    # Transform the test array into an imbricated array along the temporal axis.
    temporal_imbricated_array = DataPreprocessor.transform_3d_array_to_imbricated_arrays(
        macular_dict_array_SMS.data["VSDI"], "temporal")

    # Execution of the average function of the temporal section.
    temporal_mean_section = preprocessor.imbricated_arrays_axis_averaging(temporal_imbricated_array)

    # Test in the case of a temporal averaged section.
    assert np.array_equal(temporal_mean_section, macular_dict_array_SMS_mean_sectioned.data["temporal_mean_section"].round(4))
