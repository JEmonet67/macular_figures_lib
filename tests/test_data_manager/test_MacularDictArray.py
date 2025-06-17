import os
import pickle
import re

import numpy as np

from src.data_manager.MacularDictArray import MacularDictArray
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

# Import of simplified MacularDictArray filled with 1 value.
path_pyb_file_head100_npones = f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0026_barSpeed6dps_head100_npOnes_0f.pyb"
with open(path_pyb_file_head100_npones, "rb") as file:
    macular_dict_array_head100_npones = pickle.load(file)

# Import of a reduced MacularDictArray control with only the 3000 first rows.
path_pyb_file_head3000 = f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0026_barSpeed6dps_head3000_copy_0f.pyb"
with open(path_pyb_file_head3000, "rb") as file_head3000:
    macular_dict_array_head3000 = pickle.load(file_head3000)

# Initialisation of the MacularDictArray for tests with the values of the reduced control MacularDictArray.
path_pyb_file_head100_30dps = f"{path_data_test}/RC_RM_dSGpCP0033_barSpeed30dps_head100_0f.pyb"
with open(f"{path_pyb_file_head100_30dps}", "rb") as file_30dps:
    macular_dict_array_head100_30dps = pickle.load(file_30dps)

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

path_pyb_file_head3000_default = f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0026_barSpeed6dps_default_head3000_0f.pyb"
# Import of the default MacularDictArray to be compared with preprocessing.
with open(path_pyb_file_head3000_default, "rb") as file_default:
    macular_dict_array_default_head3000 = pickle.load(file_default)

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
    # "transient": 0
}
dict_preprocessing_default = {}

name_file_head100_30dps = "RC_RM_dSGpCP0033_barSpeed30dps_head100_0f"
dict_simulation_head100_30dps = {
    "path_csv": f"../data_test/data_manager/{name_file_head100_30dps}.csv",
    "path_pyb": f"../data_test/data_manager/{name_file_head100_30dps}.pyb",
    "n_cells_x": 83,
    "n_cells_y": 15,
    "dx": 0.225,
    "delta_t": 0.0167,
    "end": "max",
    "speed": 30,
    "size_bar": 0.67,
    "axis": "horizontal"
}

dict_simulation_SMS = {
    "path_csv": f"../data_test/data_manager/RC_RM_dSGpCP0134_barSpeed200dps_0f.csv",
    "path_pyb": f"../data_test/data_manager/RC_RM_dSGpCP0134_barSpeed200dps_mean_sectioned_0f.pyb",
    "n_cells_x": 83,
    "n_cells_y": 15,
    "dx": 0.225,
    "delta_t": 0.0007,
    "end": "max",
    "speed": 200,
    "size_bar": 1.08,
    "axis": "horizontal"
}

dict_preprocessing_SMS = {
    "temporal_centering": True,
    "spatial_x_centering": True,
    "spatial_y_centering": True,
    "binning": 0.0007,
    "VSDI": True,
    "derivative": {"VSDI": 31,
                   "FiringRate_GanglionGainControl": 31},
    "temporal_index_ms": 1000,
    "spatial_index_mm_retina": 0.3,
    "spatial_index_mm_cortex": 3,
    "edge": (5, 0),
    "mean_sections": {
        "horizontal": [{"measurement": "VSDI", "cropping_type": "", "cropping_dict": {}},
                       {"measurement": "VSDI", "cropping_type": "fixed_edge", "cropping_dict": {"x_min_edge": 13,
                                                                                                "x_max_edge": 13}},
                       {"measurement": "VSDI", "cropping_type": "threshold", "cropping_dict": {"threshold": 0.005}},
                       {"measurement": "VSDI", "cropping_type": "max_ratio_threshold",
                        "cropping_dict": {"ratio_threshold": 0.5}}],
        "vertical": [{"measurement": "VSDI", "cropping_type": "", "cropping_dict": {}},
                     {"measurement": "VSDI", "cropping_type": "fixed_edge", "cropping_dict": {"y_min_edge": 2,
                                                                                              "y_max_edge": 2}},
                     {"measurement": "VSDI", "cropping_type": "threshold", "cropping_dict": {"threshold": 0.005}},
                     {"measurement": "VSDI", "cropping_type": "max_ratio_threshold",
                      "cropping_dict": {"ratio_threshold": 0.5}}],
        "temporal": [{"measurement": "VSDI", "cropping_type": "", "cropping_dict": {}}]
    }
}


def test_init(monkeypatch):
    # Delete the pyb file of the MacularDictArray to be used for the test if it exists.
    try:
        os.remove(f"{path_data_test}/{name_file_head100}.pyb")
    except FileNotFoundError:
        pass

    # Creation of a new MacularDictArray.
    macular_dict_array_test = MacularDictArray(dict_simulation_head100, dict_preprocessing_default)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100)

    # Checking the save.
    assert os.path.exists(f"{path_data_test}/{name_file_head100}.pyb")

    # Reimport of the newly created MacularDictArray.
    macular_dict_array_test = MacularDictArray(dict_simulation_head100, dict_preprocessing_default)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100)

    # Reimport with a json/pyb conflict.
    monkeypatch.setattr('builtins.input', lambda _: "json")
    dict_simulation_head100_conflict = dict_simulation_head100.copy()
    dict_simulation_head100_conflict["speed"] = 30
    MacularDictArray(dict_simulation_head100_conflict, dict_preprocessing_default)
    with open(dict_simulation_head100["path_pyb"], "rb") as file:
        macular_dict_array_conflict = pickle.load(file)
    assert not MacularDictArray.equal(macular_dict_array_conflict, macular_dict_array_head100)


def test_equal():
    # Import of the second MacularDictArray for comparison.
    with open(path_pyb_file_head100, "rb") as file:
        macular_dict_array_head100_copy = pickle.load(file)

    # Case of a tie between the two MacularDictArrays.
    assert MacularDictArray.equal(macular_dict_array_head100, macular_dict_array_head100_copy)

    # Cases where only the path_pyb attribute differs.
    macular_dict_array_head100_copy.path_pyb = "../data_test/new_path.pyb"
    assert macular_dict_array_head100_copy.path_pyb != macular_dict_array_head100.path_pyb
    assert MacularDictArray.equal(macular_dict_array_head100, macular_dict_array_head100_copy)
    macular_dict_array_head100_copy.path_pyb = macular_dict_array_head100.path_pyb

    # Case with one more attribute.
    macular_dict_array_head100_copy.new_attribute = "Test"
    assert not MacularDictArray.equal(macular_dict_array_head100, macular_dict_array_head100_copy)
    del macular_dict_array_head100_copy.new_attribute

    # Case of an additional output in the data.
    macular_dict_array_head100_copy._data["new"] = 0
    assert not MacularDictArray.equal(macular_dict_array_head100, macular_dict_array_head100_copy)
    del macular_dict_array_head100_copy._data["new"]

    # Case of a difference in dict_simulation.
    macular_dict_array_head100_copy.dict_simulation["speed"] = 10
    assert not MacularDictArray.equal(macular_dict_array_head100, macular_dict_array_head100_copy)
    macular_dict_array_head100_copy.dict_simulation["speed"] = 6

    # Case of a difference in the data.
    macular_dict_array_head100_copy._data["FiringRate_GanglionGainControl"] = np.zeros((43, 15, 2943))
    assert not MacularDictArray.equal(macular_dict_array_head100, macular_dict_array_head100_copy)


def test_equal_dict_array():
    # Case of equality between two dict_array.
    assert MacularDictArray.equal_dict_array(macular_dict_array_head100.data, macular_dict_array_head100.data)
    assert MacularDictArray.equal_dict_array(macular_dict_array_head100.index, macular_dict_array_head100.index)

    # Case with different sizes of MacularDictArray measurement arrays.
    assert not MacularDictArray.equal_dict_array(macular_dict_array_head100.data,
                                                 macular_dict_array_head100_binning.data)
    assert not MacularDictArray.equal_dict_array(macular_dict_array_head100.index,
                                                 macular_dict_array_head100_binning.index)

    # Case with an additional measurement in the MacularDictArray data dictionary.
    assert not MacularDictArray.equal_dict_array(macular_dict_array_head100.data,
                                                 macular_dict_array_head100_VSDI.data)

    # Case with an extra index in the MacularDictArray.
    assert not MacularDictArray.equal_dict_array(macular_dict_array_head100.index,
                                                 macular_dict_array_head100_centered.index)


def test_cleaning_dict_preprocessing():
    # Case of an empty dictionary.
    macular_dict_array_test._dict_preprocessing = {}
    assert macular_dict_array_test.cleaning_dict_preprocessing(macular_dict_array_test.dict_preprocessing) == {}

    # Case of a dictionary without any preprocess.
    macular_dict_array_test._dict_preprocessing = {"temporal_centering": False, "binning": False, "derivative": False,
                                                   "edge": False}
    assert macular_dict_array_test.cleaning_dict_preprocessing(macular_dict_array_test.dict_preprocessing) == {}

    # Case of True and False values.
    macular_dict_array_test._dict_preprocessing = {"temporal_centering": True, "binning": False, "derivative": False,
                                                   "edge": False}
    assert (macular_dict_array_test.cleaning_dict_preprocessing(macular_dict_array_test.dict_preprocessing) ==
            {"temporal_centering": True})

    # Case of a float.
    macular_dict_array_test._dict_preprocessing = {"temporal_centering": False, "binning": 0.0016, "derivative": False,
                                                   "edge": False}
    assert (macular_dict_array_test.cleaning_dict_preprocessing(macular_dict_array_test.dict_preprocessing) ==
            {"binning": 0.0016})

    # Case of a dictionary
    macular_dict_array_test._dict_preprocessing = {"temporal_centering": False, "binning": False,
                                                   "derivative": {"VSDI": 3, "FiringRate_GanglionGainControl": 1},
                                                   "edge": False}
    assert (macular_dict_array_test.cleaning_dict_preprocessing(macular_dict_array_test.dict_preprocessing) ==
            {"derivative": {"VSDI": 3, "FiringRate_GanglionGainControl": 1}})

    # Case of a tuple.
    macular_dict_array_test._dict_preprocessing = {"temporal_centering": False, "binning": False, "derivative": False,
                                                   "edge": (5, 0)}
    assert (macular_dict_array_test.cleaning_dict_preprocessing(macular_dict_array_test.dict_preprocessing) ==
            {"edge": (5, 0)})


def test_managing_pre_existing_file():
    # Deletion of the file to be used for the test, if it exists.
    try:
        os.remove(dict_simulation_head100['path_pyb'])
    except FileNotFoundError:
        pass

    # Test of the construction of a new MacularDictArray in the absence of a file to import.
    macular_dict_array_test.managing_pre_existing_file(dict_simulation_head100.copy(), dict_preprocessing_default)

    # Case of non-existent file.
    assert MacularDictArray.equal(macular_dict_array_head100, macular_dict_array_test)
    macular_dict_array_head100.save()

    # Update of the MacularDictArray from an existing pyb file.
    dict_simulation_head100_30dps = macular_dict_array_head100_30dps.dict_simulation.copy()
    dict_simulation_head100_30dps[
        "path_csv"] = f"../data_test/data_manager/RC_RM_dSGpCP0033_barSpeed30dps_head100_0f.csv"
    dict_simulation_head100_30dps[
        "path_pyb"] = f"../data_test/data_manager/RC_RM_dSGpCP0033_barSpeed30dps_head100_0f.pyb"
    macular_dict_array_test.managing_pre_existing_file(dict_simulation_head100_30dps, dict_preprocessing_default)

    # Case of the existing file.
    assert MacularDictArray.equal(macular_dict_array_head100_30dps, macular_dict_array_test)

    # Case of direct import of an existing file without comparison.
    macular_dict_array_test.managing_pre_existing_file({"path_pyb": path_pyb_file_head100}, {})
    assert MacularDictArray.equal(macular_dict_array_head100, macular_dict_array_test)


def test_checking_difference_file_json(monkeypatch):
    # Import of MacularDictArray to be compared.
    with open(f"{path_pyb_file_head100_30dps}", "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Cases where there is no difference.
    error = macular_dict_array_test.checking_difference_file_json(dict_simulation_head100_30dps,
                                                                  dict_preprocessing_default)
    assert not error

    # Case of incorrect user input.
    monkeypatch.setattr('builtins.input', lambda _: "No")
    try:
        macular_dict_array_test.checking_difference_file_json(dict_simulation_head100, dict_preprocessing_default)
        assert False
    except ValueError:
        assert True

    # Cases where the path_pyb differs.
    dict_simulation_head100_30dps_copy = dict_simulation_head100_30dps.copy()
    dict_simulation_head100_30dps_copy["path_pyb"] = "../data_test/RC_RM_dSGpCP0033_barSpeed30dps_head100_0f.pyb"
    assert dict_simulation_head100_30dps_copy["path_pyb"] != macular_dict_array_test._path_pyb
    error = macular_dict_array_head100.checking_difference_file_json(dict_simulation_head100,
                                                                     dict_preprocessing_default)
    assert not error

    # # Case of keeping the pyb when path_csv differs.
    monkeypatch.setattr('builtins.input', lambda _: "pyb")
    dict_simulation_head100_30dps_copy = dict_simulation_head100_30dps.copy()
    dict_simulation_head100_30dps_copy["path_csv"] = "../data_test/RC_RM_dSGpCP0033_barSpeed30dps_head100_0f.csv"
    error = macular_dict_array_test.checking_difference_file_json(dict_simulation_head100, dict_preprocessing_default)
    assert error
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_30dps)

    # Case of keeping the pyb when dict_simulation differs.
    dict_simulation_head100_30dps_copy = dict_simulation_head100_30dps.copy()
    dict_simulation_head100_30dps_copy["speed"] = 6
    error = macular_dict_array_test.checking_difference_file_json(dict_simulation_head100, dict_preprocessing_default)
    assert error
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_30dps)

    # Case of keeping the pyb when dict_preprocessing differs.
    dict_preprocessing_modified = macular_dict_array_head100_30dps.dict_preprocessing.copy()
    dict_preprocessing_modified["VSDI"] = False
    error = macular_dict_array_test.checking_difference_file_json(dict_simulation_head100, dict_preprocessing_modified)
    assert error
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_30dps)

    # Keep the json in case of difference file/json.
    monkeypatch.setattr('builtins.input', lambda _: "json")
    error = macular_dict_array_test.checking_difference_file_json(dict_simulation_head100.copy(),
                                                                  dict_preprocessing_default)
    assert error
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100)


def test_dict_simulation_getter():
    dict_simulation_head100_no_path = dict_simulation_head100.copy()
    del dict_simulation_head100_no_path["path_csv"]
    del dict_simulation_head100_no_path["path_pyb"]
    assert macular_dict_array_head100.dict_simulation == dict_simulation_head100_no_path


def test_dict_simulation_setter():
    # Import of MacularDictArray to be modified.
    with open(f"{path_pyb_file_head100_30dps}", "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Use of dict_simulation setter.
    dict_simulation_head100_copy = dict_simulation_head100.copy()
    macular_dict_array_test.dict_simulation = dict_simulation_head100_copy

    # Case of the correct modification of simulation dictionary.
    dict_simulation_head100_no_path = dict_simulation_head100.copy()
    del dict_simulation_head100_no_path["path_csv"]
    del dict_simulation_head100_no_path["path_pyb"]
    assert macular_dict_array_test.dict_simulation == dict_simulation_head100_no_path

    # Case of the correct update of the MacularDictArray with the new simulation dictionary.
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100)


def test_dict_preprocessing_getter():
    assert macular_dict_array_head100.dict_preprocessing == dict_preprocessing_default


def test_dict_preprocessing_setter():
    # Import of MacularDictArray to be modified.
    with open(path_pyb_file_head100, "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Use of dict_preprocessing setter.
    dict_preprocessing_head100_vsdi = macular_dict_array_head100.dict_preprocessing.copy()
    dict_preprocessing_head100_vsdi["VSDI"] = True
    macular_dict_array_test.dict_preprocessing = dict_preprocessing_head100_vsdi

    # Case of the correct modification of preprocessing dictionary.
    assert macular_dict_array_test.dict_preprocessing == dict_preprocessing_head100_vsdi

    # Case of the correct update of the MacularDictArray with the new preprocessing dictionary.
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_VSDI)


def test_path_csv_getter():
    # Case of the relative path stored in the attribute.
    assert macular_dict_array_head100._path_csv == dict_simulation_head100["path_csv"]

    # Case of the absolute path accessible from the attribute.
    assert (macular_dict_array_head100.path_csv == f"{path_data_test}/{name_file_head100}.csv")


def test_path_csv_setter():
    macular_dict_array_test.path_csv = "../data_test/data_manager/test.csv"
    assert macular_dict_array_test._path_csv == "../data_test/data_manager/test.csv"


def test_path_pyb_getter():
    # Case of the relative path stored in the attribute.
    assert macular_dict_array_head100._path_pyb == dict_simulation_head100["path_pyb"].replace("_head100_",
                                                                                               "_head100_copy_")

    # Case of the absolute path accessible from the attribute.
    assert (macular_dict_array_head100.path_pyb == f"{path_data_test}/{name_file_head100}.pyb"
            .replace("_head100_", "_head100_copy_"))


def test_path_pyb_setter():
    # Case of the use of the setter.
    new_path_pyb = "../data_test/data_manager/test.pyb"
    macular_dict_array_test.path_pyb = new_path_pyb
    assert macular_dict_array_test._path_pyb == new_path_pyb

    # Delete the pyb file of the MacularDictArray to be used for the test if it exists.
    try:
        os.remove(new_path_pyb)
    except FileNotFoundError:
        pass

    # Attempt to save a pyb file whose name has been changed.
    macular_dict_array_test.save()
    assert os.path.exists(new_path_pyb)
    os.remove(new_path_pyb)


def test_data_getter():
    # Use data getter and test it
    for output in macular_dict_array_head100_npones.data:
        assert np.array_equal(macular_dict_array_head100_npones.data[output], np.ones((15, 83, 99)))


def test_data_setter():
    # Attempt to modify the data attribute
    try:
        macular_dict_array_head100_npones.data = []
        assert False
    except AttributeError:
        assert True

    # Verification that the data attribute has not been modified
    for output in macular_dict_array_head100_npones.data:
        assert np.array_equal(macular_dict_array_head100_npones.data[output], np.ones((15, 83, 99)))


def test_index_getter():
    with open(f"{path_data_test}/MacularDictArray/index.pyb", "rb") as file:
        index_head100 = pickle.load(file)

    # Use index getter and test it
    assert macular_dict_array_head100_npones.index.keys() == index_head100.keys()
    assert np.array_equal(macular_dict_array_head100_npones.index["temporal"], index_head100["temporal"])
    assert np.array_equal(macular_dict_array_head100_npones.index["spatial_x"], index_head100["spatial_x"])
    assert np.array_equal(macular_dict_array_head100_npones.index["spatial_y"], index_head100["spatial_y"])


def test_index_setter():
    # Attempt to modify the index attribute
    try:
        macular_dict_array_head100_npones.data = {}
        assert False
    except AttributeError:
        assert True
    with open(f"{path_data_test}/MacularDictArray/index.pyb", "rb") as file:
        index_head100 = pickle.load(file)

    # Verification that the index attribute has not been modified
    assert macular_dict_array_head100_npones.index.keys() == index_head100.keys()
    assert np.array_equal(macular_dict_array_head100_npones.index["temporal"], index_head100["temporal"])


def test_transient_reg_getter():
    assert macular_dict_array_head100.transient_reg == re.compile(".*/[A-Za-z]{1,2}_[A-Za-z]{1,3}_[A-Za-z]{6}[0-9]{"
                                                                  "4}_.*_([0-9]{0,4}f?)")


def test_transient_reg_setter():
    macular_dict_array_test.transient_reg = "test"
    assert macular_dict_array_test.transient_reg == "test"


def test_repr():
    # Import of the control display.
    with open(f"{path_data_test}/MacularDictArray/repr.txt", "r") as file:
        display_head100_npones = "".join(file.readlines())[:-1]

    assert display_head100_npones == macular_dict_array_head100_npones.__repr__()


def test_update_from_simulation_dict():
    # Import of the initial MacularDictArray to be modified.
    with open(f"{path_pyb_file_head100_binning}", "rb") as file:
        macular_dict_array_test = pickle.load(file)
    # Use update from simulation dict and test it
    macular_dict_array_test.update_from_simulation_dict(dict_simulation_head100.copy())
    # Verification of the inequality of the macular dict array of test with the control and without any modifications.
    assert not MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100)

    # Modification of the preprocessing dictionary so that it is identical and does not cause the test to fail.
    del macular_dict_array_test.dict_preprocessing["binning"]

    # Verification of the equality of the macular dict array of test with the control.
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100)


def test_update_from_preprocessing_dict():
    # Import of the initial MacularDictArray to be modified.
    with open(path_pyb_file_head100, "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Creation of a preprocessing dictionary with binning
    dict_preprocessing_binning = dict_preprocessing_default.copy()
    dict_preprocessing_binning["binning"] = 0.0016

    # Use update from simulation dict and test it
    macular_dict_array_test.update_from_preprocessing_dict(dict_preprocessing_binning)

    # Verification of the equality of the macular dict array of test with the control.
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_binning)


def test_update_from_file():
    macular_dict_array_test.update_from_file(path_pyb_file_head100)

    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100)


def test_load():
    assert MacularDictArray.equal(MacularDictArray.load(path_pyb_file_head100), macular_dict_array_head100)


def test_save():
    MacularDictArray.load(path_pyb_file_head100).save()

    assert MacularDictArray.equal(MacularDictArray.load(path_pyb_file_head100), macular_dict_array_head100)


def test_setup_data_index_dict_array():
    # Import of the initial MacularDictArray with empty data and index to be filled.
    with open(f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0026_barSpeed6dps_head3000_no_data_no_index_0f.pyb",
              "rb") as file:
        macular_dict_array_test = pickle.load(file)

    print(macular_dict_array_test)

    # Use setup data index dict array to test it.
    macular_dict_array_test.setup_data_index_dict_array()

    # Checking equality between indexes and data.
    assert MacularDictArray.equal_dict_array(macular_dict_array_test.data,
                                             macular_dict_array_head3000.data)
    assert MacularDictArray.equal_dict_array(macular_dict_array_test.index,
                                             macular_dict_array_head3000.index)


def test_extract_data_index_from_macular_csv():
    # Import of the initial MacularDictArray with empty data and index to be filled.
    with open(f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0026_barSpeed6dps_head3000_no_data_no_index_0f.pyb",
              "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Import of the control MacularDictArray for comparison.
    with open(f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0026_barSpeed6dps_head3000_extractedDataIndex_0f.pyb",
              "rb") as file:
        macular_dict_array_head3000_extracted = pickle.load(file)

    # Use extract data index from macular csv to test it.
    macular_dict_array_test.extract_data_index_from_macular_csv()

    # Checking equality between data.
    assert macular_dict_array_test.data.keys() == macular_dict_array_head3000_extracted.data.keys()
    for output in macular_dict_array_test.data:
        for i in range(len(macular_dict_array_test.data[output])):
            assert np.array_equal(macular_dict_array_test.data[output][i],
                                  macular_dict_array_head3000_extracted.data[output][i])

    # Checking equality between indexes.
    assert macular_dict_array_test.index.keys() == macular_dict_array_head3000_extracted.index.keys()
    for output in macular_dict_array_test.index:
        for i in range(len(macular_dict_array_test.index[output])):
            assert np.array_equal(macular_dict_array_test.index[output][i],
                                  macular_dict_array_head3000_extracted.index[output][i])


def test_dataframe_chunk_processing():
    # Import of the initial MacularDictArray with empty data and index to be filled.
    with open(f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0026_barSpeed6dps_head3000_no_data_no_index_0f.pyb",
              "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Import of the first dataframe chunk to be processed.
    with open(f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0026_barSpeed6dps_first_chunk_0f.pyb", "rb") as file:
        dataframe_chunk = pickle.load(file)

    # Use dataframe chunk processing to test it.
    macular_dict_array_test.dataframe_chunk_processing(dataframe_chunk, 0)

    # Checking equality between data.
    for output in macular_dict_array_test.data:
        assert np.array_equal(np.rot90(macular_dict_array_test.data[output][0]),
                              macular_dict_array_head3000.data[output][:, :, :2000])

    # Checking equality between indexes.
    assert np.array_equal(macular_dict_array_test.index["temporal"][0],
                          macular_dict_array_head3000.index["temporal"][:2000])


def test_transient_computing():
    # Case using frames in a name following the nomenclature.
    macular_dict_array_test.path_csv = ("/".join(macular_dict_array_test.path_csv.split("/")[:-1]) +
                                        "/RC_RM_dSGpCP0026_barSpeed6dps_head100_10f")
    assert macular_dict_array_test.transient_computing() == 10 * dict_simulation_head100["delta_t"]

    # Case using the simulation dictionary with the transient number in frames.
    macular_dict_array_test._dict_simulation["transient"] = "30f"
    assert macular_dict_array_test.transient_computing() == 30 * dict_simulation_head100["delta_t"]

    # Case using the simulation dictionary with the transient number in seconds.
    macular_dict_array_test._dict_simulation["transient"] = "0.4s"
    assert macular_dict_array_test.transient_computing() == 0.4

    # Default case.
    macular_dict_array_test.path_csv = ("/".join(macular_dict_array_test.path_csv.split("/")[:-1]) +
                                        "/RC_RM_dSGpCP0026_barSpeed6dps_head100")
    del macular_dict_array_test._dict_simulation["transient"]
    assert macular_dict_array_test.transient_computing() == 0


def test_transient_extraction():
    macular_dict_array_test._path_csv = macular_dict_array_head100._path_csv.replace("_0f", "_12f")
    print(macular_dict_array_test._path_csv)
    assert macular_dict_array_test.transient_extraction() == 12


def test_concatenate_data_index_dict_array():
    # Import of the initial MacularDictArray with empty data and index to be filled.
    with open(f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0026_barSpeed6dps_head3000_extractedDataIndex_0f.pyb",
              "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Import of the control MacularDictArray for comparison.
    with open(f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0026_barSpeed6dps_head3000_concatenated_0f.pyb",
              "rb") as file:
        macular_dict_array_head3000_concatenated = pickle.load(file)

    # Use concatenate data index dict array to test it.
    macular_dict_array_test.concatenate_data_index_dict_array()

    # Checking equality between indexes and data.
    assert MacularDictArray.equal_dict_array(macular_dict_array_test.data,
                                             macular_dict_array_head3000_concatenated.data)
    assert MacularDictArray.equal_dict_array(macular_dict_array_test.index,
                                             macular_dict_array_head3000_concatenated.index)


def test_setup_spatial_index():
    # Import of the index to be compared.
    with open(f"{path_data_test}/MacularDictArray/index.pyb", "rb") as file:
        index_head100 = pickle.load(file)

    # Case of the spatial index of the x-axis.
    macular_dict_array_test.dict_simulation["n_cells_x"] = 83
    macular_dict_array_test._index["spatial_x"] = []
    macular_dict_array_test.setup_spatial_index("x")
    assert np.array_equal(macular_dict_array_test.index["spatial_x"], index_head100["spatial_x"])

    # Case of the spatial index of the y-axis.
    macular_dict_array_test.dict_simulation["n_cells_y"] = 15
    macular_dict_array_test._index["spatial_y"] = []
    macular_dict_array_test.setup_spatial_index("y")
    assert np.array_equal(macular_dict_array_test.index["spatial_y"], index_head100["spatial_y"])


def test_setup_data_dict_array_preprocessing():
    # Import of the initial MacularDictArray without any preprocessing.
    with open(path_pyb_file_head3000, "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Modification of the preprocessing dictionary.
    macular_dict_array_test.dict_preprocessing["temporal_centering"] = True
    macular_dict_array_test.dict_preprocessing["spatial_x_centering"] = True
    macular_dict_array_test.dict_preprocessing["spatial_y_centering"] = True
    macular_dict_array_test.dict_preprocessing["binning"] = 0.0016
    macular_dict_array_test.dict_preprocessing["VSDI"] = True
    macular_dict_array_test.dict_preprocessing["derivative"] = {"VSDI": 31, "FiringRate_GanglionGainControl": 31}
    macular_dict_array_test.dict_preprocessing["edge"] = (5, 0)
    macular_dict_array_test.dict_preprocessing["temporal_index_ms"] = 1000
    macular_dict_array_test.dict_preprocessing["spatial_index_mm_retina"] = 0.3
    macular_dict_array_test.dict_preprocessing["spatial_index_mm_cortex"] = 3

    # Use setup data dict array preprocessing to test it.
    macular_dict_array_test.setup_data_dict_array_preprocessing()

    # Checking equality between indexes and data.
    assert MacularDictArray.equal_dict_array(macular_dict_array_default_head3000.data, macular_dict_array_test.data)
    assert MacularDictArray.equal_dict_array(macular_dict_array_default_head3000.index, macular_dict_array_test.index)

    # Modification of the preprocessing dictionary to make it empty.
    macular_dict_array_test.dict_preprocessing = {}

    # Use setup data dict array preprocessing to test it with empty preprocessing dictionary.
    macular_dict_array_test.setup_data_dict_array_preprocessing()

    # Checking equality between indexes and data with empty preprocessing dictionary..
    assert MacularDictArray.equal_dict_array(macular_dict_array_head3000.data, macular_dict_array_test.data)
    assert MacularDictArray.equal_dict_array(macular_dict_array_head3000.index, macular_dict_array_test.index)


def test_binning_preprocess():
    # Initialisation of the MacularDictArray for tests with the values of the reduced control MacularDictArray.
    with open(path_pyb_file_head100, "rb") as file_test:
        macular_dict_array_test = pickle.load(file_test)

    # Addition and calculation of binning in the macular dict array test.
    macular_dict_array_test.dict_preprocessing["binning"] = 0.0016
    macular_dict_array_test.binning_preprocess()

    # Verification of the correct working of the binning calculation function.
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_binning)


def test_edge_cropping_preprocess():
    # Loading of a MacularDictArray with a edge cropping on FiringRate_GanglionGainControl with n=3.
    with (open(f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0026_barSpeed6dps_head100_cropEdge_0f.pyb", "rb") as
          file_cropEdge):
        macular_dict_array_head100_cropEdge = pickle.load(file_cropEdge)

    # Initialisation of the MacularDictArray for tests with the default MacularDictArray.
    with open(path_pyb_file_head100, "rb") as file_test:
        macular_dict_array_test = pickle.load(file_test)

    # Addition and calculation of edge cropping in the macular dict array test.
    macular_dict_array_test.dict_preprocessing["edge"] = (5, 0)
    macular_dict_array_test._path_pyb = macular_dict_array_head100_cropEdge.path_pyb
    macular_dict_array_test.edge_cropping_preprocess()

    # Case of edge cropping in the macular dict array test, different for x and y-axis.
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_cropEdge)

    print(macular_dict_array_test.data["FiringRate_GanglionGainControl"].shape)
    print(macular_dict_array_test.index["spatial_x"].shape)
    print(macular_dict_array_test.index["spatial_y"].shape)

    # Case of symmetrical edge cropping in the macular dict array test.
    macular_dict_array_test.dict_preprocessing["edge"] = 2
    macular_dict_array_test.edge_cropping_preprocess()
    assert macular_dict_array_test.data["FiringRate_GanglionGainControl"].shape == (11, 69, 99)
    assert macular_dict_array_test.index["spatial_x"].shape[0] == 69
    assert macular_dict_array_test.index["spatial_y"].shape[0] == 11

    # Case of asymmetrical edge cropping in the macular dict array test, different for x and y-axis.
    macular_dict_array_test.dict_preprocessing["edge"] = ((1, 3), 1)
    macular_dict_array_test.edge_cropping_preprocess()
    assert macular_dict_array_test.data["FiringRate_GanglionGainControl"].shape == (9, 65, 99)
    assert macular_dict_array_test.index["spatial_x"].shape[0] == 65
    assert macular_dict_array_test.index["spatial_y"].shape[0] == 9


def test_derivating_preprocess():
    # Loading of a MacularDictArray with a derivative on FiringRate_GanglionGainControl with n=3.
    with open(f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_dFRGang3_0f.pyb", "rb") as file_dFRGang3:
        macular_dict_array_head100_dFRGang3 = pickle.load(file_dFRGang3)

    # Initialisation of the MacularDictArray for tests with the default MacularDictArray.
    with open(path_pyb_file_head100, "rb") as file_test:
        macular_dict_array_test = pickle.load(file_test)

    # Addition and calculation of derivatives in the macular dict array test.
    macular_dict_array_test.dict_preprocessing["derivative"] = {"FiringRate_GanglionGainControl": 31}
    macular_dict_array_test._path_pyb = macular_dict_array_head100_dFRGang3.path_pyb
    macular_dict_array_test.derivating_preprocess()

    # Verification of the correct working of the derivative calculation function.
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_dFRGang3)


def test_temporal_centering_preprocess():
    # Initialisation of the MacularDictArray for tests with the default MacularDictArray.
    with open(path_pyb_file_head100, "rb") as file_test:
        macular_dict_array_test = pickle.load(file_test)

    # Addition and calculation of binning in the macular dict array test.
    macular_dict_array_test.dict_preprocessing["temporal_centering"] = True
    macular_dict_array_test._path_pyb = macular_dict_array_head100_centered.path_pyb
    macular_dict_array_test.temporal_centering_preprocess()

    # Verification of the correct working of the centering calculation function.
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_centered)

    # Checking centering taking into account horizontal edge cropping.
    macular_dict_array_test.dict_preprocessing["edge"] = ((5, 5), 2)
    macular_dict_array_test.temporal_centering_preprocess()
    assert np.array_equal(macular_dict_array_test.index["temporal_centered"],
                          macular_dict_array_head100_centered.index["temporal_centered"][5: 78])

    # Checking centering taking into account vertical edge cropping.
    macular_dict_array_test.dict_preprocessing["edge"] = ((5, 5), 2)
    macular_dict_array_test.dict_simulation["axis"] = "vertical"
    macular_dict_array_test.temporal_centering_preprocess()
    assert np.array_equal(macular_dict_array_test.index["temporal_centered"][:, 0],
                          [-0.13083, -0.16833, -0.20583, -0.24333, -0.28083,
                           -0.31833, -0.35583, -0.39333, -0.43083, -0.46833, -0.50583])


def test_mean_sectioning_preprocess():
    # Import of a SMS MacularDictArray with 1440Hz frame rate and 200°/s bar speed with sections already averaged.
    path_pyb_file_SMS_default = f"{path_data_test}/RC_RM_dSGpCP0134_barSpeed200dps_mean_sectioned_0f.pyb"
    with open(f"{path_pyb_file_SMS_default}", "rb") as file_SMS:
        macular_dict_array_SMS_mean_sectioned = pickle.load(file_SMS)

    # Import of a SMS MacularDictArray with 1440Hz frame rate and 200°/s bar speed for test.
    path_pyb_file_SMS_default = f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0134_barSpeed200dps_0f.pyb"
    with open(f"{path_pyb_file_SMS_default}", "rb") as file_SMS:
        macular_dict_array_SMS = pickle.load(file_SMS)

    # Case of calculating all average sections of axes without cropping.
    macular_dict_array_SMS.dict_preprocessing["mean_sections"] = {
        "horizontal": [
            {"measurement": "VSDI", "cropping_type": "", "cropping_dict": {}}],
        "vertical": [
            {"measurement": "VSDI", "cropping_type": "", "cropping_dict": {}}],
        "temporal": [
            {"measurement": "VSDI", "cropping_type": "", "cropping_dict": {}}]
    }
    macular_dict_array_SMS.mean_sectioning_preprocess()

    assert np.array_equal(macular_dict_array_SMS.data["horizontal_mean_section"],
                          macular_dict_array_SMS_mean_sectioned.data["horizontal_mean_section"])
    assert np.array_equal(macular_dict_array_SMS.data["vertical_mean_section"],
                          macular_dict_array_SMS_mean_sectioned.data["vertical_mean_section"])
    assert np.array_equal(macular_dict_array_SMS.data["temporal_mean_section"],
                          macular_dict_array_SMS_mean_sectioned.data["temporal_mean_section"])

    # Case of calculating average sections of spatial axes with fixed cropping.
    macular_dict_array_SMS.dict_preprocessing["mean_sections"] = {
        "horizontal": [
            {"measurement": "VSDI", "cropping_type": "fixed_edge",
             "cropping_dict": {"edge_start": 13, "edge_end": 13}}],
        "vertical": [
            {"measurement": "VSDI", "cropping_type": "fixed_edge", "cropping_dict": {"edge_start": 2, "edge_end": 2}}
        ]
    }
    macular_dict_array_SMS.mean_sectioning_preprocess()

    assert np.array_equal(macular_dict_array_SMS.data["horizontal_mean_section_fixed_edge"],
                          macular_dict_array_SMS_mean_sectioned.data["horizontal_mean_section_fixed_edge"])
    assert np.array_equal(macular_dict_array_SMS.data["vertical_mean_section_fixed_edge"],
                          macular_dict_array_SMS_mean_sectioned.data["vertical_mean_section_fixed_edge"])

    # Case of calculating average sections of spatial axes with threshold cropping.
    macular_dict_array_SMS.dict_preprocessing["mean_sections"] = {
        "horizontal": [
            {"measurement": "VSDI", "cropping_type": "threshold", "cropping_dict": {"threshold": 0.005}}],
        "vertical": [
            {"measurement": "VSDI", "cropping_type": "threshold", "cropping_dict": {"threshold": 0.005}}
        ]
    }
    macular_dict_array_SMS.mean_sectioning_preprocess()

    assert np.array_equal(macular_dict_array_SMS.data['horizontal_mean_section_threshold'][
                              ~np.isnan(macular_dict_array_SMS.data['horizontal_mean_section_threshold'])],
                          macular_dict_array_SMS_mean_sectioned.data["horizontal_mean_section_threshold"][
                              ~np.isnan(
                                  macular_dict_array_SMS_mean_sectioned.data["horizontal_mean_section_threshold"])])
    assert np.array_equal(macular_dict_array_SMS.data['vertical_mean_section_threshold'][
                              ~np.isnan(macular_dict_array_SMS.data['vertical_mean_section_threshold'])],
                          macular_dict_array_SMS_mean_sectioned.data["vertical_mean_section_threshold"][
                              ~np.isnan(macular_dict_array_SMS_mean_sectioned.data["vertical_mean_section_threshold"])])

    # Case of calculating average sections of spatial axes with threshold ratio cropping.
    macular_dict_array_SMS.dict_preprocessing["mean_sections"] = {
        "horizontal": [
            {"measurement": "VSDI", "cropping_type": "max_ratio_threshold",
             "cropping_dict": {"ratio_threshold": 0.5}}],
        "vertical": [
            {"measurement": "VSDI", "cropping_type": "max_ratio_threshold",
             "cropping_dict": {"ratio_threshold": 0.5}}
        ]
    }
    macular_dict_array_SMS.mean_sectioning_preprocess()

    assert np.array_equal(macular_dict_array_SMS.data["horizontal_mean_section_max_ratio_threshold"],
                          macular_dict_array_SMS_mean_sectioned.data["horizontal_mean_section_max_ratio_threshold"])
    assert np.array_equal(macular_dict_array_SMS.data["vertical_mean_section_max_ratio_threshold"],
                          macular_dict_array_SMS_mean_sectioned.data["vertical_mean_section_max_ratio_threshold"])


def test_make_all_indexes_units_conversion_preprocess():
    # Loading of a MacularDictArray with units conversion (mm retina, mm cortex and ms).
    with (open(f"{path_data_test}/MacularDictArray/RC_RM_dSGpCP0026_barSpeed6dps_head100_unitsConversion_0f.pyb", "rb")
          as file_unitsConversion):
        macular_dict_array_head100_unitsConversion = pickle.load(file_unitsConversion)

    # Initialisation of the MacularDictArray for tests with the default MacularDictArray.
    with open(path_pyb_file_head100, "rb") as file_test:
        macular_dict_array_test = pickle.load(file_test)

    # Addition of units conversion and centering in the macular dict array test.
    macular_dict_array_test._dict_preprocessing = {
        "temporal_centering": True,
        "spatial_x_centering": True,
        "spatial_y_centering": True,
        "temporal_index_ms": 1000,
        "spatial_index_mm_retina": 0.3,
        "spatial_index_mm_cortex": 3
    }
    macular_dict_array_test._path_pyb = macular_dict_array_head100_unitsConversion.path_pyb
    macular_dict_array_test.temporal_centering_preprocess()
    macular_dict_array_test.index["spatial_x_centered"] = DataPreprocessor.spatial_centering(
        macular_dict_array_test.index["spatial_x"], macular_dict_array_test.dict_simulation["n_cells_x"])
    macular_dict_array_test.index["spatial_y_centered"] = DataPreprocessor.spatial_centering(
        macular_dict_array_test.index["spatial_y"], macular_dict_array_test.dict_simulation["n_cells_y"])

    # Calculation of units conversion in the macular dict array test.
    macular_dict_array_test.make_all_indexes_units_conversion_preprocess()

    # Verification of the correct working of units conversion calculation function.
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_unitsConversion)


def test_copy():
    mda_copy = macular_dict_array_head100.copy()

    # Verification that the copy is a MacularDictArray.
    assert isinstance(mda_copy, MacularDictArray)

    # Verification that the two copies are identical.
    assert MacularDictArray.equal(mda_copy, macular_dict_array_head100)

    # Verification that the copy is independent.
    path_csv_copy = mda_copy._path_csv
    mda_copy._path_csv = ""
    assert not MacularDictArray.equal(mda_copy, macular_dict_array_head100)
    mda_copy._path_csv = path_csv_copy

    # Verification that the objects included in MacularDictArray are different.
    mda_copy.dict_simulation["speed"] = 30
    assert not MacularDictArray.equal(mda_copy, macular_dict_array_head100)

    # Verification of the creation of a copy with a different pyb file path.
    new_path_pyb = "../data_test/data_manager/RC_RM_dSGpCP0026_barSpeed6dps_head100_new_0f.pyb"
    mda_copy_new_path_pyb = macular_dict_array_head100.copy(new_path_pyb)
    assert MacularDictArray.equal(mda_copy_new_path_pyb, macular_dict_array_head100)
    assert mda_copy_new_path_pyb.path_pyb != macular_dict_array_head100.path_pyb
    assert mda_copy_new_path_pyb.path_pyb == (os.path.normpath(f"{os.getcwd()}/{new_path_pyb}"))

    # Test of saving a copy of a MacularDictArray with a different path pyb.
    try:
        os.remove(new_path_pyb)
    except FileNotFoundError:
        pass
    mda_copy_new_path_pyb.save()
    assert os.path.exists(new_path_pyb)
    os.remove(new_path_pyb)


def test_make_multiple_macular_dict_array():
    # Import pyb file for comparison.
    with open(f"{path_data_test}/RC_RM_dSGpCP0028_barSpeed15dps_head100_copy_0f.pyb", "rb") as file:
        macular_dict_array_head100_15dps = pickle.load(file)

    # Delete the pyb files of the MacularDictArray to be used for the test if they exist.
    try:
        os.remove(f"{path_data_test}/{name_file_head100}.pyb")
    except FileNotFoundError:
        pass
    path_pyb_file_head100_15dps = f"../data_test/data_manager/RC_RM_dSGpCP0028_barSpeed15dps_head100_0f"
    try:
        os.remove(f"{path_pyb_file_head100_15dps}.pyb")
    except FileNotFoundError:
        pass

    # Case of importing multiple MacularDictArrays with only individual simulation and preprocessing parameters.
    multiple_dicts_simulations = {"barSpeed6dps": {"path_csv": dict_simulation_head100["path_csv"],
                                                   "path_pyb": dict_simulation_head100["path_pyb"], "n_cells_x": 83,
                                                   "n_cells_y": 15, "dx": 0.225, "delta_t": 0.0167, "end": "max",
                                                   "speed": 6, "size_bar": 0.67, "axis": "horizontal"},
                                  "barSpeed15dps": {"path_csv": f"{path_pyb_file_head100_15dps}.csv",
                                                    "path_pyb": f"{path_pyb_file_head100_15dps}.pyb", "n_cells_x": 83,
                                                    "n_cells_y": 15, "dx": 0.225, "delta_t": 0.0167, "end": "max",
                                                    "speed": 15, "size_bar": 0.67, "axis": "horizontal"}
                                  }
    multiple_dicts_preprocessings = {"barSpeed6dps": {}, "barSpeed15dps": {}}

    MacularDictArray.make_multiple_macular_dict_array(multiple_dicts_simulations, multiple_dicts_preprocessings)
    assert os.path.exists(f"{path_data_test}/{name_file_head100}.pyb")
    assert os.path.exists(f"{path_pyb_file_head100_15dps}.pyb")

    with open(f"{path_data_test}/{name_file_head100}.pyb", "rb") as file:
        macular_dict_array_test = pickle.load(file)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100)

    with open(f"{path_pyb_file_head100_15dps}.pyb", "rb") as file:
        macular_dict_array_test = pickle.load(file)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_15dps)
    os.remove(f"{path_data_test}/{name_file_head100}.pyb")
    os.remove(f"{path_pyb_file_head100_15dps}.pyb")

    # Case of importing multiple MacularDictArrays with only individual and global simulation parameters.
    multiple_dicts_simulations_global = {"global": {"n_cells_x": 83, "n_cells_y": 15, "dx": 0.225, "delta_t": 0.0167,
                                                    "end": "max", "size_bar": 0.67, "axis": "horizontal"},
                                         "barSpeed6dps": {"path_csv": dict_simulation_head100["path_csv"],
                                                          "path_pyb": dict_simulation_head100["path_pyb"],
                                                          "speed": 6},
                                         "barSpeed15dps": {"path_csv": f"{path_pyb_file_head100_15dps}.csv",
                                                           "path_pyb": f"{path_pyb_file_head100_15dps}.pyb",
                                                           "speed": 15}
                                         }
    MacularDictArray.make_multiple_macular_dict_array(multiple_dicts_simulations_global, multiple_dicts_preprocessings)
    assert os.path.exists(f"{path_data_test}/{name_file_head100}.pyb")
    assert os.path.exists(f"{path_pyb_file_head100_15dps}.pyb")

    with open(f"{path_data_test}/{name_file_head100}.pyb", "rb") as file:
        macular_dict_array_test = pickle.load(file)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100)

    with open(f"{path_pyb_file_head100_15dps}.pyb", "rb") as file:
        macular_dict_array_test = pickle.load(file)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_15dps)
    os.remove(f"{path_data_test}/{name_file_head100}.pyb")
    os.remove(f"{path_pyb_file_head100_15dps}.pyb")

    # Case of importing multiple MacularDictArrays with a global preprocessing parameter.
    multiple_dicts_preprocessings_global = {"global": {"VSDI": True}, "barSpeed6dps": {}, "barSpeed15dps": {}}
    MacularDictArray.make_multiple_macular_dict_array(multiple_dicts_simulations, multiple_dicts_preprocessings_global)
    assert os.path.exists(f"{path_data_test}/{name_file_head100}.pyb")
    assert os.path.exists(f"{path_pyb_file_head100_15dps}.pyb")

    with open(f"{path_data_test}/{name_file_head100}.pyb", "rb") as file:
        macular_dict_array_test = pickle.load(file)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_VSDI)

    with open(f"{path_pyb_file_head100_15dps}.pyb", "rb") as file:
        macular_dict_array_test = pickle.load(file)
    assert "VSDI" in macular_dict_array_test.data
    os.remove(f"{path_data_test}/{name_file_head100}.pyb")
    os.remove(f"{path_pyb_file_head100_15dps}.pyb")

    # Case of importing multiple MacularDictArrays with different individual preprocessing parameters.
    multiple_dicts_preprocessings = {"barSpeed6dps": {"VSDI": True}, "barSpeed15dps": {}}
    MacularDictArray.make_multiple_macular_dict_array(multiple_dicts_simulations, multiple_dicts_preprocessings)
    assert os.path.exists(f"{path_data_test}/{name_file_head100}.pyb")
    assert os.path.exists(f"{path_pyb_file_head100_15dps}.pyb")

    with open(f"{path_data_test}/{name_file_head100}.pyb", "rb") as file:
        macular_dict_array_test = pickle.load(file)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_VSDI)

    with open(f"{path_pyb_file_head100_15dps}.pyb", "rb") as file:
        macular_dict_array_test = pickle.load(file)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_15dps)
    os.remove(f"{path_data_test}/{name_file_head100}.pyb")
    os.remove(f"{path_pyb_file_head100_15dps}.pyb")

    # Case of importing multiple MacularDictArrays with only one global preprocessing parameter.
    multiple_dicts_preprocessings_global = {"global": {"VSDI": True}}
    MacularDictArray.make_multiple_macular_dict_array(multiple_dicts_simulations, multiple_dicts_preprocessings_global)
    assert os.path.exists(f"{path_data_test}/{name_file_head100}.pyb")
    assert os.path.exists(f"{path_pyb_file_head100_15dps}.pyb")

    with open(f"{path_data_test}/{name_file_head100}.pyb", "rb") as file:
        macular_dict_array_test = pickle.load(file)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_VSDI)

    with open(f"{path_pyb_file_head100_15dps}.pyb", "rb") as file:
        macular_dict_array_test = pickle.load(file)
    assert "VSDI" in macular_dict_array_test.data
    os.remove(f"{path_data_test}/{name_file_head100}.pyb")
    os.remove(f"{path_pyb_file_head100_15dps}.pyb")
