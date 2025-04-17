import os
import pickle
import numpy as np

from src.data_manager.MacularDictArray import MacularDictArray

# Get data for test from relative path.
path_data_test = os.path.normpath(f"{os.getcwd()}/../data_test/data_manager/")

# Import of a reduced MacularDictArray control with only the 100 first rows.
path_pyb_file_head100 = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_copy_0f.pyb"
with open(path_pyb_file_head100, "rb") as file_head100:
    macular_dict_array_head100 = pickle.load(file_head100)

# Initialisation of the MacularDictArray for tests with the values of the reduced control MacularDictArray.
with open(path_pyb_file_head100, "rb") as file_test:
    macular_dict_array_test = pickle.load(file_test)

# Import of a reduced MacularDictArray control with only the 3000 first rows.
path_pyb_file_head3000 = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head3000_copy_0f.pyb"
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
path_pyb_file_head100_centered = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_centered_0f.pyb"
with open(f"{path_pyb_file_head100_centered}", "rb") as file_centered:
    macular_dict_array_head100_centered = pickle.load(file_centered)

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
    dict_preprocessing_default_test = {}
    assert macular_dict_array_head100.cleaning_dict_preprocessing(dict_preprocessing_default_test) == {}

    # Case of a dictionary without any preprocess.
    dict_preprocessing_default_test = {"temporal_centering": False, "binning": False, "derivative": False,
                                       "edge": False}
    assert macular_dict_array_head100.cleaning_dict_preprocessing(dict_preprocessing_default_test) == {}

    # Case of True and False values.
    dict_preprocessing_default_test = {"temporal_centering": True, "binning": False, "derivative": False, "edge": False}
    assert macular_dict_array_head100.cleaning_dict_preprocessing(dict_preprocessing_default_test) == {
        "temporal_centering": True}

    # Case of a float.
    dict_preprocessing_default_test = {"temporal_centering": False, "binning": 0.0016, "derivative": False,
                                       "edge": False}
    assert macular_dict_array_head100.cleaning_dict_preprocessing(dict_preprocessing_default_test) == {
        "binning": 0.0016}

    # Case of a dictionary
    dict_preprocessing_default_test = {"temporal_centering": False, "binning": False,
                                       "derivative": {"VSDI": 3, "FiringRate_GanglionGainControl": 1}, "edge": False}
    assert macular_dict_array_head100.cleaning_dict_preprocessing(dict_preprocessing_default_test) == {
        "derivative": {"VSDI": 3, "FiringRate_GanglionGainControl": 1}}

    # Case of a tuple.
    dict_preprocessing_default_test = {"temporal_centering": False, "binning": False, "derivative": False,
                                       "edge": (5, 0)}
    assert macular_dict_array_head100.cleaning_dict_preprocessing(dict_preprocessing_default_test) == {"edge": (5, 0)}


def test_checking_pre_existing_file():
    # Deletion of the file to be used for the test, if it exists.
    try:
        os.remove(dict_simulation_head100['path_pyb'])
    except FileNotFoundError:
        pass

    # Test of the construction of a new MacularDictArray in the absence of a file to import.
    macular_dict_array_test.checking_pre_existing_file(dict_simulation_head100.copy(), dict_preprocessing_default)

    # Case of non-existent file.
    assert MacularDictArray.equal(macular_dict_array_head100, macular_dict_array_test)
    macular_dict_array_head100.save()

    # Update of the MacularDictArray from an existing pyb file.
    dict_simulation_head100_30dps = macular_dict_array_head100_30dps.dict_simulation.copy()
    dict_simulation_head100_30dps[
        "path_csv"] = f"../data_test/data_manager/RC_RM_dSGpCP0033_barSpeed30dps_head100_0f.csv"
    dict_simulation_head100_30dps[
        "path_pyb"] = f"../data_test/data_manager/RC_RM_dSGpCP0033_barSpeed30dps_head100_0f.pyb"
    macular_dict_array_test.checking_pre_existing_file(dict_simulation_head100_30dps, dict_preprocessing_default)

    # Case of the existing file.
    assert MacularDictArray.equal(macular_dict_array_head100_30dps, macular_dict_array_test)

    # Case of direct import of an existing file without comparison.
    macular_dict_array_test.checking_pre_existing_file({"path_pyb": path_pyb_file_head100}, {})
    assert MacularDictArray.equal(macular_dict_array_head100, macular_dict_array_test)


def test_checking_difference_file_json(monkeypatch):
    # Import of MacularDictArray to be compared.
    with open(f"{path_pyb_file_head100_30dps}", "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Cases where there is no difference.
    error = macular_dict_array_test.checking_difference_file_json(dict_simulation_head100_30dps,
                                                                  dict_preprocessing_default)
    assert not error

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

    # Case of incorrect user input.
    monkeypatch.setattr('builtins.input', lambda _: "No")
    try:
        MacularDictArray(dict_simulation_head100, dict_preprocessing_default)
    except ValueError:
        assert True


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
    # Import of MacularDictArray to be compared.
    path_pyb_file_head100_npones = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_npOnes_0f"
    with open(f"{path_pyb_file_head100_npones}.pyb", "rb") as file:
        macular_dict_array_head100_npones = pickle.load(file)

    # Use data getter and test it
    for output in macular_dict_array_head100_npones.data:
        assert np.array_equal(macular_dict_array_head100_npones.data[output], np.ones((15, 83, 99)))


def test_data_setter():
    # Import of MacularDictArray to be modified.
    path_pyb_file_head100_npones = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_npOnes_0f"
    with open(f"{path_pyb_file_head100_npones}.pyb", "rb") as file:
        macular_dict_array_head100_npones = pickle.load(file)

    # Attempt to modify the data attribute
    macular_dict_array_head100_npones.data = []

    # Verification that the data attribute has not been modified
    for output in macular_dict_array_head100_npones.data:
        assert np.array_equal(macular_dict_array_head100_npones.data[output], np.ones((15, 83, 99)))


def test_index_getter():
    # Import of MacularDictArray to be compared.
    path_pyb_file_head100_npones = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_npOnes_0f"
    with open(f"{path_pyb_file_head100_npones}.pyb", "rb") as file:
        macular_dict_array_head100_npones = pickle.load(file)

    with open(f"{path_data_test}/index.pyb", "rb") as file:
        index_head100 = pickle.load(file)

    # Use index getter and test it
    assert macular_dict_array_head100_npones.index.keys() == index_head100.keys()
    assert np.array_equal(macular_dict_array_head100_npones.index["temporal"], index_head100["temporal"])
    assert np.array_equal(macular_dict_array_head100_npones.index["spatial_x"], index_head100["spatial_x"])
    assert np.array_equal(macular_dict_array_head100_npones.index["spatial_y"], index_head100["spatial_y"])


def test_index_setter():
    # Import of MacularDictArray to be modified.
    path_pyb_file_head100_npones = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_npOnes_0f"
    with open(f"{path_pyb_file_head100_npones}.pyb", "rb") as file:
        macular_dict_array_head100_npones = pickle.load(file)

    # Attempt to modify the index attribute
    macular_dict_array_head100_npones.index = {}

    with open(f"{path_data_test}/index.pyb", "rb") as file:
        index_head100 = pickle.load(file)

    # Verification that the index attribute has not been modified
    assert macular_dict_array_head100_npones.index.keys() == index_head100.keys()
    assert np.array_equal(macular_dict_array_head100_npones.index["temporal"], index_head100["temporal"])


def test_repr():
    # Import of MacularDictArray to be displayed.
    path_pyb_file_head100_npones = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_npOnes_0f"
    with open(f"{path_pyb_file_head100_npones}.pyb", "rb") as file:
        macular_dict_array_head100_npones = pickle.load(file)

    # Import of the control display.
    with open(f"{path_data_test}/repr.txt", "r") as file:
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
    with open(f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head3000_no_data_no_index_0f.pyb", "rb") as file:
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
    with open(f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head3000_no_data_no_index_0f.pyb", "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Import of the control MacularDictArray for comparison.
    with open(f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head3000_extractedDataIndex_0f.pyb", "rb") as file:
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
    with open(f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head3000_no_data_no_index_0f.pyb", "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Import of the first dataframe chunk to be processed.
    with open(f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_first_chunk_0f.pyb", "rb") as file:
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


def test_concatenate_data_index_dict_array():
    # Import of the initial MacularDictArray with empty data and index to be filled.
    with open(f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head3000_extractedDataIndex_0f.pyb", "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Import of the control MacularDictArray for comparison.
    with open(f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head3000_concatenated_0f.pyb", "rb") as file:
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
    with open(f"{path_data_test}/index.pyb", "rb") as file:
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

    # Import of the default MacularDictArray to be compared with preprocessing.
    with open(f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_default_head3000_0f.pyb", "rb") as file:
        macular_dict_array_default_head3000 = pickle.load(file)

    # Modification of the preprocessing dictionary.
    macular_dict_array_test.dict_preprocessing["binning"] = 0.0016
    macular_dict_array_test.dict_preprocessing["temporal_centering"] = True
    macular_dict_array_test.dict_preprocessing["VSDI"] = True

    # Use setup data dict array preprocessing to test it.
    macular_dict_array_test.setup_data_dict_array_preprocessing()

    # Checking equality between indexes and data.
    assert MacularDictArray.equal_dict_array(macular_dict_array_default_head3000.data, macular_dict_array_test.data)
    assert MacularDictArray.equal_dict_array(macular_dict_array_default_head3000.index, macular_dict_array_test.index)

    # Case of the use of millisecond indexes in the preprocessing dictionary.
    macular_dict_array_test.dict_preprocessing["ms"] = True
    macular_dict_array_test.setup_data_dict_array_preprocessing()
    assert np.array_equal(macular_dict_array_test.index["temporal_ms"],
                          macular_dict_array_test.index["temporal"] * 1000)
    for index_ms, index_s in zip(macular_dict_array_test.index["temporal_centered_ms"],
                                 macular_dict_array_test.index["temporal_centered"]):
        assert np.array_equal(index_s * 1000, index_ms)

    # Case of the use of millisecond indexes in the preprocessing dictionary.
    with open(f"{path_data_test}/index_spatial_centered.pyb", "rb") as file:
        index_spatial_centered = pickle.load(file)

    # Case with spatial index centred in x and y.
    macular_dict_array_test.dict_preprocessing["spatial_x_centering"] = True
    macular_dict_array_test.dict_preprocessing["spatial_y_centering"] = True
    macular_dict_array_test.setup_data_dict_array_preprocessing()
    assert np.array_equal(macular_dict_array_test.index["spatial_x_centered"],
                          index_spatial_centered["spatial_x_centered"])
    assert np.array_equal(macular_dict_array_test.index["spatial_y_centered"],
                          index_spatial_centered["spatial_y_centered"])


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
    with open(f"{path_data_test}/RC_RM_dSGpCP0028_barSpeed15dps_head100_copy_0f", "rb") as file:
        macular_dict_array_head100_15dps = pickle.load(file)

    # Delete the pyb files of the MacularDictArray to be used for the test if they exist.
    try:
        os.remove(f"{path_data_test}/{name_file_head100}.pyb")
    except FileNotFoundError:
        pass
    path_pyb_file_head100_15dps = f"{path_data_test}/RC_RM_dSGpCP0028_barSpeed15dps_head100_0f"
    try:
        os.remove(f"{path_pyb_file_head100_15dps}.pyb")
    except FileNotFoundError:
        pass

    # Case of importing multiple MacularDictArrays with only individual simulation and preprocessing parameters.
    path_pyb_file_head100_15dps = f"{path_data_test}/RC_RM_dSGpCP0028_barSpeed15dps_head100_0f"
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