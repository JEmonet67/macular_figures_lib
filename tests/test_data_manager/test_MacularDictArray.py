import os
import pickle
import numpy as np

from src.data_manager.MacularDictArray import MacularDictArray

path_data_test = "/user/jemonet/home/Documents/These/Code/macular_figures_lib/tests/data_test"

# Import of MacularDictArray control.
path_pyb_file_default = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_default_0f.pyb"
with open(path_pyb_file_default, "rb") as file_default:
    macular_dict_array_default = pickle.load(file_default)

# Import of a reduced MacularDictArray control with only the 100 first rows.
path_pyb_file_head100 = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_0f.pyb"
with open(path_pyb_file_head100, "rb") as file_head100:
    macular_dict_array_head100 = pickle.load(file_head100)

# Initialisation of the MacularDictArray for tests with the values of the reduced control MacularDictArray.
with open(path_pyb_file_head100, "rb") as file_test:
    macular_dict_array_test = pickle.load(file_test)

# Import of a reduced MacularDictArray control with only the 3000 first rows.
path_pyb_file_head3000 = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head3000_0f.pyb"
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
name_file_default = "RC_RM_dSGpCP0026_barSpeed6dps_default_0f"
dict_simulation_default = {
    "path_data": f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_default_0f",
    "n_cells_x": 83,
    "n_cells_y": 15,
    "dx": 0.225,
    "delta_t": 0.0167,
    "end": "max",
    "speed": 6,
    "size_bar": 0.67,
    "axis": "horizontal"
}
dict_preprocessing_default = {
    "temporal_centering": False,
    "binning": False,  # 0.0016,
    "VSDI": False,
    "derivative": False  # {"FiringRate_GanglionGainControl": 3}
}
name_file_head100 = "RC_RM_dSGpCP0026_barSpeed6dps_head100_0f"
dict_simulation_head100 = dict_simulation_default.copy()
dict_simulation_head100["path_data"] = f"{path_data_test}/{name_file_head100}"

# def test_make():
#     MacularDictArray(dict_simulation_head100, dict_preprocessing_default)

def test_init():
    # Delete the pyb file of the MacularDictArray to be used for the test if it exists.
    try:
        os.remove(f"{dict_simulation_head100['path_data']}.pyb")
    except FileNotFoundError:
        pass

    # Creation of a new MacularDictArray.
    macular_dict_array_test = MacularDictArray(dict_simulation_head100, dict_preprocessing_default)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100)

    # Checking the save.
    assert os.path.exists(f"{dict_simulation_head100['path_data']}.pyb")

    # Reimport of the newly created MacularDictArray.
    macular_dict_array_test = MacularDictArray(dict_simulation_head100, dict_preprocessing_default)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100)


def test_equal():
    # Import of the second MacularDictArray for comparison.
    with open(path_pyb_file_head100, "rb") as file:
        macular_dict_array_head100_copy = pickle.load(file)

    # Case of a tie between the two MacularDictArrays.
    assert MacularDictArray.equal(macular_dict_array_head100, macular_dict_array_head100_copy)

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
    assert MacularDictArray.equal_dict_array(macular_dict_array_head100.data, macular_dict_array_head100.data)
    assert MacularDictArray.equal_dict_array(macular_dict_array_head100.index, macular_dict_array_head100.index)

    assert not MacularDictArray.equal_dict_array(macular_dict_array_head100.data,
                                                 macular_dict_array_head100_binning.data)
    assert not MacularDictArray.equal_dict_array(macular_dict_array_head100.index,
                                                 macular_dict_array_head100_binning.index)

    assert not MacularDictArray.equal_dict_array(macular_dict_array_head100.data,
                                                 macular_dict_array_head100_VSDI.data)
    assert not MacularDictArray.equal_dict_array(macular_dict_array_head100.index,
                                                 macular_dict_array_head100_centered.index)


def test_checking_pre_existing_file():
    # Deletion of the file to be used for the test, if it exists.
    try:
        os.remove(f"{dict_simulation_head100['path_data']}.pyb")
    except FileNotFoundError:
        pass

    # Test of the construction of a new MacularDictArray in the absence of a file to import.
    macular_dict_array_test.checking_pre_existing_file(dict_simulation_head100, dict_preprocessing_default)
    # Case of non-existent file.
    assert MacularDictArray.equal(macular_dict_array_head100, macular_dict_array_test)
    macular_dict_array_head100.save()

    # Update of the MacularDictArray from an existing pyb file.
    macular_dict_array_test.checking_pre_existing_file(macular_dict_array_head100_30dps.dict_simulation.copy(),
                                                       dict_preprocessing_default)
    # Case of the existing file.
    assert MacularDictArray.equal(macular_dict_array_head100_30dps, macular_dict_array_test)


def test_checking_difference_file_json(monkeypatch):
    # Import of MacularDictArray to be compared.
    with open(f"{path_pyb_file_head100_30dps}", "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Case of keeping the pyb when dict_simulation differs.
    monkeypatch.setattr('builtins.input', lambda _: "pyb")
    macular_dict_array_test.checking_difference_file_json(dict_simulation_head100, dict_preprocessing_default)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_30dps)

    # Case of keeping the pyb when dict_preprocessing differs.
    dict_preprocessing_modified = macular_dict_array_head100_30dps.dict_preprocessing.copy()
    dict_preprocessing_modified["VSDI"] = False
    macular_dict_array_test.checking_difference_file_json(dict_simulation_head100, dict_preprocessing_modified)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100_30dps)

    # Keep the json in case of difference file/json.
    monkeypatch.setattr('builtins.input', lambda _: "json")
    macular_dict_array_test.checking_difference_file_json(dict_simulation_head100, dict_preprocessing_default)
    assert MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100)

    # Case of incorrect user input.
    monkeypatch.setattr('builtins.input', lambda _: "No")
    try:
        MacularDictArray(dict_simulation_default, dict_simulation_default)
    except ValueError:
        assert True


def test_simulation_id_getter():
    assert macular_dict_array_head100.simulation_id == name_file_head100


def test_simulation_id_setter():
    macular_dict_array_test.simulation_id = "test"
    assert macular_dict_array_test.simulation_id == "test"


def test_dict_simulation_getter():
    assert macular_dict_array_head100.dict_simulation == dict_simulation_head100


def test_dict_simulation_setter():
    # Import of MacularDictArray to be modified.
    with open(f"{path_pyb_file_head100_30dps}", "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Use of dict_simulation setter.
    macular_dict_array_test.dict_simulation = dict_simulation_head100

    # Case of the correct modification of simulation dictionary.
    assert macular_dict_array_test.dict_simulation == dict_simulation_head100

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


def test_cond_getter():
    assert macular_dict_array_head100.cond == "barSpeed6dps_head100"


def test_cond_setter():
    macular_dict_array_head100.cond = "test"
    assert macular_dict_array_head100.cond == "barSpeed6dps_head100"


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
    assert np.array_equal(macular_dict_array_head100_npones.index["default"], index_head100["default"])


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
    assert np.array_equal(macular_dict_array_head100_npones.index["default"], index_head100["default"])


def test_repr():
    # Import of MacularDictArray to be displayed.
    path_pyb_file_head100_npones = f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_npOnes_0f"
    with open(f"{path_pyb_file_head100_npones}.pyb", "rb") as file:
        macular_dict_array_head100_npones = pickle.load(file)

    # Import of the control display.
    with open(f"{path_data_test}/repr.txt", "r") as file:
        display_head100_npones = "".join(file.readlines())

    assert display_head100_npones == macular_dict_array_head100_npones.__repr__()


def test_update_from_simulation_dict():
    # Import of the initial MacularDictArray to be modified.
    with open(f"{path_pyb_file_head100_binning}", "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Use update from simulation dict and test it
    macular_dict_array_test.update_from_simulation_dict(dict_simulation_head100)

    # Verification of the inequality of the macular dict array of test with the control and without any modifications.
    assert not MacularDictArray.equal(macular_dict_array_test, macular_dict_array_head100)

    # Modification of the preprocessing dictionary so that it is identical and does not cause the test to fail.
    macular_dict_array_test.dict_preprocessing["binning"] = False

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
    macular_dict_array_test.dataframe_chunk_processing(dataframe_chunk, path_pyb_file_head3000[:-3] + "csv", 0)

    # Checking equality between data.
    for output in macular_dict_array_test.data:
        assert np.array_equal(np.rot90(macular_dict_array_test.data[output][0]),
                              macular_dict_array_head3000.data[output][:, :, :2000])

    # Checking equality between indexes.
    assert np.array_equal(macular_dict_array_test.index["default"][0],
                          macular_dict_array_head3000.index["default"][:2000])


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


def test_setup_data_dict_array_preprocessing():
    # Import of the default MacularDictArray to be compared.
    with open(path_pyb_file_head3000, "rb") as file:
        macular_dict_array_test = pickle.load(file)

    # Import of the initial MacularDictArray without any preprocessing.
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
