import os
import pickle
import re

from src.data_manager.MacularAnalysisDataframes import MacularAnalysisDataframes
from src.data_manager.MacularDictArray import MacularDictArray

# Get data for test from relative path.
path_data_test = os.path.normpath(f"{os.getcwd()}/../data_test/data_manager/")

# Import of a default MacularAnalysisDataframes based on reduced MacularDictArray (100 first rows).
with open(f"{path_data_test}/initialized_macular_analysis_dataframe.pyb", "rb") as file:
    macular_analysis_dataframes_barSpeed_head100 = pickle.load(file)

# Import of a reduced MacularAnalysisDataframes for tests.
with open(f"{path_data_test}/initialized_macular_analysis_dataframe.pyb", "rb") as file:
    macular_analysis_dataframes_test = pickle.load(file)

# Import a default multiple reduced macular dict array of bar speed condition.
with open(f"{path_data_test}/multiple_macula_dict_array_barSpeed_head100.pyb", "rb") as file:
    multi_macular_dict_array_barSpeed_head100 = pickle.load(file)

# Import a multiple reduced macular dict array of bar speed condition for tests.
with open(f"{path_data_test}/multiple_macula_dict_array_barSpeed_head100.pyb", "rb") as file:
    multi_macular_dict_array_test = pickle.load(file)

multiple_dicts_simulations_barSpeed_head100 = {
    "global": {
        "n_cells_x": 83,
        "n_cells_y": 15,
        "dx": 0.225,
        "delta_t": 0.0167,
        "end": "max",
        "size_bar": 0.67,
        "axis": "horizontal"
    },
    "barSpeed6dps": {
        "path_pyb": f"../data_test/data_manager/RC_RM_dSGpCP0026_barSpeed6dps_head100_copy_0f.pyb",
        "path_csv": f"../data_test/data_manager/RC_RM_dSGpCP0026_barSpeed6dps_head100_0f.csv",
        "speed": 6
    },
    "barSpeed15dps": {
        "path_pyb": f"../data_test/data_manager/RC_RM_dSGpCP0028_barSpeed15dps_head100_copy_0f.pyb",
        "path_csv": f"../data_test/data_manager/RC_RM_dSGpCP0028_barSpeed15dps_head100_0f.csv",
        "speed": 15
    },
    "barSpeed30dps": {
        "path_pyb": f"../data_test/data_manager/RC_RM_dSGpCP0033_barSpeed30dps_head100_0f.pyb",
        "path_csv": f"../data_test/data_manager/RC_RM_dSGpCP0033_barSpeed30dps_head100_0f.csv",
        "speed": 30
    }
}

multiple_dicts_preprocessings_barSpeed_head100 = {
    "global": {},
    "barSpeed6dps": {},
    "barSpeed15dps": {},
    "barSpeed30dps": {}
}

multiple_dicts_analysis_barSpeed_head100 = {
    "Conditions": {},
    "X": {},
    "Y": {},
    "Time": {}
}



def test_init():
    print()
    print(macular_analysis_dataframes_test.multiple_dicts_analysis)
    print(macular_analysis_dataframes_test.multiple_dicts_simulations)
    print(macular_analysis_dataframes_test.multiple_dicts_preprocessings)


def test_dict_paths_pyb_getter():
    assert macular_analysis_dataframes_barSpeed_head100.dict_paths_pyb == {
        "barSpeed6dps": f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_copy_0f.pyb",
        "barSpeed15dps": f"{path_data_test}/RC_RM_dSGpCP0028_barSpeed15dps_head100_copy_0f.pyb",
        "barSpeed30dps": f"{path_data_test}/RC_RM_dSGpCP0033_barSpeed30dps_head100_0f.pyb"}


def test_dict_paths_pyb_setter():
    macular_analysis_dataframes_test.dict_paths_pyb = {"barSpeed9dps": ""}
    assert macular_analysis_dataframes_test.dict_paths_pyb == {"barSpeed9dps": ""}


def test_dict_analysis_dataframes_getter():
    # Import d'un dictionnaire d'analyses de dataframes d'exemple.
    with open(f"{path_data_test}/dict_analysis_dataframes_barSpeed_head100.pyb", "rb") as file:
        dict_analysis_dataframes_barSpeed_head100 = pickle.load(file)

    # Comparison of the example dictionary with that of the setter.
    for dataframe_name in macular_analysis_dataframes_barSpeed_head100.dict_analysis_dataframes:
        if dataframe_name == "Conditions":
            assert macular_analysis_dataframes_barSpeed_head100.dict_analysis_dataframes[dataframe_name].equals(
                dict_analysis_dataframes_barSpeed_head100[dataframe_name])
        else:
            for condition in macular_analysis_dataframes_barSpeed_head100.dict_analysis_dataframes[dataframe_name]:
                assert (macular_analysis_dataframes_barSpeed_head100.dict_analysis_dataframes[dataframe_name][condition]
                        .equals(dict_analysis_dataframes_barSpeed_head100[dataframe_name][condition]))


def test_dict_analysis_dataframes_setter():
    old_dict_analysis_dataframes = macular_analysis_dataframes_test.dict_analysis_dataframes.copy()
    macular_analysis_dataframes_test.dict_analysis_dataframes = {}
    assert macular_analysis_dataframes_test.dict_analysis_dataframes != old_dict_analysis_dataframes
    assert macular_analysis_dataframes_test.dict_analysis_dataframes == {}


def test_multiple_dicts_analysis_getter():
    assert (macular_analysis_dataframes_barSpeed_head100.multiple_dicts_analysis
            == {'Conditions': {}, 'Time': {}, 'X': {}, 'Y': {}})


def test_multiple_dicts_analysis_setter():
    macular_analysis_dataframes_test.multiple_dicts_analysis = {"test": {"setter1": True, "setter2": False},
                                                                "test2": {"setter3": False}, "test3": {}}
    assert macular_analysis_dataframes_test.multiple_dicts_analysis == {"test": {"setter1": True}}


def test_multiple_dicts_preprocessings_getter():
    assert (macular_analysis_dataframes_barSpeed_head100.multiple_dicts_preprocessings
            == {})


def test_multiple_dicts_preprocessings_setter():
    # Case of an attempt to modify multiple_dicts_preprocessings.
    try:
        macular_analysis_dataframes_test.multiple_dicts_preprocessings = {"test": "setter"}
        assert False
    except AttributeError:
        assert True

    # Verification that the value of multiple_dicts_preprocessings has not changed.
    assert (macular_analysis_dataframes_test.multiple_dicts_preprocessings
            == {})


def test_multiple_dicts_simulations_getter():
    assert (macular_analysis_dataframes_barSpeed_head100.multiple_dicts_simulations
            == multiple_dicts_simulations_barSpeed_head100)


def test_multiple_dicts_simulations_setter():
    # Case of an attempt to modify multiple_dicts_simulations.
    try:
        macular_analysis_dataframes_test.multiple_dicts_simulations = {"test": "setter"}
        assert False
    except AttributeError:
        assert True

    # Verification that the value of multiple_dicts_simulations has not changed.
    assert (macular_analysis_dataframes_test.multiple_dicts_simulations
            == multiple_dicts_simulations_barSpeed_head100)


def test_condition_reg_getter():
    assert macular_analysis_dataframes_barSpeed_head100.condition_reg == re.compile(
        "(^[A-Za-z]+)(-?[0-9]{1,4},?[0-9]{0,4})([A-Za-z]+$)")


def test_condition_reg_setter():
    # Test changing regular expression name value unit.
    macular_analysis_dataframes_test.condition_reg = "(^[A-Za-z]+)"
    assert macular_analysis_dataframes_test.condition_reg == re.compile("(^[A-Za-z]+)")

    # Reset regular expression name value unit.
    macular_analysis_dataframes_test.condition_reg = "(^[A-Za-z]+)(-?[0-9]{1,4},?[0-9]{0,4})([A-Za-z]+$)"


def test_get_maximal_index_multi_macular_dict_array():
    # Creation of a multi_macular_dict_array with spatio-temporal indexes varying between each MacularDictArray.
    multi_macular_dict_array_index_modified = MacularDictArray.make_multiple_macular_dict_array(
        multiple_dicts_simulations_barSpeed_head100, multiple_dicts_preprocessings_barSpeed_head100)
    i = 1
    for name_dict_array in multi_macular_dict_array_index_modified:
        dict_array = multi_macular_dict_array_index_modified[name_dict_array]
        dict_array.index["temporal"] = dict_array.index["temporal"][:99 - 10 * i]
        dict_array.index["spatial_x"] = dict_array.index["spatial_x"][:73 - 4 * i]
        dict_array.index["spatial_y"] = dict_array.index["spatial_y"][:15 - 2 * i]
        i += 1

    # Case of the time index.
    assert macular_analysis_dataframes_test.get_maximal_index_multi_macular_dict_array(
        multi_macular_dict_array_index_modified, "temporal").shape[0] == 89

    # Case of spatial index x.
    assert macular_analysis_dataframes_test.get_maximal_index_multi_macular_dict_array(
        multi_macular_dict_array_index_modified, "spatial_x").shape[0] == 69

    # Case of spatial index y.
    assert macular_analysis_dataframes_test.get_maximal_index_multi_macular_dict_array(
        multi_macular_dict_array_index_modified, "spatial_y").shape[0] == 13


def test_initialize_dict_analysis_dataframes():
    # Opening a macular analysis dataframe that has already been initialised as an example.
    with open(f"{path_data_test}/initialized_macular_analysis_dataframe.pyb", "rb") as file:
        macular_analysis_dataframes_init = pickle.load(file)

    # Opening a macular analysis dataframe that has already been created but has yet to be initialised.
    with open(f"{path_data_test}/macular_analysis_dataframes_to_init.pyb", "rb") as file:
        macular_analysis_dataframes_to_init = pickle.load(file)

    # Index listing.
    t_index = multi_macular_dict_array_barSpeed_head100["barSpeed6dps"].index["temporal"]
    x_index = multi_macular_dict_array_barSpeed_head100["barSpeed6dps"].index["spatial_x"]
    y_index = multi_macular_dict_array_barSpeed_head100["barSpeed6dps"].index["spatial_y"]

    # Changed the multiple analysis dictionary to prevent it from being empty.
    macular_analysis_dataframes_to_init._multiple_dicts_analysis = {"Conditions": {}, "X": {}, "Y": {}, "Time": {}}

    # Initialisation of Macular analysis dataframes.
    macular_analysis_dataframes_to_init.initialize_dict_analysis_dataframes(x_index, y_index, t_index)

    # Verify that the conditions dataframe is correct.
    assert macular_analysis_dataframes_init.dict_analysis_dataframes["Conditions"].equals(
        macular_analysis_dataframes_to_init.dict_analysis_dataframes["Conditions"])

    # Verify that the X, Y, and T dataframes for each condition are correct.
    for condition in macular_analysis_dataframes_init.dict_paths_pyb:
        assert macular_analysis_dataframes_init.dict_analysis_dataframes["X"][condition].equals(
            macular_analysis_dataframes_to_init.dict_analysis_dataframes["X"][condition])
        assert macular_analysis_dataframes_init.dict_analysis_dataframes["Y"][condition].equals(
            macular_analysis_dataframes_to_init.dict_analysis_dataframes["Y"][condition])
        assert macular_analysis_dataframes_init.dict_analysis_dataframes["Time"][condition].equals(
            macular_analysis_dataframes_to_init.dict_analysis_dataframes["Time"][condition])


def test_initialize_analysis_dataframe():
    # Open a example dataframe from the spatial dataframe x.
    with open(f"{path_data_test}/empty_spatial_x_dataframe.pyb", "rb") as file:
        spatial_x_dataframe = pickle.load(file)

    # Create a spatial dataframe x using the initialize_analysis_dataframe function.
    dataframe = macular_analysis_dataframes_test.initialize_analysis_dataframe(
        multi_macular_dict_array_barSpeed_head100["barSpeed6dps"].index["spatial_x"], "X")

    assert dataframe.equals(spatial_x_dataframe)


def test_dataframe_conditions_sorting():
    # Creation of an ordered example list.
    macular_analysis_dataframes_test.dict_paths_pyb = {"wAmaGang10Hz": '',
                                                       "wAmaGang3,8Hz": '',
                                                       "barSpeed6dps": ''}

    # Case of default sorting by alphabetical order and KeyError.
    assert (macular_analysis_dataframes_test.dataframe_conditions_sorting() ==
            ["barSpeed6dps", "wAmaGang10Hz", "wAmaGang3,8Hz"])

    # Case of default sorting by alphabetical order.
    macular_analysis_dataframes_test.multiple_dicts_analysis["Conditions"] = {"sorting": False}
    assert (macular_analysis_dataframes_test.dataframe_conditions_sorting() ==
            ["barSpeed6dps", "wAmaGang10Hz", "wAmaGang3,8Hz"])

    # Case of sorting based on a list defined in the multiple condition analysis dictionary.
    macular_analysis_dataframes_test.multiple_dicts_analysis["Conditions"]["sorting"] = ["wAmaBip10Hz", "wAmaGang3,8Hz",
                                                                                        "barSpeed6dps"]
    assert (macular_analysis_dataframes_test.dataframe_conditions_sorting() ==
            ["wAmaBip10Hz", "wAmaGang3,8Hz", "barSpeed6dps"])

    # Case of sorting based on condition names and the ‘NameValueUnit’ format.
    macular_analysis_dataframes_test.multiple_dicts_analysis["Conditions"]["sorting"] = "NameValueUnit"
    assert (macular_analysis_dataframes_test.dataframe_conditions_sorting() ==
            ["barSpeed6dps", "wAmaGang3,8Hz", "wAmaGang10Hz"])

    # Case of sorting based on condition names and the ‘NameValueUnit’ format and with complex conditions to sort.
    macular_analysis_dataframes_test.dict_paths_pyb = {"barSpeed30dps_wAmaBip10Hz_wAmaGang0,1Hz": '',
                                                       "barSpeed6dps_wAmaBip3Hz": '',
                                                       "barSpeed6dps_wAmaGang0,1Hz": '',
                                                       "barSpeed6dps": ''}
    assert macular_analysis_dataframes_test.dataframe_conditions_sorting() == ["barSpeed6dps",
                                                                               "barSpeed6dps_wAmaBip3Hz",
                                                                               "barSpeed6dps_wAmaGang0,1Hz",
                                                                               "barSpeed30dps_wAmaBip10Hz_wAmaGang0,1Hz"
                                                                               ]


def test_name_value_unit_sorting_conditions():
    # Creation of an ordered sample list.
    sorted_conditions = ["barSpeed6dps_wAmaBip3Hz_wAmaGang0,8Hz", "barSpeed6dps_wAmaBip10Hz_wAmaGang0,1Hz",
                         "barSpeed6dps_wAmaBip10Hz_wAmaGang0,8Hz", "barSpeed30dps_wAmaBip10Hz_wAmaGang0,1Hz"]

    # Modification of dict_paths_pyb to add conditions to 3 disordered parameters to be sorted.
    macular_analysis_dataframes_test.dict_paths_pyb = {"barSpeed30dps_wAmaBip10Hz_wAmaGang0,1Hz": '',
                                                       "barSpeed6dps_wAmaBip10Hz_wAmaGang0,8Hz": '',
                                                       "barSpeed6dps_wAmaBip3Hz_wAmaGang0,8Hz": '',
                                                       "barSpeed6dps_wAmaBip10Hz_wAmaGang0,1Hz": ''}

    # Verify that the conditions are sorted correctly.
    macular_analysis_dataframes_test.name_value_unit_sorting_conditions()
    assert macular_analysis_dataframes_test.name_value_unit_sorting_conditions() == sorted_conditions


def test_cleaning_multiple_dicts_analysis():
    # Cases of empty analysis dataframe dictionaries.
    assert macular_analysis_dataframes_test.cleaning_multiple_dicts_features(
        multiple_dicts_analysis_barSpeed_head100) == {}

    # Case of analysis dataframe dictionaries with True and False values.
    multiple_dicts_preprocessings_barSpeed_head100_copy = multiple_dicts_preprocessings_barSpeed_head100.copy()
    multiple_dicts_preprocessings_barSpeed_head100_copy["global"]["VSDI"] = True
    multiple_dicts_preprocessings_barSpeed_head100_copy["global"]["centering"] = False
    multiple_dicts_preprocessings_barSpeed_head100_copy["barSpeed6dps"]["centering"] = False
    assert macular_analysis_dataframes_test.cleaning_multiple_dicts_features(
        multiple_dicts_preprocessings_barSpeed_head100_copy) == {"global": {"VSDI": True}}

    # Case of a dictionary of analysis dataframes containing only True values.
    assert macular_analysis_dataframes_test.cleaning_multiple_dicts_features(
        multiple_dicts_simulations_barSpeed_head100) == multiple_dicts_simulations_barSpeed_head100


def test_setup_conditions_values_to_condition_dataframe():
    # Opening a dataframe of conditions that have already been set up.
    with open(f"{path_data_test}/setup_condition_dataframe.pyb", "rb") as file:
        setup_conditions_dataframe = pickle.load(file)

    # Opening a dataframe of conditions in setup.
    with open(f"{path_data_test}/empty_condition_dataframe.pyb", "rb") as file:
        empty_dataframe = pickle.load(file)

    # Adaptation of a Macular Analysis Dataframe to set up your conditions dataframe.
    macular_analysis_dataframes_test._dict_analysis_dataframes = {"Conditions": empty_dataframe.copy()}
    macular_analysis_dataframes_test.setup_conditions_values_to_condition_dataframe()

    # Test of setting up the conditions dataframe.
    assert setup_conditions_dataframe.equals(macular_analysis_dataframes_test._dict_analysis_dataframes["Conditions"])

    # Opening a dataframe of complex conditions that have already been set up.
    with open(f"{path_data_test}/setup_complex_condition_dataframe.pyb", "rb") as file:
        setup_complex_conditions_dataframe = pickle.load(file)

    # Preparation of a macular analysis dataframe for complex conditions.
    macular_analysis_dataframes_test.dict_paths_pyb = {"barSpeed6dps_wAmaGang0,8Hz": '',
                                                       "barSpeed6dps_wAmaBip10Hz": '',
                                                       "barSpeed6dps": "",
                                                       "barSpeed30dps": "",
                                                       "barSpeed6dps_wAmaBip10Hz_wAmaGang0,1Hz": ""}
    macular_analysis_dataframes_test.multiple_dicts_analysis["Conditions"] = {"sorting": "NameValueUnit"}
    macular_analysis_dataframes_test.initialize_dict_analysis_dataframes()

    # Test of setting up a complex conditions dataframe.
    assert setup_complex_conditions_dataframe.equals(macular_analysis_dataframes_test._dict_analysis_dataframes["Conditions"])
