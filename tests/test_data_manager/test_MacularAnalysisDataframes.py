import copy
import os
import pickle
import re

import numpy as np

from src.data_manager.MacularDictArray import MacularDictArray
from src.data_manager.MacularAnalysisDataframes import MacularAnalysisDataframes

# Get data for test from relative path.
path_data_test = os.path.normpath(f"{os.getcwd()}/../data_test/data_manager/")

# Import of a MacularAnalysisDataframes based on reduced MacularDictArray (100 first rows).
with open(f"{path_data_test}/MacularAnalysisDataframes/initialized_macular_analysis_dataframe.pyb", "rb") as file:
    macular_analysis_dataframes_head100 = pickle.load(file)

# Import of a reduced MacularAnalysisDataframes for tests.
with open(f"{path_data_test}/MacularAnalysisDataframes/initialized_macular_analysis_dataframe.pyb", "rb") as file:
    macular_analysis_dataframes_test = pickle.load(file)

# Import a multiple reduced macular dict array of bar speed condition.
with open(f"{path_data_test}/MacularAnalysisDataframes/multiple_macular_dict_array_head100.pyb", "rb") as file:
    multi_macular_dict_array_head100 = pickle.load(file)

# Import a multiple reduced macular dict array of bar speed condition for tests.
with open(f"{path_data_test}/MacularAnalysisDataframes/multiple_macular_dict_array_head100.pyb", "rb") as file:
    multi_macular_dict_array_test = pickle.load(file)

# Import a default multiple macular dict array of bar speed condition with multiple preprocess.
with open(f"{path_data_test}/MacularAnalysisDataframes/multiple_macular_dict_array_default.pyb", "rb") as file:
    multi_macular_dict_array_default = pickle.load(file)

# Import the list of conditions/measures from the default multi macular dict array.
with open(f"{path_data_test}/MacularAnalysisDataframes/levels_multiple_dictionaries_default.pyb", "rb") as file:
    levels_multiple_dictionaries_default = pickle.load(file)

# Import default macular analysis dataframes of bar speed condition with activation time common group analysis.
with open(f"{path_data_test}/MacularAnalysisDataframes/activation_time_common_group_analysis.pyb", "rb") as file:
    activation_time_common_group_analysis = pickle.load(file)

multiple_dicts_simulations_head100 = {
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
multiple_dicts_simulations_head100_nopyb = copy.deepcopy(multiple_dicts_simulations_head100)
del multiple_dicts_simulations_head100_nopyb["barSpeed6dps"]["path_pyb"]
del multiple_dicts_simulations_head100_nopyb["barSpeed15dps"]["path_pyb"]
del multiple_dicts_simulations_head100_nopyb["barSpeed30dps"]["path_pyb"]

multiple_dicts_preprocessings_head100 = {
    "global": {},
    "barSpeed6dps": {},
    "barSpeed15dps": {},
    "barSpeed30dps": {}
}

multiple_dicts_analysis_head100 = {
    "Conditions": {"sorting": "NameValueUnit"},
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
    assert macular_analysis_dataframes_head100.dict_paths_pyb == {
        "barSpeed6dps": f"{path_data_test}/RC_RM_dSGpCP0026_barSpeed6dps_head100_copy_0f.pyb",
        "barSpeed15dps": f"{path_data_test}/RC_RM_dSGpCP0028_barSpeed15dps_head100_copy_0f.pyb",
        "barSpeed30dps": f"{path_data_test}/RC_RM_dSGpCP0033_barSpeed30dps_head100_0f.pyb"}


def test_dict_paths_pyb_setter():
    # Case of an attempt to modify dict_paths_pyb.
    try:
        macular_analysis_dataframes_test.dict_paths_pyb = {"test": "setter"}
        assert False
    except AttributeError:
        assert True

    dict_paths_pyb_initial = {'barSpeed6dps': '/home/jemonet/Documents/These/Code/macular_figures_lib/tests/data_test/'
                                              'data_manager/RC_RM_dSGpCP0026_barSpeed6dps_head100_copy_0f.pyb',
                              'barSpeed15dps': '/home/jemonet/Documents/These/Code/macular_figures_lib/tests/data_test/'
                                               'data_manager/RC_RM_dSGpCP0028_barSpeed15dps_head100_copy_0f.pyb',
                              'barSpeed30dps': '/home/jemonet/Documents/These/Code/macular_figures_lib/tests/data_test/'
                                               'data_manager/RC_RM_dSGpCP0033_barSpeed30dps_head100_0f.pyb'}

    # Verification that the value of dict_paths_pyb has not changed.
    assert (macular_analysis_dataframes_test.dict_paths_pyb
            == dict_paths_pyb_initial)



def test_dict_analysis_dataframes_getter():
    # Import d'un dictionnaire d'analyses de dataframes d'exemple.
    with open(f"{path_data_test}/MacularAnalysisDataframes/dict_analysis_dataframes_head100.pyb", "rb") as file:
        dict_analysis_dataframes_head100 = pickle.load(file)

    # Comparison of the example dictionary with that of the setter.
    for name_dataframe in macular_analysis_dataframes_head100.dict_analysis_dataframes:
        if name_dataframe == "Conditions":
            assert macular_analysis_dataframes_head100.dict_analysis_dataframes[name_dataframe].equals(
                dict_analysis_dataframes_head100[name_dataframe])
        else:
            for condition in macular_analysis_dataframes_head100.dict_analysis_dataframes[name_dataframe]:
                assert (macular_analysis_dataframes_head100.dict_analysis_dataframes[name_dataframe][condition]
                        .equals(dict_analysis_dataframes_head100[name_dataframe][condition]))


def test_dict_analysis_dataframes_setter():
    old_dict_analysis_dataframes = copy.deepcopy(macular_analysis_dataframes_test.dict_analysis_dataframes)
    macular_analysis_dataframes_test.dict_analysis_dataframes = {}
    assert macular_analysis_dataframes_test.dict_analysis_dataframes != old_dict_analysis_dataframes
    assert macular_analysis_dataframes_test.dict_analysis_dataframes == {}


def test_multiple_dicts_analysis_getter():
    assert (macular_analysis_dataframes_head100.multiple_dicts_analysis
            == {'Conditions': {"sorting": "NameValueUnit"}, "X": {"test": "test"}, "Y": {"test": "test"},
                "Time": {"test": "test"}})


def test_multiple_dicts_analysis_setter():
    macular_analysis_dataframes_test.multiple_dicts_analysis = {"test": {"setter1": True, "setter2": False},
                                                                "test2": {"setter3": False}, "test3": {}}
    assert macular_analysis_dataframes_test.multiple_dicts_analysis == {"test": {"setter1": True}}


def test_multiple_dicts_preprocessings_getter():
    assert (macular_analysis_dataframes_head100.multiple_dicts_preprocessings
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
    assert (macular_analysis_dataframes_head100.multiple_dicts_simulations
            == multiple_dicts_simulations_head100_nopyb)


def test_multiple_dicts_simulations_setter():
    # Case of an attempt to modify multiple_dicts_simulations.
    try:
        macular_analysis_dataframes_test.multiple_dicts_simulations = {"test": "setter"}
        assert False
    except AttributeError:
        assert True

    # Verification that the value of multiple_dicts_simulations has not changed.
    assert (macular_analysis_dataframes_test.multiple_dicts_simulations
            == multiple_dicts_simulations_head100_nopyb)


def test_condition_reg_getter():
    assert macular_analysis_dataframes_head100.condition_reg == re.compile(
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
        multiple_dicts_simulations_head100, multiple_dicts_preprocessings_head100)
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
    # Opening a macular analysis dataframe that has already been created but has yet to be initialised.
    with open(f"{path_data_test}/MacularAnalysisDataframes/macular_analysis_dataframes_to_init.pyb", "rb") as file:
        macular_analysis_dataframes_to_init = pickle.load(file)

    # Index listing.
    t_index = multi_macular_dict_array_head100["barSpeed6dps"].index["temporal"]
    x_index = multi_macular_dict_array_head100["barSpeed6dps"].index["spatial_x"]
    y_index = multi_macular_dict_array_head100["barSpeed6dps"].index["spatial_y"]

    # Changed the multiple analysis dictionaries to prevent it from being empty.
    macular_analysis_dataframes_to_init._multiple_dicts_analysis = {"Conditions": {"sorting": "NameValueUnit"}, "X": {},
                                                                    "Y": {}, "Time": {}}

    # Initialisation of Macular analysis dataframes.
    macular_analysis_dataframes_to_init.initialize_dict_analysis_dataframes(x_index, y_index, t_index)

    # Verify that the conditions dataframe is correct.
    assert macular_analysis_dataframes_head100.dict_analysis_dataframes["Conditions"].equals(
        macular_analysis_dataframes_to_init.dict_analysis_dataframes["Conditions"])

    # Verify that the X, Y, and T dataframes for each condition are equal.
    for condition in macular_analysis_dataframes_head100.dict_paths_pyb:
        assert macular_analysis_dataframes_head100.dict_analysis_dataframes["X"][condition].equals(
            macular_analysis_dataframes_to_init.dict_analysis_dataframes["X"][condition])
        assert macular_analysis_dataframes_head100.dict_analysis_dataframes["Y"][condition].equals(
            macular_analysis_dataframes_to_init.dict_analysis_dataframes["Y"][condition])
        assert macular_analysis_dataframes_head100.dict_analysis_dataframes["Time"][condition].equals(
            macular_analysis_dataframes_to_init.dict_analysis_dataframes["Time"][condition])


def test_initialize_analysis_dataframe():
    # Open a example dataframe from the spatial dataframe x.
    with open(f"{path_data_test}/MacularAnalysisDataframes/empty_spatial_x_dataframe.pyb", "rb") as file:
        spatial_x_dataframe = pickle.load(file)

    # Create a spatial dataframe x using the initialize_analysis_dataframe function.
    dataframe = macular_analysis_dataframes_test.initialize_analysis_dataframe(
        multi_macular_dict_array_head100["barSpeed6dps"].index["spatial_x"], "X")

    assert dataframe.equals(spatial_x_dataframe)


def test_dataframe_conditions_sorting():
    # Creation of an ordered example list.
    macular_analysis_dataframes_test._dict_paths_pyb = {"wAmaGang10Hz": '',
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
    macular_analysis_dataframes_test._dict_paths_pyb = {"barSpeed30dps_wAmaBip10Hz_wAmaGang0,1Hz": '',
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
    macular_analysis_dataframes_test._dict_paths_pyb = {"barSpeed30dps_wAmaBip10Hz_wAmaGang0,1Hz": '',
                                                       "barSpeed6dps_wAmaBip10Hz_wAmaGang0,8Hz": '',
                                                       "barSpeed6dps_wAmaBip3Hz_wAmaGang0,8Hz": '',
                                                       "barSpeed6dps_wAmaBip10Hz_wAmaGang0,1Hz": ''}

    # Verify that the conditions are sorted correctly.
    macular_analysis_dataframes_test.name_value_unit_sorting_conditions()
    assert macular_analysis_dataframes_test.name_value_unit_sorting_conditions() == sorted_conditions


def test_cleaning_multiple_dicts_analysis():
    # Create empty analysis dataframe dictionaries.
    multiple_dicts_analysis_empty = {"Conditions": {}, "X": {}, "Y": {}, "Time": {}}
    # Cases of empty analysis dataframe dictionaries.
    assert macular_analysis_dataframes_test.cleaning_multiple_dicts_features(
        multiple_dicts_analysis_empty) == {}

    # Case of analysis dataframe dictionaries with True and False values.
    multiple_dicts_preprocessings_head100_copy = copy.deepcopy(multiple_dicts_preprocessings_head100)
    multiple_dicts_preprocessings_head100_copy["global"]["VSDI"] = True
    multiple_dicts_preprocessings_head100_copy["global"]["centering"] = False
    multiple_dicts_preprocessings_head100_copy["barSpeed6dps"]["centering"] = False
    assert macular_analysis_dataframes_test.cleaning_multiple_dicts_features(
        multiple_dicts_preprocessings_head100_copy) == {"global": {"VSDI": True}}

    # Case of a dictionary of analysis dataframes containing only True values.
    assert macular_analysis_dataframes_test.cleaning_multiple_dicts_features(
        multiple_dicts_simulations_head100) == multiple_dicts_simulations_head100


def test_setup_conditions_values_to_condition_dataframe():
    # Opening a dataframe of conditions that have already been set up.
    with open(f"{path_data_test}/MacularAnalysisDataframes/setup_condition_dataframe.pyb", "rb") as file:
        setup_conditions_dataframe = pickle.load(file)

    # Opening a dataframe of conditions in setup.
    with open(f"{path_data_test}/SpatialAnalyser/empty_condition_dataframe.pyb", "rb") as file:
        empty_dataframe = pickle.load(file)

    # Adaptation of a Macular Analysis Dataframe to set up your conditions dataframe.
    macular_analysis_dataframes_test._dict_analysis_dataframes = {"Conditions": copy.deepcopy(empty_dataframe)}
    macular_analysis_dataframes_test.setup_conditions_values_to_condition_dataframe()

    # Test of setting up the conditions dataframe.
    assert setup_conditions_dataframe.equals(macular_analysis_dataframes_test._dict_analysis_dataframes["Conditions"])

    # Opening a dataframe of complex conditions that have already been set up.
    with open(f"{path_data_test}/MacularAnalysisDataframes/setup_complex_condition_dataframe.pyb", "rb") as file:
        setup_complex_conditions_dataframe = pickle.load(file)

    # Preparation of a macular analysis dataframe for complex conditions.
    macular_analysis_dataframes_test._dict_paths_pyb = {"barSpeed6dps_wAmaGang0,8Hz": '',
                                                       "barSpeed6dps_wAmaBip10Hz": '',
                                                       "barSpeed6dps": "",
                                                       "barSpeed30dps": "",
                                                       "barSpeed6dps_wAmaBip10Hz_wAmaGang0,1Hz": ""}
    macular_analysis_dataframes_test.multiple_dicts_analysis["Conditions"] = {"sorting": "NameValueUnit"}
    macular_analysis_dataframes_test.initialize_dict_analysis_dataframes()

    # Test of setting up a complex conditions dataframe.
    assert setup_complex_conditions_dataframe.equals(macular_analysis_dataframes_test
                                                     ._dict_analysis_dataframes["Conditions"])


def test_setup_multiple_dicts_analysis():
    # Import an empty default macular analysis dataframes of bar speed condition.
    with open(f"{path_data_test}/MacularAnalysisDataframes/macular_analysis_dataframe_default_empty.pyb", "rb") as file:
        macular_analysis_dataframes_default_empty = pickle.load(file)

    # Define strings containing all conditions and measures.
    all_conditions = levels_multiple_dictionaries_default[0]
    all_measurements = levels_multiple_dictionaries_default[1]["barSpeed30dps"]

    # Creation of the model multiple analysis dictionaries.
    multiple_dicts_analysis_substituted_correct = {
        'Conditions': {
            'maximal_latency': {'barSpeed27dps': {'VSDI': {'threshold': 0.001, 'y': 7, 'index': 'temporal_ms'}},
                                'barSpeed30dps': {all_measurements: {'threshold': 0.001, 'x': 7,
                                                                     'index': 'temporal'}}}},
        'X': {
            'activation_time': {'barSpeed30dps': {all_measurements: {'threshold': 0.001, 'y': 7,
                                                                     'index': 'temporal'}},
                                all_conditions: {'VSDI': {'threshold': 0.001, 'y': 7, 'index': 'temporal_ms'},
                                                 all_measurements: {'threshold': 0.01, 'y': 7,
                                                                    'index': 'temporal_ms'}}}},
        'Y': {'test': 'test'},
        'Time': {'test': 'test'}}

    # Creation of the model sort order dictionary.
    dict_sort_order_correct = {
        'Conditions': {
            'maximal_latency': {'conditions': ['barSpeed27dps', 'barSpeed30dps'],
                                'measurements': {'barSpeed27dps': ['VSDI'],
                                                 'barSpeed30dps': [all_measurements]}}},
        'X': {
            'activation_time': {'conditions': [all_conditions, 'barSpeed30dps'],
                                'measurements': {'barSpeed30dps': [all_measurements],
                                                 all_conditions: [all_measurements, 'VSDI']}}},
        'Y': {},
        'Time': {}}

    # Initialisation of a dictionary for a complex X analysis.
    spatial_x_dictionary = {"activation_time": {
        "all_conditions": {"all_measurements": {"threshold": 0.01, "y": 7, "index": "temporal_ms"},
                           "VSDI": {"threshold": 0.001, "y": 7, "index": "temporal_ms"}},
        "barSpeed30dps": {"all_measurements": {"threshold": 0.001, "y": 7, "index": "temporal"}}}}

    # Initialisation of a dictionary for a complex conditions analysis.
    conditions_dictionary = {"maximal_latency": {
        "barSpeed27dps": {"VSDI": {"threshold": 0.001, "y": 7, "index": "temporal_ms"}},
        "barSpeed30dps": {"all_measurements": {"threshold": 0.001, "x": 7, "index": "temporal"}}}}

    # Put dictionaries for analysing X and conditions in the empty default macular analysis dataframes.
    macular_analysis_dataframes_default_empty.multiple_dicts_analysis["X"] = copy.deepcopy(spatial_x_dictionary)
    macular_analysis_dataframes_default_empty.multiple_dicts_analysis["Conditions"] = copy.deepcopy(
        conditions_dictionary)

    # Execute the setup_multiple_dicts_analysis function to be tested.
    multiple_dicts_analysis_substituted, dict_sort_order = (macular_analysis_dataframes_default_empty.
    setup_multiple_dicts_analysis(
        multi_macular_dict_array_default))

    # Verification of the validity of the substituted multiple analysis dictionaries and the sort order dictionary.
    assert multiple_dicts_analysis_substituted == multiple_dicts_analysis_substituted_correct
    assert dict_sort_order == dict_sort_order_correct

    # Checking that the multiple analysis dictionaries has not been modified in the macular analysis dataframes.
    assert macular_analysis_dataframes_default_empty.multiple_dicts_analysis["X"] == spatial_x_dictionary
    assert macular_analysis_dataframes_default_empty.multiple_dicts_analysis["Conditions"] == conditions_dictionary


def test_get_levels_of_multi_macular_dict_array():
    # Import a test multiple reduced macular dict array of bar speed condition.
    with open(f"{path_data_test}/MacularAnalysisDataframes/multiple_macular_dict_array_head100.pyb",
              "rb") as file_variousLevel:
        multi_macular_dict_array_head100_variousLevel = pickle.load(file_variousLevel)

    # Modification of multi_macular_dict_array so that one of the MacularDictArray has one less measurement.
    del multi_macular_dict_array_head100_variousLevel["barSpeed30dps"].data["v_i_CorticalInhibitory"]

    # Modification of multi_macular_dict_array so that one of the MacularDictArray has different measurements.
    for measurement in multi_macular_dict_array_head100_variousLevel["barSpeed15dps"].data.copy():
        if measurement == "FiringRate_GanglionGainControl":
            multi_macular_dict_array_head100_variousLevel["barSpeed15dps"].data["test"] = (
                multi_macular_dict_array_head100_variousLevel["barSpeed15dps"].data)[measurement]
        del multi_macular_dict_array_head100_variousLevel["barSpeed15dps"].data[measurement]

    # Character string containing all the measures in the MacularDictArray.
    all_measurements = ("BipolarResponse_BipolarGainControl:FiringRate_GanglionGainControl:V_Amacrine:"
                        "V_BipolarGainControl:V_GanglionGainControl:muVn_CorticalExcitatory:muVn_CorticalInhibitory:"
                        "v_e_CorticalExcitatory:v_i_CorticalInhibitory")

    # Creation of a dictionary associating conditions in the test MacularDictArray with the measurements it contains.
    dict_all_measurements = {"barSpeed6dps": all_measurements, "barSpeed15dps": "test",
                             "barSpeed30dps": ":".join(all_measurements.split(":")[:-1])}

    # Extracting condition names and measures with get_levels_of_multi_macular_dict_array.
    levels_multiple_dictionaries = macular_analysis_dataframes_head100.get_levels_of_multi_macular_dict_array(
        multi_macular_dict_array_head100_variousLevel)

    # Verification of the character string of the conditions.
    assert levels_multiple_dictionaries[0] == 'barSpeed15dps:barSpeed30dps:barSpeed6dps'

    # Verification of the character string dictionary for measurements for each condition.
    assert levels_multiple_dictionaries[1] == dict_all_measurements


def test_substituting_all_alias_in_multiple_analysis_dictionaries():
    # Define strings containing all conditions and measures.
    all_conditions = levels_multiple_dictionaries_default[0]
    all_measurements = levels_multiple_dictionaries_default[1]["barSpeed30dps"]

    # Creation of the model multiple analysis dictionaries.
    multiple_dicts_analysis_substitued_correct = {
        'Conditions': {'maximal_latency': {
            'barSpeed30dps': {'VSDI': {'param1': 1}, all_measurements: {'param1': 2},
                              'BipolarResponse_BipolarGainControl': {'param1': 2}},
            all_conditions: {'VSDI': {'param1': 1}}}},
        'X': {'activation_time': {
            'barSpeed30dps': {'VSDI': {'param1': 1},
                              all_measurements: {'param1': 2},
                              'BipolarResponse_BipolarGainControl': {'param1': 2}},
            'barSpeed27dps:barSpeed30dps': {'VSDI': {'param1': 1}}}},
        'Y': {'activation_time': {
            'barSpeed30dps': {'VSDI': {'param1': 1},
                              all_measurements: {'param1': 2},
                              'BipolarResponse_BipolarGainControl': {'param1': 2}},
            all_conditions: {'VSDI': {'param1': 1}}}},
        'Time': {'test': {
            'barSpeed30dps': {'VSDI': {'param1': 1},
                              all_measurements: {'param1': 2},
                              'BipolarResponse_BipolarGainControl': {'param1': 2}},
            all_conditions: {'VSDI': {'param1': 1}}}}}

    # Initialisation of a multiple analysis dictionaries for a complex analysis.
    dict_analysis_test_default = {
        "Conditions": {"maximal_latency": {
            "barSpeed30dps": {"VSDI": {"param1": 1}, "all_measurements": {"param1": 2},
                              "BipolarResponse_BipolarGainControl": {"param1": 2}},
            "all_conditions": {"VSDI": {"param1": 1}}
        }},
        "X": {"activation_time": {
            "barSpeed30dps": {"VSDI": {"param1": 1}, "all_measurements": {"param1": 2},
                              "BipolarResponse_BipolarGainControl": {"param1": 2}},
            "all_conditions": {"VSDI": {"param1": 1}}
        }},
        "Y": {"activation_time": {
            "barSpeed30dps": {"VSDI": {"param1": 1}, "all_measurements": {"param1": 2},
                              "BipolarResponse_BipolarGainControl": {"param1": 2}},
            "all_conditions": {"VSDI": {"param1": 1}}
        }},
        "Time": {"test": {
            "barSpeed30dps": {"VSDI": {"param1": 1}, "all_measurements": {"param1": 2},
                              "BipolarResponse_BipolarGainControl": {"param1": 2}},
            "all_conditions": {"VSDI": {"param1": 1}}
        }}
    }

    # Execute the substituting_all_alias_in_multiple_analysis_dictionaries function to be tested.
    multiple_dicts_analysis_substitued = (macular_analysis_dataframes_test.
    substituting_all_alias_in_multiple_analysis_dictionaries(
        dict_analysis_test_default, levels_multiple_dictionaries_default))

    assert multiple_dicts_analysis_substitued == multiple_dicts_analysis_substitued_correct


def test_substituting_all_alias_in_analysis_dictionary():
    # Initialisation of a dictionary for a complex analysis.
    dict_analysis_head100 = {"activation_time": {
        "all_conditions": {"all_measurements": {"threshold": 0.01, "y": 7, "index": "temporal_index"},
                           "VSDI": {"threshold": 0.001, "y": 7, "index": "temporal_index"}},
        "barSpeed30dps": {"all_measurements": {"threshold": 0.005, "y": 7, "index": "temporal_index"}},
        "barSpeed6dps:barSpeed30dps": {"BipolarResponse_BipolarGainControl": {"threshold": 0.005, "y": 7,
                                                                              "index": "temporal_index"}},
        "barSpeed15dps:barSpeed30dps:barSpeed6dps": {"VSDI": {"threshold": 0.001, "y": 10, "index": "temporal_index"}}
    },
        "test1": 1,
        "test2": {"all_conditions": 2},
        "test3": {"all_conditions": {"all_measurements": 3}}
    }

    # Creation of character strings with all conditions of the MacularAnalysisDataframe.
    all_conditions = ":".join(sorted([condition for condition in macular_analysis_dataframes_head100.dict_paths_pyb]))

    # Creation of character strings with all measurements from the MacularAnalysisDataframe.
    all_measurements = ":".join(sorted([measure for measure in multi_macular_dict_array_head100[
        all_conditions.split(":")[0]].data]))

    # Creation of the structure containing the names of conditions and measures separated by ‘:’.
    levels_multiple_dictionaries = [all_conditions, {"barSpeed15dps": all_measurements,
                                                     "barSpeed6dps": all_measurements,
                                                     "barSpeed30dps": ":".join(all_measurements.split(":")[:5])}]

    # Definition of the same multiple analysis dictionaries but with the aliases substituted.
    multiple_dicts_analysis_head100_spatialAnalysis_substracted = {"activation_time": {
        f"{all_conditions}": {f"{all_measurements}": {"threshold": 0.01, "y": 7, "index": "temporal_index"},
                              "VSDI": {"threshold": 0.001, "y": 7, "index": "temporal_index"}},
        "barSpeed30dps": {"BipolarResponse_BipolarGainControl:FiringRate_GanglionGainControl:V_Amacrine:"
                          "V_BipolarGainControl:V_GanglionGainControl": {"threshold": 0.005, "y": 7,
                                                                         "index": "temporal_index"}},
        "barSpeed6dps:barSpeed30dps": {"BipolarResponse_BipolarGainControl": {"threshold": 0.005, "y": 7,
                                                                              "index": "temporal_index"}}
    },
        "test1": 1,
        "test2": {all_conditions: 2},
        "test3": {all_conditions: {all_measurements: 3}}
    }

    # Substitution of aliases with the substituting_analysis_dictionary_all_alias function.
    macular_analysis_dataframes_test.substituting_all_alias_in_analysis_dictionary(
        dict_analysis_head100, "activation_time", levels_multiple_dictionaries)
    macular_analysis_dataframes_test.substituting_all_alias_in_analysis_dictionary(
        dict_analysis_head100, "test1", levels_multiple_dictionaries)
    macular_analysis_dataframes_test.substituting_all_alias_in_analysis_dictionary(
        dict_analysis_head100, "test2", levels_multiple_dictionaries)
    macular_analysis_dataframes_test.substituting_all_alias_in_analysis_dictionary(
        dict_analysis_head100, "test3", levels_multiple_dictionaries)

    # Verification of correct substitutions.
    assert (dict_analysis_head100 ==
            multiple_dicts_analysis_head100_spatialAnalysis_substracted)


def test_creating_sort_order_from_multiple_dicts_analysis():
    # Define strings containing all conditions and measures.
    all_conditions = levels_multiple_dictionaries_default[0]
    all_measurements = levels_multiple_dictionaries_default[1]["barSpeed30dps"]

    # Initialisation of multiple analysis dictionaries for a complex analysis.
    dict_analysis_test_default = {
        "Conditions": {"analysis1": 1, "analysis2": {"barSpeed30dps": 2},
                       "analysis3": {"barSpeed30dps": {"measure3": 3}}},
        "X": {"activation_time": {
            "barSpeed27dps": {"VSDI": {"param1": 1}},
            "barSpeed30dps": {"VSDI": {"param1": 1}, all_measurements: {"param1": 2},
                              "BipolarResponse_BipolarGainControl": {"param1": 2}},
            all_conditions: {"VSDI": {"param1": 1}}
        }},
        "Y": {},
        "Time": {}
    }

    # Creation of the model sort order dictionary.
    dict_sort_order_correct = {
        'Conditions': {"analysis3": {"conditions": ["barSpeed30dps"],
                                     "measurements": {"barSpeed30dps": ["measure3"]}}},
        'X': {'activation_time': {'conditions': [all_conditions, 'barSpeed27dps', 'barSpeed30dps'],
                                  'measurements': {'barSpeed27dps': ['VSDI'],
                                                   'barSpeed30dps': [all_measurements,
                                                                     'BipolarResponse_BipolarGainControl', 'VSDI'],
                                                   all_conditions: ['VSDI']}}},
        'Y': {},
        'Time': {}}

    # Execute the creating_sort_order_from_multiple_dicts_analysis function to be tested.
    dict_sort_order = macular_analysis_dataframes_test.creating_sort_order_from_multiple_dicts_analysis(
        dict_analysis_test_default, levels_multiple_dictionaries_default)

    # Verification of correct sorting.
    assert dict_sort_order == dict_sort_order_correct


def test_creating_sort_order_from_dict_analysis():
    # Define strings containing all conditions and measures or one less.
    all_conditions = levels_multiple_dictionaries_default[0]
    all_measurements = levels_multiple_dictionaries_default[1]["barSpeed30dps"]
    all_measurements_minus_one = ":".join(all_measurements.split(":")[:-1])

    # Initialisation of a dictionary for a complex analysis.
    dict_analysis = {
        all_conditions: {all_measurements: {"threshold": 0.01, "y": 7, "index": "temporal_index"}},
        "barSpeed1dps": {all_measurements: {"threshold": 0.005, "y": 7, "index": "temporal_index"},
                         "VSDI": {"threshold": 0.001, "y": 7, "index": "temporal_index"},
                         "Activity": {"threshold": 0.001, "y": 7, "index": "temporal_index"}},
        "barSpeed3dps": {all_measurements_minus_one: {"threshold": 0.005, "y": 7,
                                                      "index": "temporal_index"},
                         "Activity": {"threshold": 0.001, "y": 7, "index": "temporal_index"}
                         }
    }

    # Case of a sort order for conditions.
    sorted_list_conditions = MacularAnalysisDataframes.creating_sort_order_from_dict_analysis(
        dict_analysis, all_conditions)
    assert sorted_list_conditions == [all_conditions, "barSpeed1dps", "barSpeed3dps"]

    # Case of a sort order for measurements.
    sorted_list_mesurements = MacularAnalysisDataframes.creating_sort_order_from_dict_analysis(
        dict_analysis["barSpeed1dps"], all_measurements)
    assert sorted_list_mesurements == [all_measurements, "Activity", "VSDI"]

    # Case of a sort order without ‘all’ terms.
    sorted_list_mesurements_no_all = MacularAnalysisDataframes.creating_sort_order_from_dict_analysis(
        dict_analysis["barSpeed3dps"], all_measurements)
    assert sorted_list_mesurements_no_all == ["Activity", ":".join(all_measurements.split(":")[:-1])]


def test_make_spatial_dataframes_analysis():
    pass


def test_analysis():
    # Import a default macular analysis dataframes of bar speed condition with one complex analysis done.
    with open(f"{path_data_test}/MacularAnalysisDataframes/"
              f"macular_analysis_dataframe_default_complex_make_analysis.pyb", "rb") as file:
        macular_analysis_dataframes_default_complex_make_analysis = pickle.load(file)

    # Import an empty default macular analysis dataframes of bar speed condition.
    with open(f"{path_data_test}/MacularAnalysisDataframes/macular_analysis_dataframe_default_empty.pyb", "rb") as file:
        macular_analysis_dataframes_default_empty = pickle.load(file)

    # Define strings containing all conditions and measures.
    all_conditions = levels_multiple_dictionaries_default[0]
    all_measurements = levels_multiple_dictionaries_default[1]["barSpeed30dps"]

    # Initialisation of a dictionary for a complex analysis.
    dict_analysis_default_complex = {"X": {"activation_time": {
        all_conditions: {all_measurements: {"threshold": 0.01, "y": 7, "index": "temporal_ms"},
                         "VSDI": {"threshold": 0.001, "y": 7, "index": "temporal_ms"}},
        "barSpeed30dps": {all_measurements: {"threshold": 0.001, "y": 7, "index": "temporal"}},
        "barSpeed27dps": {"BipolarResponse_BipolarGainControl": {"threshold": 0.005, "y": 7, "index": "temporal"}}}}}

    # Initialisation of a sort order dictionary for a complex analysis.
    dict_sort_order_default_complex = {'X': {'activation_time': {
        'conditions': [all_conditions, 'barSpeed27dps', 'barSpeed30dps'],
        'measurements': {all_conditions: [all_measurements, 'VSDI'],
                         'barSpeed30dps': [all_measurements],
                         'barSpeed27dps': ['BipolarResponse_BipolarGainControl']}}}}

    # Use activation time analysis on empty macular analysis dataframes with the complex default dictionaries.
    MacularAnalysisDataframes.activation_time_analyzing(
        macular_analysis_dataframes_default_empty, multi_macular_dict_array_default, dict_analysis_default_complex["X"],
        "X", "activation_time", dict_sort_order_default_complex)

    # Verify that the X, Y, and T dataframes for each condition are equal.
    for condition in macular_analysis_dataframes_default_empty.dict_paths_pyb:
        assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["X"][condition].equals(
            macular_analysis_dataframes_default_complex_make_analysis.dict_analysis_dataframes["X"][condition])
        assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["Y"][condition].equals(
            macular_analysis_dataframes_default_complex_make_analysis.dict_analysis_dataframes["Y"][condition])
        assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["Time"][condition].equals(
            macular_analysis_dataframes_default_complex_make_analysis.dict_analysis_dataframes["Time"][condition])


def test_common_analysis_group_parser():
    # Names of conditions in an analysis group common to two conditions.
    grouped_conditions = "barSpeed6dps_ampGang5Hz:barSpeed30dps"

    # Names of conditions in an analysis group that are common to three measurements.
    grouped_measurements = "FiringRate_GanglionGainControl:BipolarResponse_BipolarGainControl:VSDI"

    # Pairs of conditions and measurements in a common analysis group with 2 conditions and 3 measures.
    common_analysis_group_generator_correct = (analysis_pair for analysis_pair in
                                               [("barSpeed6dps_ampGang5Hz", "FiringRate_GanglionGainControl"),
                                                ("barSpeed6dps_ampGang5Hz", "BipolarResponse_BipolarGainControl"),
                                                ("barSpeed6dps_ampGang5Hz", "VSDI"),
                                                ("barSpeed30dps", "FiringRate_GanglionGainControl"),
                                                ("barSpeed30dps", "BipolarResponse_BipolarGainControl"),
                                                ("barSpeed30dps", "VSDI")])
    common_analysis_group_generator = macular_analysis_dataframes_test.common_analysis_group_parser(
        grouped_conditions, grouped_measurements)

    for analysis_pair, analysis_pair_correct in zip(common_analysis_group_generator,
                                                    common_analysis_group_generator_correct):
        assert analysis_pair == analysis_pair_correct


def test_make_common_group_analysis():
    # Import an empty default macular analysis dataframes of bar speed condition.
    with open(f"{path_data_test}/MacularAnalysisDataframes/macular_analysis_dataframe_default_empty.pyb", "rb") as file:
        macular_analysis_dataframes_default_empty = pickle.load(file)

    # Setup parameters for common group analysis.
    common_analysis_group_generator = (analysis_pair for analysis_pair in
                                       [("barSpeed27dps", "FiringRate_GanglionGainControl"), ("barSpeed30dps", "VSDI")])
    parameters_analysis_dict = {"threshold": 0.001, "y": 7, "index": "temporal_ms", "flag": "threshold0,001_y7"}

    # Make one common group analysis.
    macular_analysis_dataframes_default_empty.make_common_group_analysis(
        MacularAnalysisDataframes.activation_time_analyzing.__wrapped__,
        multi_macular_dict_array_default, common_analysis_group_generator,
        "X", "activation_time", parameters_analysis_dict)

    # Verify that the conditions dataframe is correct.
    assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["Conditions"].equals(
        activation_time_common_group_analysis.dict_analysis_dataframes["Conditions"])

    # Verify that the X, Y, and T dataframes for each condition are equal.
    for condition in macular_analysis_dataframes_default_empty.dict_paths_pyb:
        assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["X"][condition].equals(
            activation_time_common_group_analysis.dict_analysis_dataframes["X"][condition])
        assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["Y"][condition].equals(
            activation_time_common_group_analysis.dict_analysis_dataframes["Y"][condition])
        assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["Time"][condition].equals(
            activation_time_common_group_analysis.dict_analysis_dataframes["Time"][condition])

    print()
    print(tabulate(macular_analysis_dataframes_default_empty.dict_analysis_dataframes["X"]["barSpeed27dps"],
                   headers="keys", tablefmt="fancy_grid"))
    print(tabulate(macular_analysis_dataframes_default_empty.dict_analysis_dataframes["X"]["barSpeed30dps"],
                   headers="keys", tablefmt="fancy_grid"))

    # TODO Ajouter un test avec le dataframe condition et sans flag.


def test_activation_time_analyzing():
    # Create analysis dictionary for case on X dimension dataframe.
    parameters_analysis_dict_x = {"threshold": 0.001, "y": 7, "index": "temporal_ms", "flag": "threshold0,001_y7"}

    # Create new activation time array for spatial dataframe of the X dimension.
    activation_time_array_x = MacularAnalysisDataframes.activation_time_analyzing.__wrapped__(
        multi_macular_dict_array_default["barSpeed30dps"].data["VSDI"],
        multi_macular_dict_array_default["barSpeed30dps"].index,
        parameters_analysis_dict_x)

    # Extract correct activation time array X to compare.
    activation_time_array_correct_x = (activation_time_common_group_analysis.dict_analysis_dataframes
    ["X"]["barSpeed30dps"].loc["activation_time_VSDI_threshold0,001_y7"])

    # Verification of the validity of the spatial array X of the activation time.
    assert np.array_equal(activation_time_array_x, activation_time_array_correct_x)

    # Create analysis dictionary for case on Y dimension dataframe.
    parameters_analysis_dict_y = {"threshold": 0.001, "x": 36, "index": "temporal_ms", "flag": "threshold0,001_x36"}

    # Create new activation time array for spatial dataframe of the Y dimension.
    activation_time_array_y = MacularAnalysisDataframes.activation_time_analyzing.__wrapped__(
        multi_macular_dict_array_default["barSpeed30dps"].data["VSDI"],
        multi_macular_dict_array_default["barSpeed30dps"].index,
        parameters_analysis_dict_y)

    # Create correct activation time array Y to compare.
    activation_time_array_correct_y = np.array([282.2, 285.4, 282.2, 280.6, 279, 277.4, 277.4, 277.4, 277.4, 277.4,
                                                279, 280.6, 282.2, 285.4, 282.2])

    # Verification of the validity of the spatial array Y of the activation time.
    assert np.array_equal(activation_time_array_y, activation_time_array_correct_y)


def test_latency_analyzing():
    # Create analysis dictionary for latency case on X dimension dataframe.
    parameters_analysis_dict_x = {"threshold": 0.001, "y": 7, "index": "temporal_centered_ms", "axis": "horizontal",
                                  "flag": "threshold0,001_y7"}

    # Create new latency array for spatial dataframe of the X dimension.
    latency_array_x = MacularAnalysisDataframes.latency_analyzing.__wrapped__(
        multi_macular_dict_array_default["barSpeed30dps"].data["VSDI"],
        multi_macular_dict_array_default["barSpeed30dps"].index,
        parameters_analysis_dict_x)

    # Set correct latency array X to compare.
    latency_array_x_correct = np.array([3.13, 3.63, 0.93, -0.17, -2.87, -5.57, -8.27, -10.97, -13.67, -16.37, -19.07,
                                        -20.17, -22.87, -25.57, -26.67, -27.77, -30.47, -31.57, -32.67, -33.77, -34.87,
                                        -35.97, -37.07, -36.57, -37.67, -38.77, -38.27, -39.37, -40.47, -39.97, -41.07,
                                        -40.57, -40.07, -41.17, -40.67, -41.77, -41.27, -40.77, -41.87, -41.37, -42.47,
                                        -41.97, -41.47, -42.57, -42.07, -41.57, -42.67, -42.17, -41.67, -41.17, -42.27,
                                        -41.77, -42.87, -42.37, -41.87, -42.97, -42.47, -41.97, -43.07, -42.57, -42.07,
                                        -43.17, -42.67, -43.77, -43.27, -42.77, -43.87, -43.37, -44.47, -45.57, -45.07,
                                        -46.17, -45.67])

    # Verification of the validity of the spatial array X of the latency.
    assert np.array_equal(latency_array_x, latency_array_x_correct)


def test_time_to_peak_analyzing():
    # Create analysis dictionary for case on X dimension dataframe.
    parameters_analysis_dict_x = {"y": 7, "index": "temporal_ms"}

    # Create new time to peak array for spatial dataframe of the X dimension.
    time_to_peak_array_x = MacularAnalysisDataframes.time_to_peak_analyzing.__wrapped__(
        multi_macular_dict_array_default["barSpeed30dps"].data["VSDI"],
        multi_macular_dict_array_default["barSpeed30dps"].index,
        parameters_analysis_dict_x)

    # Extract correct time to peak array X to compare.
    time_to_peak_array_correct_x = [189.4, 197.4, 203.8, 211.8, 218.2, 227.8, 234.2, 242.2, 248.6, 256.6, 264.6,
                                    271.0, 279.0, 285.4, 295.0, 299.8, 309.4, 315.8, 323.8, 331.8, 338.2, 346.2,
                                    352.6, 362.2, 368.6, 376.6, 383.0, 391.0, 399.0, 407.0, 413.4, 421.4, 429.4,
                                    435.8, 443.8, 450.2, 459.8, 466.2, 474.2, 480.6, 488.6, 496.6, 503.0, 512.6,
                                    517.4, 527.0, 533.4, 541.4, 547.8, 555.8, 563.8, 571.8, 579.8, 586.2, 594.2,
                                    600.6, 610.2, 616.6, 624.6, 631.0, 639.0, 647.0, 653.4, 661.4, 667.8, 677.4,
                                    682.2, 690.2, 696.6, 703.0, 709.4, 714.2, 720.6]

    # Verification of the validity of the spatial array X of the time to peak.
    assert np.array_equal(time_to_peak_array_x, time_to_peak_array_correct_x)

    # Create analysis dictionary for case on Y dimension dataframe.
    parameters_analysis_dict_y = {"x": 36, "index": "temporal_ms"}

    # Create new time to peak array for spatial dataframe of the Y dimension.
    time_to_peak_array_y = MacularAnalysisDataframes.time_to_peak_analyzing.__wrapped__(
        multi_macular_dict_array_default["barSpeed30dps"].data["VSDI"],
        multi_macular_dict_array_default["barSpeed30dps"].index,
        parameters_analysis_dict_y)

    # Extract correct time to peak array Y to compare.
    time_to_peak_array_correct_y = [506.2, 499.8, 498.2, 498.2, 493.4, 469.4, 459.8, 459.8, 459.8, 467.8, 491.8, 498.2,
                                    498.2, 499.8, 506.2]

    # Verification of the validity of the spatial array Y of the time to peak.
    assert np.array_equal(time_to_peak_array_y, time_to_peak_array_correct_y)


def test_peak_delay_analyzing():
    # Create analysis dictionary for case on X dimension dataframe.
    parameters_analysis_dict_x = {"y": 7, "index": "temporal_centered_ms", "axis": "horizontal"}

    # Create new peak delay array for spatial dataframe of the X dimension.
    delay_to_peak_array_x = MacularAnalysisDataframes.peak_delay_analyzing.__wrapped__(
        multi_macular_dict_array_default["barSpeed30dps"].data["VSDI"],
        multi_macular_dict_array_default["barSpeed30dps"].index,
        parameters_analysis_dict_x)

    # Extract correct peak delay array X to compare.
    delay_to_peak_array_correct_x = [140.73, 141.23, 140.13, 140.63, 139.53, 141.63, 140.53, 141.03,
                                     139.93, 140.43, 140.93, 139.83, 140.33, 139.23, 141.33, 138.63,
                                     140.73, 139.63, 140.13, 140.63, 139.53, 140.03, 138.93, 141.03,
                                     139.93, 140.43, 139.33, 139.83, 140.33, 140.83, 139.73, 140.23,
                                     140.73, 139.63, 140.13, 139.03, 141.13, 140.03, 140.53, 139.43,
                                     139.93, 140.43, 139.33, 141.43, 138.73, 140.83, 139.73, 140.23,
                                     139.13, 139.63, 140.13, 140.63, 141.13, 140.03, 140.53, 139.43,
                                     141.53, 140.43, 140.93, 139.83, 140.33, 140.83, 139.73, 140.23,
                                     139.13, 141.23, 138.53, 139.03, 137.93, 136.83, 135.73, 133.03, 131.93]

    # Verification of the validity of the spatial array X of the peak delay.
    assert np.array_equal(delay_to_peak_array_x, delay_to_peak_array_correct_x)

    # Create analysis dictionary for case on Y dimension dataframe.
    parameters_analysis_dict_y = {"x": 36, "index": "temporal_centered_ms", "axis": "vertical"}

    # Create new peak delay array for spatial dataframe of the Y dimension.
    delay_to_peak_array_y = MacularAnalysisDataframes.peak_delay_analyzing.__wrapped__(
        multi_macular_dict_array_default["barSpeed30dps"].data["VSDI"],
        multi_macular_dict_array_default["barSpeed30dps"].index,
        parameters_analysis_dict_y)

    # Extract correct peak delay array Y to compare.
    delay_to_peak_array_correct_y = [457.53, 443.63, 434.53, 427.03, 414.73, 383.23, 366.13, 358.63, 351.13, 351.63,
                                     368.13, 367.03, 359.53, 353.63, 352.53]

    # Verification of the validity of the spatial array Y of the peak delay.
    assert np.array_equal(delay_to_peak_array_y, delay_to_peak_array_correct_y)


def test_peak_amplitude_analyzing():
    # Create analysis dictionary for case on X dimension dataframe.
    parameters_analysis_dict_x = {"y": 7}

    # Create new amplitude array for spatial dataframe of the X dimension.
    amplitude_array_x = MacularAnalysisDataframes.peak_amplitude_analyzing.__wrapped__(
        multi_macular_dict_array_default["barSpeed30dps"].data["VSDI"],
        multi_macular_dict_array_default["barSpeed30dps"].index,
        parameters_analysis_dict_x)

    # Extract correct amplitude array X to compare.
    amplitude_array_correct_x = [0.041, 0.041, 0.042, 0.04, 0.043, 0.039, 0.043, 0.039, 0.042, 0.039, 0.041, 0.04,
                                 0.04, 0.041, 0.039, 0.042, 0.038, 0.041, 0.038, 0.041, 0.039, 0.04, 0.04, 0.039,
                                 0.041, 0.038, 0.041, 0.038, 0.041, 0.039, 0.04, 0.04, 0.039, 0.041, 0.038, 0.041,
                                 0.038, 0.041, 0.038, 0.041, 0.039, 0.04, 0.04, 0.039, 0.041, 0.038, 0.041, 0.038,
                                 0.041, 0.039, 0.04, 0.04, 0.039, 0.041, 0.038, 0.042, 0.038, 0.042, 0.039, 0.041,
                                 0.04, 0.04, 0.041, 0.039, 0.043, 0.04, 0.044, 0.04, 0.044, 0.041, 0.043, 0.043, 0.042]

    # Verification of the validity of the spatial array X of the amplitude.
    assert np.array_equal(amplitude_array_x, amplitude_array_correct_x)

    # Create analysis dictionary for case on Y dimension dataframe.
    parameters_analysis_dict_y = {"x": 36}

    # Create new amplitude array for spatial dataframe of the Y dimension.
    amplitude_array_y = MacularAnalysisDataframes.peak_amplitude_analyzing.__wrapped__(
        multi_macular_dict_array_default["barSpeed30dps"].data["VSDI"],
        multi_macular_dict_array_default["barSpeed30dps"].index,
        parameters_analysis_dict_y)

    # Extract correct amplitude array Y to compare.
    amplitude_array_correct_y = [0.014, 0.012, 0.013, 0.014, 0.015, 0.026, 0.036, 0.038, 0.036, 0.027, 0.015, 0.014,
                                 0.013, 0.012, 0.014]

    # Verification of the validity of the spatial array Y of the amplitude.
    assert np.array_equal(amplitude_array_y, amplitude_array_correct_y)
