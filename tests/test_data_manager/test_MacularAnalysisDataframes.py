import os

from src.data_manager.MacularAnalysisDataframes import MacularAnalysisDataframes
from src.data_manager.MacularDictArray import MacularDictArray

# Get data for test from relative path.
path_data_test = os.path.normpath(f"{os.getcwd()}/../data_test/data_manager/")

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

macular_analysis_dataframes_barSpeed_head100 = MacularAnalysisDataframes(multiple_dicts_simulations_barSpeed_head100,
                                                                         multiple_dicts_preprocessings_barSpeed_head100,
                                                                         multiple_dicts_analysis_barSpeed_head100)

macular_analysis_dataframes_test = MacularAnalysisDataframes(multiple_dicts_simulations_barSpeed_head100,
                                                             multiple_dicts_preprocessings_barSpeed_head100,
                                                             multiple_dicts_analysis_barSpeed_head100)

multi_macular_dict_array_barSpeed_head100 = MacularDictArray.make_multiple_macular_dict_array(
    multiple_dicts_simulations_barSpeed_head100, multiple_dicts_preprocessings_barSpeed_head100)

multi_macular_dict_array_test = MacularDictArray.make_multiple_macular_dict_array(
    multiple_dicts_simulations_barSpeed_head100, multiple_dicts_preprocessings_barSpeed_head100)


def test_init():
    print()
    print(macular_analysis_dataframes_test.multiple_dicts_analysis)
    print(macular_analysis_dataframes_test.multiple_dicts_simulations)
    print(macular_analysis_dataframes_test.multiple_dicts_preprocessings)


def test_dict_paths_pyb_getter():
    assert macular_analysis_dataframes_barSpeed_head100.dict_paths_pyb == {
        "barSpeed6dps": "/home/jemonet/Documents/These/Code/macular_figures_lib/tests/data_test/data_manager/"
                        "RC_RM_dSGpCP0026_barSpeed6dps_head100_copy_0f.pyb",
        "barSpeed15dps": "/home/jemonet/Documents/These/Code/macular_figures_lib/tests/data_test/data_manager/"
                         "RC_RM_dSGpCP0028_barSpeed15dps_head100_copy_0f.pyb",
        "barSpeed30dps": "/home/jemonet/Documents/These/Code/macular_figures_lib/tests/data_test/data_manager/"
                         "RC_RM_dSGpCP0033_barSpeed30dps_head100_0f.pyb"}


def test_dict_paths_pyb_setter():
    macular_analysis_dataframes_test.dict_paths_pyb = {"barSpeed9dps": ""}
    assert macular_analysis_dataframes_test.dict_paths_pyb == {"barSpeed9dps": ""}


def test_dict_analysis_dataframes_getter():
    # TODO
    pass


def test_dict_analysis_dataframes_setter():
    # TODO
    pass


def test_multiple_dicts_analysis_getter():
    assert (macular_analysis_dataframes_barSpeed_head100.multiple_dicts_analysis
            == {})


def test_multiple_dicts_analysis_setter():
    macular_analysis_dataframes_test.multiple_dicts_analysis = {"test": {"setter1":True, "setter2":False},
                                                                "test2": {"setter3":False}, "test3":{}}
    assert macular_analysis_dataframes_test.multiple_dicts_analysis == {"test": {"setter1":True}}


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
