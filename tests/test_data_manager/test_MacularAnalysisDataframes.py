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

# Initialisation of the meta-analysis parameter dictionary of default multiple macular dict array.
dict_index_default = {condition: multi_macular_dict_array_default[condition].index
                      for condition in multi_macular_dict_array_default}

# Import the list of conditions/measures from the default multi macular dict array.
with open(f"{path_data_test}/MacularAnalysisDataframes/macular_analysis_dataframe_default_empty.pyb", "rb") as file:
    macular_analysis_dataframes_default_empty = pickle.load(file)
analysis_dataframes_levels = macular_analysis_dataframes_default_empty.analysis_dataframes_levels

# Define strings containing all conditions and measures.
all_conditions = analysis_dataframes_levels["conditions"]
all_measurements = analysis_dataframes_levels["measurements"]["barSpeed30dps"]

# Import default macular analysis dataframes of bar speed condition with activation time common group analysis.
with open(f"{path_data_test}/MacularAnalysisDataframes/activation_time_common_group_analysis.pyb", "rb") as file:
    activation_time_common_group_analysis = pickle.load(file)

# Import default macular analysis dataframes of bar speed condition with peak amplitude conditions common group analysis
with open(f"{path_data_test}/MacularAnalysisDataframes/peak_amplitude_conditions_common_group_analysis.pyb",
          "rb") as file:
    peak_amplitude_conditions_common_group_analysis = pickle.load(file)

# Import a default MacularAnalysisDataframes model with meta-analyses division already performed.
with open(f"{path_data_test}/MacularAnalysisDataframes/peak_amplitudes_meta_analysis_normalization.pyb",
          "rb") as file:
    peak_amplitude_meta_analysis_normalized = pickle.load(file)

# Import of a fully analyzed MacularAnalysisDataframes based on default multiple MacularDictArray.
with (open(f"{path_data_test}/MacularAnalysisDataframes/fully_meta_analyzed_macular_analysis_dataframe.pyb", "rb")
      as file):
    macular_analysis_dataframes_default = pickle.load(file)

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

multiple_dicts_simulations_default = {
    "global": {
        "n_cells_x": 83,
        "n_cells_y": 15,
        "dx": 0.225,
        "delta_t": 0.0167,
        "end": "max",
        "size_bar": 0.67,
        "axis": "horizontal"
    },
    "barSpeed30dps": {
        "path_pyb": f"../data_test/data_manager/MacularAnalysisDataframes/"
                    f"RC_RM_dSGpCP0033_barSpeed30dps_default_0f.pyb",
        "path_csv": f"../data_test/data_manager/RC_RM_dSGpCP0033_barSpeed30dps_0f.csv",
        "speed": 30
    },
    "barSpeed28,5dps": {
        "path_pyb": f"../data_test/data_manager/MacularAnalysisDataframes/"
                    f"RC_RM_dSGpCP0083_barSpeed28,5dps_default_0f.pyb",
        "path_csv": f"../data_test/data_manager/RC_RM_dSGpCP0083_barSpeed28,5dps_0f.csv",
        "speed": 28.5
    }
}

multiple_dicts_preprocessings_default = {
    "global": {
        "temporal_centering": True,
        "spatial_x_centering": True,
        "spatial_y_centering": True,
        "binning": 0.0016,
        "VSDI": True,
        "derivative": {"VSDI": 31, "FiringRate_GanglionGainControl": 31},
        "temporal_index_ms": 1000,
        "spatial_index_mm_retina": 0.3,
        "spatial_index_mm_cortex": 3,
        "edge": (5, 0)
    },
    "barSpeed28,5dps": {},
    "barSpeed30dps": {}
}

multiple_dicts_analysis_default = {
    "Conditions": {
        "sorting": "NameValueUnit",
        "peak_amplitude": [{"conditions": "all_conditions", "measurements": "FiringRate_GanglionGainControl:VSDI",
                            "params": {"x": 36, "y": 7, "flag": ""}}]
    },
    "X": {
        "activation_time": [{"conditions": "all_conditions", "measurements": "VSDI",
                             "params": {"threshold": 0.001, "index": "temporal_ms", "y": 7, "flag": "ms"}}],
        "latency": [{"conditions": "all_conditions", "measurements": "VSDI",
                     "params": {"threshold": 0.001, "index": "temporal_centered_ms", "y": 7,
                                "axis": "horizontal", "flag": "ms"}}],
        "time_to_peak": [{"conditions": "all_conditions", "measurements": "all_measurements",
                          "params": {"index": "temporal_ms", "y": 7, "flag": "ms"}}],
        "peak_delay": [{"conditions": "all_conditions", "measurements": "VSDI",
                        "params": {"index": "temporal_centered_ms", "y": 7, "axis": "horizontal", "flag": "ms"}}],
        "peak_amplitude": [{"conditions": "all_conditions", "measurements": "all_measurements",
                            "params": {"y": 7, "flag": ""}}],
    },
    "Y": {
        "activation_time": [{"conditions": "all_conditions", "measurements": "VSDI",
                             "params": {"threshold": 0.001, "index": "temporal_ms", "x": 36, "flag": "ms"}}],
        "time_to_peak": [{"conditions": "all_conditions", "measurements": "all_measurements",
                          "params": {"index": "temporal_ms", "x": 36, "flag": "ms"}}],
        "peak_amplitude": [{"conditions": "all_conditions", "measurements": "all_measurements",
                            "params": {"x": 36, "flag": ""}}],
    },
    "Time": {"test": "test"},
    "Multidimensional": {
        "multi_analysis": [{"conditions": "all_conditions", "measurements": "VSDI",
                            "params": {"main_dimension": "X", "secondary_dimension": "Conditions", "flag": ""}}]
    },
    "MetaAnalysis": {
        "peak_speed": [
            {"time_to_peak": {"dimensions": "X", "conditions": "all_conditions", "measurements": "VSDI",
                              "analyses": "time_to_peak", "flag": ""},
             "params": {"output": "horizontal_peak_speed", "index": "spatial_x"}}
        ],
        "normalization": [
            {"numerator": {"dimensions": "X:Y", "conditions": "all_conditions", "measurements": "VSDI",
                           "analyses": "peak_amplitude", "flag": ""},
             "denominator": {"dimensions": "X:Y", "conditions": "all_conditions", "measurements":
                 "FiringRate_GanglionGainControl", "analyses": "peak_amplitude", "flag": ""},
             "output": {"dimensions": "X:Y", "conditions": "all_conditions", "measurements": "VSDI",
                        "analyses": "spatial_peak_amplitudes_normalization"},
             "params": {"factor": 8}},

            {"numerator": {"dimensions": "X", "conditions": "all_conditions", "measurements": "VSDI",
                           "analyses": "peak_amplitude", "flag": ""},
             "denominator": {"dimensions": "Conditions", "conditions": "all_conditions", "measurements": "VSDI",
                             "analyses": "peak_amplitude", "flag": ""},
             "output": {"dimensions": "X", "conditions": "all_conditions", "measurements": "VSDI",
                        "analyses": "norm_peak_amplitudes_normalization"},
             "params": {"factor": 8}},

            {"numerator": {"dimensions": "Conditions", "conditions": "all_conditions", "measurements": "VSDI",
                           "analyses": "peak_amplitude", "flag": ""},
             "denominator": {"dimensions": "Conditions", "conditions": "all_conditions", "measurements": "VSDI",
                             "analyses": "peak_amplitude", "flag": ""},
             "output": {"dimensions": "Conditions", "conditions": "all_conditions", "measurements": "VSDI",
                        "analyses": "cond_peak_amplitudes_normalization"},
             "params": {"factor": 8}}
        ]
    }
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


def test_analysis_dataframes_levels_getter():
    analysis_dataframes_levels_correct = {
        "conditions": 'barSpeed15dps:barSpeed30dps:barSpeed6dps',
        "measurements": {'barSpeed6dps':
                             'BipolarResponse_BipolarGainControl:FiringRate_GanglionGainControl:'
                             'V_Amacrine:V_BipolarGainControl:V_GanglionGainControl:muVn_CorticalExcitatory:'
                             'muVn_CorticalInhibitory:v_e_CorticalExcitatory:v_i_CorticalInhibitory',
                         'barSpeed15dps': 'BipolarResponse_BipolarGainControl:FiringRate_GanglionGainControl:'
                                          'V_Amacrine:V_BipolarGainControl:V_GanglionGainControl:'
                                          'muVn_CorticalExcitatory:muVn_CorticalInhibitory:v_e_CorticalExcitatory:'
                                          'v_i_CorticalInhibitory',
                         'barSpeed30dps': 'BipolarResponse_BipolarGainControl:FiringRate_GanglionGainControl:'
                                          'V_Amacrine:V_BipolarGainControl:V_GanglionGainControl:'
                                          'muVn_CorticalExcitatory:muVn_CorticalInhibitory:v_e_CorticalExcitatory:'
                                          'v_i_CorticalInhibitory'},
        "dimensions": "Conditions:Time:X:Y",
        "analyses": {'Conditions': 'barSpeed (dps)',
                     'X': {'barSpeed6dps': '', 'barSpeed15dps': '', 'barSpeed30dps': ''},
                     'Y': {'barSpeed6dps': '', 'barSpeed15dps': '', 'barSpeed30dps': ''},
                     'Time': {'barSpeed6dps': '', 'barSpeed15dps': '', 'barSpeed30dps': ''}}
    }

    assert macular_analysis_dataframes_head100.analysis_dataframes_levels == analysis_dataframes_levels_correct


def test_analysis_dataframes_levels_setter():
    # Case of an attempt to modify analysis_dataframes_levels.
    try:
        macular_analysis_dataframes_test.analysis_dataframes_levels = {"test": "setter"}
        assert False
    except AttributeError:
        assert True

    # Verification that the value of dict_paths_pyb has not changed.
    assert (macular_analysis_dataframes_test.analysis_dataframes_levels
            == macular_analysis_dataframes_head100.analysis_dataframes_levels)


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
    # Creation of the model multiple analysis dictionaries.

    multiple_dicts_analysis_substitued_correct = {
        'Conditions': {'maximal_latency': [
            {"conditions": "barSpeed30dps", "measurements": "VSDI", "params": {"param1": 1}},
            {"conditions": "barSpeed30dps", "measurements": all_measurements, "params": {'param1': 2}},
            {"conditions": all_conditions, "measurements": "VSDI", "params": {'param1': 1}}
        ]

        },
        'X': {'activation_time': [
            {"conditions": "barSpeed30dps", "measurements": "VSDI", "params": {"param1": 1}},
            {"conditions": "barSpeed30dps", "measurements": all_measurements, "params": {'param1': 2}},
            {"conditions": "barSpeed28,5dps:barSpeed30dps", "measurements": "VSDI", "params": {'param1': 1}}
        ]
        }
    }

    # Initialisation of a multiple analysis dictionaries for a complex analysis.
    dict_analysis_test_default = {
        "Conditions": {"maximal_latency": [
            {"conditions": "barSpeed30dps", "measurements": "VSDI", "params": {"param1": 1}},
            {"conditions": "barSpeed30dps", "measurements": "all_measurements", "params": {'param1': 2}},
            {"conditions": "all_conditions", "measurements": "VSDI", "params": {'param1': 1}}
        ]
        },
        "X": {"activation_time": [
            {"conditions": "barSpeed30dps", "measurements": "VSDI", "params": {"param1": 1}},
            {"conditions": "barSpeed30dps", "measurements": "all_measurements", "params": {'param1': 2}},
            {"conditions": "all_conditions", "measurements": "VSDI", "params": {'param1': 1}}
        ]
        }
    }
    # Set default levels multiple dictionaries in test Macular analysis dataframes.
    macular_analysis_dataframes_test._analysis_dataframes_levels = analysis_dataframes_levels
    macular_analysis_dataframes_test._multiple_dicts_analysis = dict_analysis_test_default

    # Case with substitution of the getter.
    assert (macular_analysis_dataframes_test.multiple_dicts_analysis
            == multiple_dicts_analysis_substitued_correct)

    # Case without substitution of the getter.
    assert (macular_analysis_dataframes_test._multiple_dicts_analysis
            == dict_analysis_test_default)


def test_multiple_dicts_analysis_setter():
    macular_analysis_dataframes_test.multiple_dicts_analysis = {
        "dimension1": {"analysis1": [{"conditions": "condition1", "measurements": "measurement1"}, {}]},
        "dimension2": {"analysis2": []},
        "dimension3": {"analysis3": [{}]},
        "dimension4": {}}

    assert macular_analysis_dataframes_test.multiple_dicts_analysis == {"dimension1": {
        "analysis1": [{"conditions": "condition1", "measurements": "measurement1"}]}}


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
    macular_analysis_dataframes_test._multiple_dicts_analysis["Conditions"] = {}
    assert (macular_analysis_dataframes_test.dataframe_conditions_sorting() ==
            ["barSpeed6dps", "wAmaGang10Hz", "wAmaGang3,8Hz"])

    # Case of default sorting by alphabetical order.
    macular_analysis_dataframes_test._multiple_dicts_analysis["Conditions"]["sorting"] = False
    assert (macular_analysis_dataframes_test.dataframe_conditions_sorting() ==
            ["barSpeed6dps", "wAmaGang10Hz", "wAmaGang3,8Hz"])

    # Case of sorting based on a list defined in the multiple condition analysis dictionary.
    macular_analysis_dataframes_test._multiple_dicts_analysis["Conditions"]["sorting"] = ["wAmaBip10Hz",
                                                                                          "wAmaGang3,8Hz",
                                                                                          "barSpeed6dps"]
    assert (macular_analysis_dataframes_test.dataframe_conditions_sorting() ==
            ["wAmaBip10Hz", "wAmaGang3,8Hz", "barSpeed6dps"])

    # Case of sorting based on condition names and the ‘NameValueUnit’ format.
    macular_analysis_dataframes_test._multiple_dicts_analysis["Conditions"]["sorting"] = "NameValueUnit"
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
    assert levels_multiple_dictionaries["conditions"] == 'barSpeed15dps:barSpeed30dps:barSpeed6dps'

    # Verification of the character string dictionary for measurements for each condition.
    assert levels_multiple_dictionaries["measurements"] == dict_all_measurements


def test_get_levels_of_macular_analysis_dataframes():
    # Creation of an empty dictionary of macular analysis dataframe levels.
    dict_levels_macular_analysis_dataframes_correct = {
        'dimensions': 'Conditions:Time:X:Y',
        'analyses': {
            'Conditions': 'barSpeed (dps)',
            'X': {'barSpeed6dps': '', 'barSpeed15dps': '', 'barSpeed30dps': ''},
            'Y': {'barSpeed6dps': '', 'barSpeed15dps': '', 'barSpeed30dps': ''},
            'Time': {'barSpeed6dps': '', 'barSpeed15dps': '', 'barSpeed30dps': ''}}}

    assert (macular_analysis_dataframes_head100.get_levels_of_macular_analysis_dataframes() ==
            dict_levels_macular_analysis_dataframes_correct)

    # Import of a fully analyzed MacularAnalysisDataframes based on default multiple MacularDictArray.
    with (open(f"{path_data_test}/MacularAnalysisDataframes/fully_meta_analyzed_macular_analysis_dataframe.pyb", "rb")
          as file_test):
        macular_analysis_dataframes_default = pickle.load(file_test)

    all_analyses_X = ('activation_time_VSDI_ms:latency_VSDI_ms:peak_amplitude_BipolarResponse_BipolarGainControl:'
                      'peak_amplitude_FiringRate_GanglionGainControl:'
                      'peak_amplitude_FiringRate_GanglionGainControl_derivative:peak_amplitude_VSDI:'
                      'peak_amplitude_VSDI_derivative:peak_amplitude_V_Amacrine:'
                      'peak_amplitude_V_BipolarGainControl:peak_amplitude_V_GanglionGainControl:'
                      'peak_amplitude_muVn_CorticalExcitatory:peak_amplitude_muVn_CorticalInhibitory:'
                      'peak_amplitude_v_e_CorticalExcitatory:peak_amplitude_v_i_CorticalInhibitory:peak_delay_VSDI_ms:'
                      'time_to_peak_BipolarResponse_BipolarGainControl_ms:'
                      'time_to_peak_FiringRate_GanglionGainControl_derivative_ms:'
                      'time_to_peak_FiringRate_GanglionGainControl_ms:time_to_peak_VSDI_derivative_ms:time_to_peak_VSDI_ms:'
                      'time_to_peak_V_Amacrine_ms:'
                      'time_to_peak_V_BipolarGainControl_ms:time_to_peak_V_GanglionGainControl_ms:'
                      'time_to_peak_muVn_CorticalExcitatory_ms:'
                      'time_to_peak_muVn_CorticalInhibitory_ms:time_to_peak_v_e_CorticalExcitatory_ms:'
                      'time_to_peak_v_i_CorticalInhibitory_ms')
    all_analyses_Y = ('activation_time_VSDI_ms:peak_amplitude_BipolarResponse_BipolarGainControl:'
                      'peak_amplitude_FiringRate_GanglionGainControl:'
                      'peak_amplitude_FiringRate_GanglionGainControl_derivative:peak_amplitude_VSDI:'
                      'peak_amplitude_VSDI_derivative:peak_amplitude_V_Amacrine:peak_amplitude_V_BipolarGainControl:'
                      'peak_amplitude_V_GanglionGainControl:peak_amplitude_muVn_CorticalExcitatory:'
                      'peak_amplitude_muVn_CorticalInhibitory:peak_amplitude_v_e_CorticalExcitatory:'
                      'peak_amplitude_v_i_CorticalInhibitory:time_to_peak_BipolarResponse_BipolarGainControl_ms:'
                      'time_to_peak_FiringRate_GanglionGainControl_derivative_ms:'
                      'time_to_peak_FiringRate_GanglionGainControl_ms:time_to_peak_VSDI_derivative_ms:time_to_peak_VSDI_ms:'
                      'time_to_peak_V_Amacrine_ms:time_to_peak_V_BipolarGainControl_ms:'
                      'time_to_peak_V_GanglionGainControl_ms:time_to_peak_muVn_CorticalExcitatory_ms:'
                      'time_to_peak_muVn_CorticalInhibitory_ms:time_to_peak_v_e_CorticalExcitatory_ms:'
                      'time_to_peak_v_i_CorticalInhibitory_ms')

    # Creation of a dictionary of complex and filled macular analysis dataframe levels
    dict_levels_macular_analysis_dataframes_correct = {
        'dimensions': 'Conditions:Time:X:Y',
        'analyses': {
            'Conditions': 'barSpeed (dps):peak_amplitude_FiringRate_GanglionGainControl:peak_amplitude_VSDI',
            'X': {'barSpeed30dps': all_analyses_X,
                  'barSpeed28,5dps': all_analyses_X},
            'Y': {'barSpeed30dps': all_analyses_Y,
                  'barSpeed28,5dps': all_analyses_Y},
            'Time': {'barSpeed30dps': '', 'barSpeed28,5dps': ''}}}

    assert (macular_analysis_dataframes_default.get_levels_of_macular_analysis_dataframes() ==
            dict_levels_macular_analysis_dataframes_correct)


def test_substituting_all_alias_in_multiple_analysis_dictionaries():
    # Creation of the model multiple analysis dictionaries with 4 common group analysis for each dimension.
    multiple_dicts_analysis_substitued_correct = {
        "Conditions": {
            "maximal_latency": [
                {"conditions": "barSpeed30dps", "measurements": "VSDI", "params": {"param1": 1}},
                {"conditions": "barSpeed30dps", "measurements": all_measurements, "params": {"param1": 2}},
                {"conditions": "barSpeed30dps", "measurements": "BipolarResponse_BipolarGainControl",
                 "params": {"param1": 2}},
                {"conditions": all_conditions, "measurements": "VSDI", "params": {"param1": 1}}
            ]}
        ,
        "X": {
            "activation_time": [
                {"conditions": "barSpeed30dps", "measurements": "VSDI", "params": {"param1": 1}},
                {"conditions": "barSpeed30dps", "measurements": all_measurements, "params": {"param1": 2}},
                {"conditions": "barSpeed30dps", "measurements": "BipolarResponse_BipolarGainControl",
                 "params": {"param1": 2}},
                {"conditions": all_conditions, "measurements": "VSDI", "params": {"param1": 1}}
            ]
        },
        "Y": {
            "activation_time": [
                {"conditions": "barSpeed30dps", "measurements": "VSDI", "params": {"param1": 1}},
                {"conditions": "barSpeed30dps", "measurements": all_measurements, "params": {"param1": 2}},
                {"conditions": "barSpeed30dps", "measurements": "BipolarResponse_BipolarGainControl",
                 "params": {"param1": 2}},
                {"conditions": all_conditions, "measurements": "VSDI", "params": {"param1": 1}}
            ]
        },
        "Time": {
            "test_time": [
                {"conditions": "barSpeed30dps", "measurements": "VSDI", "params": {"param1": 1}},
                {"conditions": "barSpeed30dps", "measurements": all_measurements, "params": {"param1": 2}},
                {"conditions": "barSpeed30dps", "measurements": "BipolarResponse_BipolarGainControl",
                 "params": {"param1": 2}},
                {"conditions": all_conditions, "measurements": "VSDI", "params": {"param1": 1}}
            ]
        },
        "MetaAnalysis": {
            "normalization": [
                {"arg1": {"dimensions": "X:Y", "conditions": all_conditions, "measurements": "VSDI",
                          "analyses": "peak_amplitude", "flag": ""},
                 "arg2": {"dimensions": "X:Y", "conditions": all_conditions, "measurements": "VSDI",
                          "analyses": "peak_amplitude_normalization", "flag": ""},
                 "arg3": {"dimensions": "X:Y", "conditions": all_conditions, "measurements": "VSDI",
                          "analyses": "peak_amplitude_normalization", "flag": ""},
                 "params": {"params1": 8, "flag": ""}},

                {"arg1": {"dimensions": "Y", "conditions": all_conditions, "measurements": "VSDI",
                          "analyses": "peak_amplitude", "flag": "y6"},
                 "arg2": {"dimensions": "Conditions", "conditions": all_conditions, "measurements": "VSDI",
                          "analyses": "peak_amplitude", "flag": ""},
                 "arg3": {"dimensions": "Y", "conditions": all_conditions, "measurements": "VSDI",
                          "analyses": "peak_amplitude_normalization", "flag": "y6"},
                 "params": {"params1": 8, "flag": "y6"}}
            ]
        }
    }

    # Initialisation of a complex multiple analysis dictionaries with 4 common group analysis for each dimension.
    dict_analysis_test_default = {
        "Conditions": {
            "maximal_latency": [
                {"conditions": "barSpeed30dps", "measurements": "VSDI", "params": {"param1": 1}},
                {"conditions": "barSpeed30dps", "measurements": "all_measurements", "params": {"param1": 2}},
                {"conditions": "barSpeed30dps", "measurements": "BipolarResponse_BipolarGainControl",
                 "params": {"param1": 2}},
                {"conditions": "all_conditions", "measurements": "VSDI", "params": {"param1": 1}}
            ]}
        ,
        "X": {
            "activation_time": [
                {"conditions": "barSpeed30dps", "measurements": "VSDI", "params": {"param1": 1}},
                {"conditions": "barSpeed30dps", "measurements": "all_measurements", "params": {"param1": 2}},
                {"conditions": "barSpeed30dps", "measurements": "BipolarResponse_BipolarGainControl",
                 "params": {"param1": 2}},
                {"conditions": "all_conditions", "measurements": "VSDI", "params": {"param1": 1}}
            ]
        },
        "Y": {
            "activation_time": [
                {"conditions": "barSpeed30dps", "measurements": "VSDI", "params": {"param1": 1}},
                {"conditions": "barSpeed30dps", "measurements": "all_measurements", "params": {"param1": 2}},
                {"conditions": "barSpeed30dps", "measurements": "BipolarResponse_BipolarGainControl",
                 "params": {"param1": 2}},
                {"conditions": "all_conditions", "measurements": "VSDI", "params": {"param1": 1}}
            ]
        },
        "Time": {
            "test_time": [
                {"conditions": "barSpeed30dps", "measurements": "VSDI", "params": {"param1": 1}},
                {"conditions": "barSpeed30dps", "measurements": "all_measurements", "params": {"param1": 2}},
                {"conditions": "barSpeed30dps", "measurements": "BipolarResponse_BipolarGainControl",
                 "params": {"param1": 2}},
                {"conditions": "all_conditions", "measurements": "VSDI", "params": {"param1": 1}}
            ]
        },
        "MetaAnalysis": {
            "normalization": [
                {"arg1": {"dimensions": "X:Y", "conditions": "all_conditions", "measurements": "VSDI",
                          "analyses": "peak_amplitude", "flag": ""},
                 "arg2": {"dimensions": "X:Y", "conditions": "all_conditions", "measurements": "VSDI",
                          "analyses": "peak_amplitude_normalization", "flag": ""},
                 "arg3": {"dimensions": "X:Y", "conditions": "all_conditions", "measurements": "VSDI",
                          "analyses": "peak_amplitude_normalization", "flag": ""},
                 "params": {"params1": 8, "flag": ""}},

                {"arg1": {"dimensions": "Y", "conditions": "all_conditions", "measurements": "VSDI",
                          "analyses": "peak_amplitude", "flag": "y6"},
                 "arg2": {"dimensions": "Conditions", "conditions": "all_conditions", "measurements": "VSDI",
                          "analyses": "peak_amplitude", "flag": ""},
                 "arg3": {"dimensions": "Y", "conditions": "all_conditions", "measurements": "VSDI",
                          "analyses": "peak_amplitude_normalization", "flag": "y6"},
                 "params": {"params1": 8, "flag": "y6"}}
            ]
        }
    }

    # Set default levels multiple dictionaries in test Macular analysis dataframes.
    macular_analysis_dataframes_test._analysis_dataframes_levels = analysis_dataframes_levels

    # Execute the substituting_all_alias_in_multiple_analysis_dictionaries function to be tested.
    multiple_dicts_analysis_substitued = (macular_analysis_dataframes_test.
    substituting_all_alias_in_multiple_analysis_dictionaries(
        dict_analysis_test_default))

    assert multiple_dicts_analysis_substitued == multiple_dicts_analysis_substitued_correct


def test_substituting_all_alias_in_analysis_dictionary():
    # Creation of character strings with all conditions of the MacularAnalysisDataframe.
    all_conditions = ":".join(sorted([condition for condition in macular_analysis_dataframes_head100.dict_paths_pyb]))

    # Creation of character strings with all measurements from the MacularAnalysisDataframe, one with different size.
    all_measurements = {
        condition: ":".join(sorted([measure for measure in multi_macular_dict_array_head100[condition].data]))
        for condition in macular_analysis_dataframes_head100.dict_paths_pyb}
    all_measurements["barSpeed15dps"] = ":".join(all_measurements["barSpeed15dps"].split(":")[:5])

    # Creation of character strings with all dimensions from the MacularAnalysisDataframe.
    all_dimensions = ":".join(macular_analysis_dataframes_head100.dict_analysis_dataframes.keys())

    # Creation of character strings with all analyses from the MacularAnalysisDataframe, one with different size.
    all_analyses = {
        dimension: ":".join(sorted(list(macular_analysis_dataframes_head100.dict_analysis_dataframes[dimension].index)))
        if dimension == "Conditions"
        else {condition: ":".join(
            sorted(list(macular_analysis_dataframes_head100.dict_analysis_dataframes[dimension][condition].index)))
            for condition in macular_analysis_dataframes_head100.dict_analysis_dataframes[dimension]}
        for dimension in macular_analysis_dataframes_head100.dict_analysis_dataframes.keys()}
    all_analyses["X"]["barSpeed30dps"] = ":".join(all_analyses["X"]["barSpeed30dps"].split(":")[:5])

    # Creation of the structure containing the names of conditions and measures separated by ‘:’.
    macular_analysis_dataframes_test._analysis_dataframes_levels = {"conditions": all_conditions,
                                                                    "measurements": all_measurements,
                                                                    "dimensions": all_dimensions,
                                                                    "analyses": all_analyses}

    # Case of substitution with aliases ‘all_conditions’ and ‘all_measurements’
    test_dict_analysis = {"conditions": "all_conditions", "measurements": "all_measurements",
                          "params": {"threshold": 0.01, "y": 7, "index": "temporal_index"}, "flag": ""}
    macular_analysis_dataframes_test.substituting_all_alias_in_common_analysis_group_dictionary(test_dict_analysis)
    assert test_dict_analysis == {"conditions": all_conditions,
                                  "measurements": ":".join(all_measurements["barSpeed15dps"].split(":")[:5]),
                                  "params": {"threshold": 0.01, "y": 7, "index": "temporal_index"}, "flag": ""}

    # Case of substitution without aliases ‘all_conditions’ and ‘all_measurements’.
    test_dict_analysis = {"conditions": "barSpeed15dps:barSpeed30dps:barSpeed6dps", "measurements": "VSDI",
                          "params": {"threshold": 0.01, "y": 7, "index": "temporal_index"}, "flag": ""}
    macular_analysis_dataframes_test.substituting_all_alias_in_common_analysis_group_dictionary(test_dict_analysis)
    assert (test_dict_analysis ==
            {"conditions": "barSpeed15dps:barSpeed30dps:barSpeed6dps", "measurements": "VSDI",
             "params": {"threshold": 0.01, "y": 7, "index": "temporal_index"}, "flag": ""}
            )

    # Case of a substitution with alias ‘all_analyses’ for the ‘Conditions’ dimension of a MacularAnalysisDataframes.
    test_dict_analysis = {"conditions": "barSpeed15dps:barSpeed30dps:barSpeed6dps", "measurements": "VSDI",
                          "dimensions": "Conditions", "analyses": "all_analyses", "flag": ""}
    macular_analysis_dataframes_test.substituting_all_alias_in_common_analysis_group_dictionary(test_dict_analysis)
    assert (test_dict_analysis ==
            {"conditions": all_conditions, "measurements": "VSDI",
             "dimensions": "Conditions", "analyses": all_analyses["Conditions"], "flag": ""}
            )

    # Case of a substitution with alias ‘all_analyses’ for spatiotemporal dimensions of a MacularAnalysisDataframes.
    test_dict_analysis = {"conditions": "barSpeed30dps", "measurements": "VSDI", "dimensions": "X",
                          "analyses": "all_analyses", "flag": ""}
    macular_analysis_dataframes_test.substituting_all_alias_in_common_analysis_group_dictionary(test_dict_analysis)
    assert (test_dict_analysis ==
            {"conditions": "barSpeed30dps", "measurements": "VSDI",
             "dimensions": "X", "analyses": ":".join(all_analyses["X"]["barSpeed30dps"].split(":")[:5]),
             "flag": ""})





def test_make_spatial_dataframes_analysis():
    # Import an empty default macular analysis dataframes of bar speed condition for test.
    with open(f"{path_data_test}/MacularAnalysisDataframes/macular_analysis_dataframe_default_empty.pyb", "rb") as file:
        macular_analysis_dataframes_default_empty = pickle.load(file)

    # Import a default macular analysis dataframes of bar speed condition with the X dimension analysed.
    with open(f"{path_data_test}/MacularAnalysisDataframes/macular_analysis_dataframe_default_spatial_X_filled.pyb",
              "rb") as file:
        macular_analysis_dataframes_default_spatial_X_filled = pickle.load(file)

    # Set up multiple analysis dictionaries in the macular analysis test dataframes.
    macular_analysis_dataframes_default_empty._multiple_dicts_analysis["X"] = multiple_dicts_analysis_default["X"]

    # Use the make spatial dataframes analysis on the macular analysis test dataframes.
    macular_analysis_dataframes_default_empty.make_spatial_dataframes_analysis(
        "X", multi_macular_dict_array_default)

    # Verify that the conditions dataframe is correct.
    assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["Conditions"].equals(
        macular_analysis_dataframes_default_spatial_X_filled.dict_analysis_dataframes["Conditions"])

    # Verify that the X, Y, and T dataframes for each condition are equal.
    for condition in macular_analysis_dataframes_default_empty.dict_paths_pyb:
        assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["X"][condition].equals(
            macular_analysis_dataframes_default_spatial_X_filled.dict_analysis_dataframes["X"][condition])
        assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["Y"][condition].equals(
            macular_analysis_dataframes_default_spatial_X_filled.dict_analysis_dataframes["Y"][condition])
        assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["Time"][condition].equals(
            macular_analysis_dataframes_default_spatial_X_filled.dict_analysis_dataframes["Time"][condition])


def test_analysis():
    # Import a default macular analysis dataframes of bar speed condition with one complex analysis done.
    with open(f"{path_data_test}/MacularAnalysisDataframes/"
              f"macular_analysis_dataframe_default_complex_make_analysis.pyb", "rb") as file:
        macular_analysis_dataframes_default_complex_make_analysis = pickle.load(file)

    # Import an empty default macular analysis dataframes of bar speed condition.
    with open(f"{path_data_test}/MacularAnalysisDataframes/macular_analysis_dataframe_default_empty.pyb", "rb") as file:
        macular_analysis_dataframes_default_empty = pickle.load(file)

    # Initialisation of a dictionary for a complex analysis with 4 common analysis group in X and 1 in Conditions.
    dict_analysis_default_complex = {
        "X": {"activation_time":
            [
                {"conditions": all_conditions, "measurements": all_measurements,
                 "params": {"threshold": 0.01, "y": 7, "index": "temporal_ms", "flag": ""}},
                {"conditions": all_conditions, "measurements": "VSDI",
                 "params": {"threshold": 0.001, "y": 7, "index": "temporal_ms", "flag": ""}},
                {"conditions": "barSpeed28,5dps", "measurements": "BipolarResponse_BipolarGainControl",
                 "params": {"threshold": 0.005, "y": 7, "index": "temporal", "flag": ""}},
                {"conditions": "barSpeed30dps", "measurements": all_measurements,
                 "params": {"threshold": 0.001, "y": 7, "index": "temporal", "flag": ""}}
            ]
        },
        "Conditions": {"peak_amplitude":
            [
                {"conditions": all_conditions, "measurements": all_measurements,
                 "params": {"x": 36, "y": 7, "flag": ""}}
            ]
        }
    }

    # Set the complex analysis dictionary in the macular analysis dataframes attributes.
    macular_analysis_dataframes_default_empty._multiple_dicts_analysis["X"] = dict_analysis_default_complex["X"]
    macular_analysis_dataframes_default_empty._multiple_dicts_analysis["Conditions"] = (
        dict_analysis_default_complex)["Conditions"]

    # Use spatial and conditions analysis on empty macular analysis dataframes with the complex default dictionaries.
    MacularAnalysisDataframes.activation_time_analyzing(macular_analysis_dataframes_default_empty,
                                                        multi_macular_dict_array_default, "X",
                                                        "activation_time")
    MacularAnalysisDataframes.peak_amplitude_analyzing(macular_analysis_dataframes_default_empty,
                                                       multi_macular_dict_array_default, "Conditions",
                                                       "peak_amplitude")

    # Verify that the conditions dataframe is correct.
    assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["Conditions"].equals(
        macular_analysis_dataframes_default_complex_make_analysis.dict_analysis_dataframes["Conditions"])

    # Verify that the X, Y, and T dataframes for each condition are equal.
    for condition in macular_analysis_dataframes_default_empty.dict_paths_pyb:
        assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["X"][condition].equals(
            macular_analysis_dataframes_default_complex_make_analysis.dict_analysis_dataframes["X"][condition])
        assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["Y"][condition].equals(
            macular_analysis_dataframes_default_complex_make_analysis.dict_analysis_dataframes["Y"][condition])
        assert macular_analysis_dataframes_default_empty.dict_analysis_dataframes["Time"][condition].equals(
            macular_analysis_dataframes_default_complex_make_analysis.dict_analysis_dataframes["Time"][condition])


def test_common_analysis_group_parser():
    # Names of conditions in a common group analysis.
    grouped_conditions = "barSpeed6dps_ampGang5Hz:barSpeed30dps"

    # Names of measurements in a common group analysis.
    grouped_measurements = "FiringRate:VSDI"

    # Names of dimensions in an analysis group common to two conditions.
    grouped_dimensions = "X:Y"

    # Names of analyses in an analysis group that are common to two measurements.
    grouped_analyses = "activation_time:latency"

    # Pairs of conditions and measurements in a common analysis group with only conditions and measures.
    common_analysis_group_generator_correct = (analysis_pair for analysis_pair in
                                               [("barSpeed6dps_ampGang5Hz", "FiringRate"),
                                                ("barSpeed6dps_ampGang5Hz", "VSDI"),
                                                ("barSpeed30dps", "FiringRate"),
                                                ("barSpeed30dps", "VSDI")])

    common_analysis_group_generator = macular_analysis_dataframes_test.common_analysis_group_parser(
        [grouped_conditions, grouped_measurements])

    # Verification of each pair of conditions and measurements.
    for analysis_pair, analysis_pair_correct in zip(common_analysis_group_generator,
                                                    common_analysis_group_generator_correct):
        assert analysis_pair == analysis_pair_correct

    # Tuples of conditions and measurements in a common analysis group with dimensions, conditions, measures, analyses.
    common_analysis_group_generator_correct = (analysis_pair for analysis_pair in
                                               [("X", "barSpeed6dps_ampGang5Hz", "FiringRate", "activation_time"),
                                                ("X", "barSpeed6dps_ampGang5Hz", "FiringRate", "latency"),
                                                ("X", "barSpeed6dps_ampGang5Hz", "VSDI", "activation_time"),
                                                ("X", "barSpeed6dps_ampGang5Hz", "VSDI", "latency"),
                                                ("X", "barSpeed30dps", "FiringRate", "activation_time"),
                                                ("X", "barSpeed30dps", "FiringRate", "latency"),
                                                ("X", "barSpeed30dps", "VSDI", "activation_time"),
                                                ("X", "barSpeed30dps", "VSDI", "latency"),
                                                ("Y", "barSpeed6dps_ampGang5Hz", "FiringRate", "activation_time"),
                                                ("Y", "barSpeed6dps_ampGang5Hz", "FiringRate", "latency"),
                                                ("Y", "barSpeed6dps_ampGang5Hz", "VSDI", "activation_time"),
                                                ("Y", "barSpeed6dps_ampGang5Hz", "VSDI", "latency"),
                                                ("Y", "barSpeed30dps", "FiringRate", "activation_time"),
                                                ("Y", "barSpeed30dps", "FiringRate", "latency"),
                                                ("Y", "barSpeed30dps", "VSDI", "activation_time"),
                                                ("Y", "barSpeed30dps", "VSDI", "latency")
                                                ])

    common_analysis_group_generator = macular_analysis_dataframes_test.common_analysis_group_parser(
        [grouped_dimensions, grouped_conditions, grouped_measurements, grouped_analyses])

    for analysis_pair, analysis_pair_correct in zip(common_analysis_group_generator,
                                                    common_analysis_group_generator_correct):
        assert analysis_pair == analysis_pair_correct


def test_make_common_group_analysis():
    # Import an empty default macular analysis dataframes of bar speed condition with spatial X dataframe analyzed.
    with open(f"{path_data_test}/MacularAnalysisDataframes/macular_analysis_dataframe_default_empty.pyb", "rb") as file:
        macular_analysis_dataframes_spatial_analysis = pickle.load(file)

    # Import an empty default macular analysis dataframes of bar speed condition with condition dataframe analyzed.
    with open(f"{path_data_test}/MacularAnalysisDataframes/macular_analysis_dataframe_default_empty.pyb", "rb") as file:
        macular_analysis_dataframes_conditions_analysis = pickle.load(file)

    # Setup generator and parameters for spatial common group analysis (activation time).
    common_analysis_group_generator = (analysis_pair for analysis_pair in
                                       [("barSpeed28,5dps", "FiringRate_GanglionGainControl"),
                                        ("barSpeed30dps", "VSDI")])
    parameters_analysis_dict_spatial_analysis = {"threshold": 0.001, "y": 7, "index": "temporal_ms",
                                                 "flag": "threshold0,001_y7"}

    # Make spatial common group analysis (activation time).
    macular_analysis_dataframes_spatial_analysis.make_common_group_analysis(
        MacularAnalysisDataframes.activation_time_analyzing.__wrapped__,
        multi_macular_dict_array_default, common_analysis_group_generator,
        "X", "activation_time", parameters_analysis_dict_spatial_analysis)

    # Verify that the conditions dataframe is correct.
    assert macular_analysis_dataframes_spatial_analysis.dict_analysis_dataframes["Conditions"].equals(
        activation_time_common_group_analysis.dict_analysis_dataframes["Conditions"])

    # Verify that the X, Y, and T dataframes for each condition are equal.
    for condition in macular_analysis_dataframes_spatial_analysis.dict_paths_pyb:
        assert macular_analysis_dataframes_spatial_analysis.dict_analysis_dataframes["X"][condition].equals(
            activation_time_common_group_analysis.dict_analysis_dataframes["X"][condition])
        assert macular_analysis_dataframes_spatial_analysis.dict_analysis_dataframes["Y"][condition].equals(
            activation_time_common_group_analysis.dict_analysis_dataframes["Y"][condition])
        assert macular_analysis_dataframes_spatial_analysis.dict_analysis_dataframes["Time"][condition].equals(
            activation_time_common_group_analysis.dict_analysis_dataframes["Time"][condition])

    # Setup generator and parameters for conditions common group analysis (peak amplitude).
    common_analysis_group_generator = (analysis_pair for analysis_pair in
                                       [("barSpeed28,5dps", "FiringRate_GanglionGainControl"),
                                        ("barSpeed28,5dps", "VSDI"),
                                        ("barSpeed30dps", "VSDI")])
    parameters_analysis_dict_conditions_analysis = {"x": 36, "y": 7, "flag": "x36_y7"}

    # Make conditions common group analysis (peak amplitude).
    macular_analysis_dataframes_conditions_analysis.make_common_group_analysis(
        MacularAnalysisDataframes.peak_amplitude_analyzing.__wrapped__,
        multi_macular_dict_array_default, common_analysis_group_generator,
        "Conditions", "peak_amplitude", parameters_analysis_dict_conditions_analysis)

    # Verify that the conditions dataframe is correct.
    assert macular_analysis_dataframes_conditions_analysis.dict_analysis_dataframes["Conditions"].equals(
        peak_amplitude_conditions_common_group_analysis.dict_analysis_dataframes["Conditions"])

    # Verify that the X, Y, and T dataframes for each condition are equal.
    for condition in macular_analysis_dataframes_conditions_analysis.dict_paths_pyb:
        assert macular_analysis_dataframes_conditions_analysis.dict_analysis_dataframes["X"][condition].equals(
            peak_amplitude_conditions_common_group_analysis.dict_analysis_dataframes["X"][condition])
        assert macular_analysis_dataframes_conditions_analysis.dict_analysis_dataframes["Y"][condition].equals(
            peak_amplitude_conditions_common_group_analysis.dict_analysis_dataframes["Y"][condition])
        assert macular_analysis_dataframes_conditions_analysis.dict_analysis_dataframes["Time"][condition].equals(
            peak_amplitude_conditions_common_group_analysis.dict_analysis_dataframes["Time"][condition])


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

    # Create analysis dictionary for case on Conditions dimension dataframe.
    parameters_analysis_dict_conditions = {"x": 36, "y": 7}

    # Create new amplitude array for spatial dataframe of the Y dimension.
    amplitude_conditions = MacularAnalysisDataframes.peak_amplitude_analyzing.__wrapped__(
        multi_macular_dict_array_default["barSpeed30dps"].data["VSDI"],
        multi_macular_dict_array_default["barSpeed30dps"].index,
        parameters_analysis_dict_conditions)

    # Verification of the validity of the value of the amplitude.
    assert amplitude_conditions == 0.038


def test_meta_analysis():
    # Import of an analyzed default MacularAnalysisDataframes to test meta-analysis.
    with (open(f"{path_data_test}/MacularAnalysisDataframes/fully_analyzed_macular_analysis_dataframe.pyb", "rb")
          as file_test):
        macular_analysis_dataframes_default_test = pickle.load(file_test)

    # Use division meta-analysis on default macular analysis dataframes with the default analyses dictionaries.
    MacularAnalysisDataframes.normalization_analyzing(macular_analysis_dataframes_default_test, "normalization",
                                                      dict_index_default)

    # Verify that the conditions dataframe is correct.
    assert macular_analysis_dataframes_default_test.dict_analysis_dataframes["Conditions"].equals(
        peak_amplitude_meta_analysis_normalized.dict_analysis_dataframes["Conditions"])

    # Verify that the X, Y, and T dataframes for each condition are equal.
    for condition in macular_analysis_dataframes_default_test.dict_paths_pyb:
        assert macular_analysis_dataframes_default_test.dict_analysis_dataframes["X"][condition].equals(
            peak_amplitude_meta_analysis_normalized.dict_analysis_dataframes["X"][condition])
        assert macular_analysis_dataframes_default_test.dict_analysis_dataframes["Y"][condition].equals(
            peak_amplitude_meta_analysis_normalized.dict_analysis_dataframes["Y"][condition])
        assert macular_analysis_dataframes_default_test.dict_analysis_dataframes["Time"][condition].equals(
            peak_amplitude_meta_analysis_normalized.dict_analysis_dataframes["Time"][condition])


def test_multiple_common_meta_analysis_group_parser():
    # Initialisation of a decondensed dictionary model of common meta-analysis groups.
    parsed_dictionaries_correct = [
        {"arg1": [("X", "barSpeed28,5dps", "VSDI", "peak_amplitude", "internal_flag1"),
                  ("X", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag1"),
                  ("Y", "barSpeed28,5dps", "VSDI", "peak_amplitude", "internal_flag1"),
                  ("Y", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag1")],
         "params": {"factor": 8, "flag": "external_flag1"}},

        {"arg1": [("X", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag2"),
                  ("Y", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag2")],
         "output1": [("X", "barSpeed30dps", "VSDI", "peak_amplitude"),
                     ("Y", "barSpeed30dps", "VSDI", "peak_amplitude")],
         "params": {"factor": 7}},

        {"arg1": [("Conditions", "barSpeed28,5dps", "VSDI", "latency", "internal_flag3"),
                  ("Conditions", "barSpeed30dps", "VSDI", "latency", "internal_flag3")],
         "params": {"factor": 8, "output": "peak_amplitude_mean"}},
    ]

    # Initialisation of a condensed dictionary of common meta-analysis groups.
    meta_analysis_dictionaries = [
        {"arg1": {"dimensions": "X:Y", "conditions": "barSpeed28,5dps:barSpeed30dps", "measurements": "VSDI",
                  "analyses": "peak_amplitude", "flag": "internal_flag1"},
         "params": {"factor": 8, "flag": "external_flag1"}},
        {"arg1": {"dimensions": "X:Y", "conditions": "barSpeed30dps", "measurements": "VSDI",
                  "analyses": "peak_amplitude", "flag": "internal_flag2"},
         "output1": {"dimensions": "X:Y", "conditions": "barSpeed30dps", "measurements": "VSDI",
                     "analyses": "peak_amplitude"},
         "params": {"factor": 7}},
        {"arg1": {"dimensions": "Conditions", "conditions": "barSpeed28,5dps:barSpeed30dps", "measurements": "VSDI",
                  "analyses": "latency", "flag": "internal_flag3"},
         "params": {"factor": 8, "output": "peak_amplitude_mean"}}
    ]

    # Parsing the dictionary of common meta-analysis groups.
    parsed_dictionaries = MacularAnalysisDataframes.multiple_common_meta_analysis_group_parser(
        meta_analysis_dictionaries)

    assert parsed_dictionaries == parsed_dictionaries_correct


def test_common_meta_analysis_group_parser():
    # Initialisation of a complex dictionary model of common meta-analysis groups.
    parsed_dictionary_correct = {
        "arg1": [
            ("X", "barSpeed28,5dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("X", "barSpeed28,5dps", "VSDI", "latency", "internal_flag"),
            ("X", "barSpeed28,5dps", "FiringRate", "peak_amplitude", "internal_flag"),
            ("X", "barSpeed28,5dps", "FiringRate", "latency", "internal_flag"),
            ("X", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("X", "barSpeed30dps", "VSDI", "latency", "internal_flag"),
            ("X", "barSpeed30dps", "FiringRate", "peak_amplitude", "internal_flag"),
            ("X", "barSpeed30dps", "FiringRate", "latency", "internal_flag"),

            ("Y", "barSpeed28,5dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Y", "barSpeed28,5dps", "VSDI", "latency", "internal_flag"),
            ("Y", "barSpeed28,5dps", "FiringRate", "peak_amplitude", "internal_flag"),
            ("Y", "barSpeed28,5dps", "FiringRate", "latency", "internal_flag"),
            ("Y", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Y", "barSpeed30dps", "VSDI", "latency", "internal_flag"),
            ("Y", "barSpeed30dps", "FiringRate", "peak_amplitude", "internal_flag"),
            ("Y", "barSpeed30dps", "FiringRate", "latency", "internal_flag")
        ],
        "arg2": [("Conditions", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag")] * 16,
        # TODO Régler le problème d'ordre des conditions dans l'arg3 comparé à l'arg1.
        "arg3": [
            ("Conditions", "barSpeed28,5dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed28,5dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed28,5dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed28,5dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed28,5dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed28,5dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed28,5dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed28,5dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag"),
            ("Conditions", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag")
        ],
        "output": [
            ("X", "barSpeed28,5dps", "VSDI", "peak_amplitude_latency_normalization"),
            ("X", "barSpeed28,5dps", "VSDI", "peak_amplitude_latency_normalization"),
            ("X", "barSpeed28,5dps", "FiringRate", "peak_amplitude_latency_normalization"),
            ("X", "barSpeed28,5dps", "FiringRate", "peak_amplitude_latency_normalization"),
            ("X", "barSpeed30dps", "VSDI", "peak_amplitude_latency_normalization"),
            ("X", "barSpeed30dps", "VSDI", "peak_amplitude_latency_normalization"),
            ("X", "barSpeed30dps", "FiringRate", "peak_amplitude_latency_normalization"),
            ("X", "barSpeed30dps", "FiringRate", "peak_amplitude_latency_normalization"),

            ("Y", "barSpeed28,5dps", "VSDI", "peak_amplitude_latency_normalization"),
            ("Y", "barSpeed28,5dps", "VSDI", "peak_amplitude_latency_normalization"),
            ("Y", "barSpeed28,5dps", "FiringRate", "peak_amplitude_latency_normalization"),
            ("Y", "barSpeed28,5dps", "FiringRate", "peak_amplitude_latency_normalization"),
            ("Y", "barSpeed30dps", "VSDI", "peak_amplitude_latency_normalization"),
            ("Y", "barSpeed30dps", "VSDI", "peak_amplitude_latency_normalization"),
            ("Y", "barSpeed30dps", "FiringRate", "peak_amplitude_latency_normalization"),
            ("Y", "barSpeed30dps", "FiringRate", "peak_amplitude_latency_normalization")
        ],
        "params": {"factor": 8}}

    # Initialisation of an unparsed complex dictionary of common meta-analysis groups.
    common_meta_analysis_group_dictionary = {
        "arg1": {"dimensions": "X:Y", "conditions": "barSpeed28,5dps:barSpeed30dps",
                 "measurements": "VSDI:FiringRate",
                 "analyses": "peak_amplitude:latency", "flag": "internal_flag"},
        "arg2": {"dimensions": "Conditions", "conditions": "barSpeed30dps", "measurements": "VSDI",
                 "analyses": "peak_amplitude", "flag": "internal_flag"},
        "arg3": {"dimensions": "Conditions", "conditions": "barSpeed28,5dps:barSpeed30dps", "measurements": "VSDI",
                 "analyses": "peak_amplitude", "flag": "internal_flag"},
        "output": {"dimensions": "X:Y", "conditions": "barSpeed28,5dps:barSpeed30dps",
                   "measurements": "VSDI:FiringRate",
                   "analyses": "peak_amplitude_latency_normalization"},
        "params": {"factor": 8}}

    # Parsing the dictionary of common meta-analysis groups.
    parsed_dictionary = MacularAnalysisDataframes.common_meta_analysis_group_parser(
        common_meta_analysis_group_dictionary)

    assert parsed_dictionary == parsed_dictionary_correct


def test_resizing_common_analysis_group_levels():
    # Initialisation of the list to be reproduced, consisting of an element repeated 4 times.
    common_analysis_group_levels_list_correct = [("dimension1", "conditions1", "measurement1", "analysis1"),
                                                 ("dimension1", "conditions1", "measurement1", "analysis1"),
                                                 ("dimension1", "conditions1", "measurement1", "analysis1"),
                                                 ("dimension1", "conditions1", "measurement1", "analysis1")]

    # Initialisation of the single list of elements.
    common_analysis_group_levels_list = [("dimension1", "conditions1", "measurement1", "analysis1")]

    # Case of a list with a single element to be repeated 4 times.
    assert MacularAnalysisDataframes.resizing_common_analysis_group_levels(
        common_analysis_group_levels_list, 4) == common_analysis_group_levels_list_correct

    # Initialisation of the list to be reproduced, consisting of 3 elements repeated twice.
    common_analysis_group_levels_list_correct = [("dimension1", "conditions1", "measurement1", "analysis1"),
                                                 ("dimension1", "conditions1", "measurement1", "analysis1"),
                                                 ("dimension2", "conditions2", "measurement2", "analysis2"),
                                                 ("dimension2", "conditions2", "measurement2", "analysis2"),
                                                 ("dimension3", "conditions3", "measurement3", "analysis3"),
                                                 ("dimension3", "conditions3", "measurement3", "analysis3")
                                                 ]

    # Initialisation of the list of 3 elements.
    common_analysis_group_levels_list = [("dimension1", "conditions1", "measurement1", "analysis1"),
                                         ("dimension2", "conditions2", "measurement2", "analysis2"),
                                         ("dimension3", "conditions3", "measurement3", "analysis3")]

    # Case of a list of 3 elements to be repeated twice.
    assert MacularAnalysisDataframes.resizing_common_analysis_group_levels(
        common_analysis_group_levels_list, 6) == common_analysis_group_levels_list_correct


def test_make_common_group_meta_analysis():
    # Import of an analyzed default MacularAnalysisDataframes to test meta-analysis.
    with (open(f"{path_data_test}/MacularAnalysisDataframes/fully_analyzed_macular_analysis_dataframe.pyb", "rb")
          as file_test):
        macular_analysis_dataframes_default_test = pickle.load(file_test)

    # Getting a list of dictionaries of common meta-analysis groups that's not condensed for tests.
    common_meta_analysis_group_dictionaries = MacularAnalysisDataframes.multiple_common_meta_analysis_group_parser(
        macular_analysis_dataframes_default_test.multiple_dicts_analysis["MetaAnalysis"]["normalization"])

    # Execution of a group of common meta-analyses based only on spatial dataframes.
    macular_analysis_dataframes_default_test.make_common_group_meta_analysis(
        MacularAnalysisDataframes.normalization_analyzing.__wrapped__, common_meta_analysis_group_dictionaries[0],
        "normalization", dict_index_default)

    # Execution of a group of common meta-analyses based on spatial and condition dataframes.
    macular_analysis_dataframes_default_test.make_common_group_meta_analysis(
        MacularAnalysisDataframes.normalization_analyzing.__wrapped__, common_meta_analysis_group_dictionaries[1],
        "normalization", dict_index_default)

    # Execution of a group of common meta-analyses based only on condition dataframes.
    macular_analysis_dataframes_default_test.make_common_group_meta_analysis(
        MacularAnalysisDataframes.normalization_analyzing.__wrapped__, common_meta_analysis_group_dictionaries[2],
        "normalization", dict_index_default)

    # Verify that the conditions dataframe is correct.
    assert macular_analysis_dataframes_default_test.dict_analysis_dataframes["Conditions"].equals(
        peak_amplitude_meta_analysis_normalized.dict_analysis_dataframes["Conditions"])

    # Verify that the X, Y, and T dataframes for each condition are equal.
    for condition in macular_analysis_dataframes_default_test.dict_paths_pyb:
        assert macular_analysis_dataframes_default_test.dict_analysis_dataframes["X"][condition].equals(
            peak_amplitude_meta_analysis_normalized.dict_analysis_dataframes["X"][condition])
        assert macular_analysis_dataframes_default_test.dict_analysis_dataframes["Y"][condition].equals(
            peak_amplitude_meta_analysis_normalized.dict_analysis_dataframes["Y"][condition])
        assert macular_analysis_dataframes_default_test.dict_analysis_dataframes["Time"][condition].equals(
            peak_amplitude_meta_analysis_normalized.dict_analysis_dataframes["Time"][condition])


def test_extract_all_analysis_array_from_dataframes():
    # Initialisation of a meta-analysis dictionary after extraction of the arrays from each analysis.
    correct_meta_analysis_dictionary = {
        "numerator": np.array([0.041, 0.045, 0.041, 0.045, 0.041, 0.044, 0.041, 0.043, 0.041,
                               0.041, 0.042, 0.041, 0.042, 0.04, 0.043, 0.04, 0.043, 0.039,
                               0.043, 0.039, 0.043, 0.04, 0.043, 0.04, 0.042, 0.04, 0.041, 0.041,
                               0.041, 0.042, 0.04, 0.042, 0.04, 0.042, 0.039, 0.042, 0.039,
                               0.043, 0.039, 0.043, 0.04, 0.042, 0.04, 0.041, 0.041, 0.041,
                               0.041, 0.04, 0.042, 0.04, 0.042, 0.039, 0.043, 0.04, 0.043, 0.04,
                               0.043, 0.04, 0.043, 0.04, 0.042, 0.041, 0.042, 0.042, 0.042,
                               0.043, 0.042, 0.044, 0.042, 0.045, 0.043, 0.047, 0.043]),
        "denominator": 0.039,
        "output": ("X", "barSpeed28,5dps", "VSDI", "peak_amplitude")
    }

    # Initialisation of a meta-analysis dictionary before extracting the arrays from each analysis.
    meta_analysis_dictionary = {
        "numerator": ("X", "barSpeed28,5dps", "VSDI", "peak_amplitude", ""),
        "denominator": ("Conditions", "barSpeed28,5dps", "VSDI", "peak_amplitude", ""),
        "output": ("X", "barSpeed28,5dps", "VSDI", "peak_amplitude")
    }

    # Extraction of arrays from each analysis.
    MacularAnalysisDataframes.extract_all_analysis_array_from_dataframes(macular_analysis_dataframes_default,
                                                                         meta_analysis_dictionary)
    # Verification that the two dictionaries are equal.
    for arguments in meta_analysis_dictionary:
        if "output" in arguments:
            assert meta_analysis_dictionary[arguments] == correct_meta_analysis_dictionary[arguments]
        else:
            assert np.array_equal(meta_analysis_dictionary[arguments], correct_meta_analysis_dictionary[arguments])


def test_extract_one_analysis_array_from_dataframes():
    # Define level names for an analysis located in the spatial dataframe X.
    meta_analysis_dictionary_x = ("X", "barSpeed28,5dps", "VSDI", "peak_amplitude", "")

    # Extraction of the analysis associated with the previously defined levels.
    analysis_array_x = MacularAnalysisDataframes.extract_one_analysis_array_from_dataframes(
        macular_analysis_dataframes_default, meta_analysis_dictionary_x)

    # Case of extracting the array of values from an analysis from the spatial dataframe X of a given condition.
    assert np.array_equal(analysis_array_x, np.array([0.041, 0.045, 0.041, 0.045, 0.041, 0.044, 0.041, 0.043, 0.041,
                                                      0.041, 0.042, 0.041, 0.042, 0.04, 0.043, 0.04, 0.043, 0.039,
                                                      0.043, 0.039, 0.043, 0.04, 0.043, 0.04, 0.042, 0.04, 0.041, 0.041,
                                                      0.041, 0.042, 0.04, 0.042, 0.04, 0.042, 0.039, 0.042, 0.039,
                                                      0.043, 0.039, 0.043, 0.04, 0.042, 0.04, 0.041, 0.041, 0.041,
                                                      0.041, 0.04, 0.042, 0.04, 0.042, 0.039, 0.043, 0.04, 0.043, 0.04,
                                                      0.043, 0.04, 0.043, 0.04, 0.042, 0.041, 0.042, 0.042, 0.042,
                                                      0.043, 0.042, 0.044, 0.042, 0.045, 0.043, 0.047, 0.043]))

    # Definition of level names for an analysis located in the conditions dataframe.
    meta_analysis_dictionary_conditions = ("Conditions", "barSpeed28,5dps", "VSDI", "peak_amplitude", "test")

    # Adding a new value line to the conditions dataframe.
    macular_analysis_dataframes_default.dict_analysis_dataframes["Conditions"].loc[
        "peak_amplitude_VSDI_test"] = (1.8, 3.2)

    # Extraction of the analysis associated with the previously defined levels.
    analysis_array_conditions = MacularAnalysisDataframes.extract_one_analysis_array_from_dataframes(
        macular_analysis_dataframes_default, meta_analysis_dictionary_conditions)

    # Case of extracting the value of an analysis for a condition from the conditions dataframe.
    assert analysis_array_conditions == 1.8
    # Remove the new value row in the conditions dataframe.
    macular_analysis_dataframes_default.dict_analysis_dataframes["Conditions"].drop("peak_amplitude_VSDI_test",
                                                                                    inplace=True)


def test_make_meta_analysis_outputs():
    # Defining a meta-analysis name for tests.
    meta_analysis_name = "normalization"

    # Definition of a dictionary of argument levels without output for the default case.
    meta_analysis_dictionary = {
        "numerator": ("X", "barSpeed28,5dps", "VSDI", "latency", ""),
        "denominator": ("X", "barSpeed28,5dps", "VSDI", "peak_amplitude", "")}

    # Definition of a meta-analysis parameter dictionary for tests.
    parameters_meta_analysis_dict = {"factor": 8, "flag": "external_flag"}

    # Case of default formatting.
    dataframe_name_dict = MacularAnalysisDataframes.make_meta_analysis_outputs(meta_analysis_name,
                                                                               meta_analysis_dictionary,
                                                                               parameters_meta_analysis_dict)
    assert dataframe_name_dict["output"] == {"name": "peak_amplitude_VSDI_latency_VSDI_normalization_external_flag"}

    # Added an output in the meta-analysis parameters.
    parameters_meta_analysis_dict["output_test_1"] = "VSDI_latency_peak_amplitude_normalization"

    # Case of formatting from the output in the meta-analysis parameters.
    dataframe_name_dict = MacularAnalysisDataframes.make_meta_analysis_outputs(meta_analysis_name,
                                                                               meta_analysis_dictionary,
                                                                               parameters_meta_analysis_dict)
    assert dataframe_name_dict["output_test_1"] == {"name": "VSDI_latency_peak_amplitude_normalization"}

    # Added two outputs to the meta-analysis function arguments.
    meta_analysis_dictionary["output_test_2"] = (
        "X", "barSpeed28,5dps", "VSDI", "latency_peak_amplitude_normalization", "")
    meta_analysis_dictionary["output_test_3"] = ("X", "barSpeed28,5dps", "VSDI", "normalization", "")

    # Case of formatting from the outputs in the arguments of the meta-analysis function.
    dataframe_name_dict = MacularAnalysisDataframes.make_meta_analysis_outputs(meta_analysis_name,
                                                                               meta_analysis_dictionary,
                                                                               parameters_meta_analysis_dict)
    assert dataframe_name_dict["output_test_2"] == {"dimension": "X", "condition": "barSpeed28,5dps",
                                                    "name": "latency_peak_amplitude_normalization"}
    assert dataframe_name_dict["output_test_3"] == {"dimension": "X", "condition": "barSpeed28,5dps",
                                                    "name": "normalization"}


def test_add_array_line_to_dataframes():
    # Adding an array to the spatial dataframe X of condition barSpeed30dps.
    MacularAnalysisDataframes.add_array_line_to_dataframes(macular_analysis_dataframes_head100, "X",
                                                           "barSpeed30dps", "test_X",
                                                           np.array([i for i in range(83)]))
    assert np.array_equal(macular_analysis_dataframes_head100.dict_analysis_dataframes["X"]["barSpeed30dps"].loc[
                              "test_X"], np.array([i for i in range(83)]))

    # Add an array to the conditions dataframe.
    MacularAnalysisDataframes.add_array_line_to_dataframes(macular_analysis_dataframes_head100, "Conditions",
                                                           "all", "test_conditions",
                                                           np.array([0, 4, 5]))
    assert np.array_equal(macular_analysis_dataframes_head100.dict_analysis_dataframes["Conditions"].loc[
                              "test_conditions"], np.array([0, 4, 5]))

    # Adds a unique value to a condition in the conditions dataframe.
    MacularAnalysisDataframes.add_array_line_to_dataframes(macular_analysis_dataframes_head100, "Conditions",
                                                           "barSpeed15dps", "test_conditions_2",
                                                           10)
    assert macular_analysis_dataframes_head100.dict_analysis_dataframes["Conditions"].loc[
               "test_conditions_2"][1] == 10
    assert np.isnan(macular_analysis_dataframes_head100.dict_analysis_dataframes["Conditions"].loc[
                        "test_conditions_2"][0])
    assert np.isnan(macular_analysis_dataframes_head100.dict_analysis_dataframes["Conditions"].loc[
                        "test_conditions_2"][2])


def test_normalization_analyzing():
    # Import of an analyzed default MacularAnalysisDataframes to test meta-analysis.
    with (open(f"{path_data_test}/MacularAnalysisDataframes/fully_analyzed_macular_analysis_dataframe.pyb", "rb")
          as file_test):
        macular_analysis_dataframes_default_test = pickle.load(file_test)

    # Initialisation of the meta-analysis parameter dictionary for tests.
    parameters_meta_analysis_dict = {"factor": 8}

    # Initialisation of a dictionary of argument associated to 2 arrays of values in numerator and denominator.
    meta_analysis_dictionary = {
        "numerator": ("X", "barSpeed28,5dps", "VSDI", "latency", "ms"),
        "denominator": ("X", "barSpeed28,5dps", "VSDI", "peak_amplitude", ""),
        "output": {"dimension": "X", "condition": "barSpeed28,5dps", "name": "latency_ms_peak_amplitude_normalization"}}

    # Performing division meta-analysis with arrays as numerators and denominators.
    MacularAnalysisDataframes.normalization_analyzing.__wrapped__(macular_analysis_dataframes_default_test,
                                                                  meta_analysis_dictionary, dict_index_default,
                                                                  parameters_meta_analysis_dict)

    # Getting the array calculated in the division meta-analysis.
    output_array = macular_analysis_dataframes_default_test.dict_analysis_dataframes["X"]["barSpeed28,5dps"].loc[
        "latency_ms_peak_amplitude_normalization"].values

    # Manual calculation of the expected array values of the meta-analysis division.
    denominator_array = macular_analysis_dataframes_default_test.dict_analysis_dataframes["X"]["barSpeed28,5dps"].loc[
        "peak_amplitude_VSDI"].values
    numerator_array = macular_analysis_dataframes_default_test.dict_analysis_dataframes["X"]["barSpeed28,5dps"].loc[
        "latency_VSDI_ms"].values
    output_array_expected = numerator_array / denominator_array * 8

    # Case of division meta-analysis with arrays as numerator and denominator values.
    assert np.array_equal(output_array, output_array_expected)

    # Initialisation of a dictionary of argument with a value in the denominator and an array in the numerator.
    meta_analysis_dictionary = {
        "numerator": ("X", "barSpeed28,5dps", "VSDI", "latency", "ms"),
        "denominator": ("Conditions", "barSpeed28,5dps", "VSDI", "peak_amplitude", ""),
        "output": {"dimension": "X", "condition": "barSpeed28,5dps",
                   "name": "cond_x_latency_ms_peak_amplitude_normalization"}}

    # Performs the division meta-analysis with a value and an array of values in the arguments.
    MacularAnalysisDataframes.normalization_analyzing.__wrapped__(macular_analysis_dataframes_default_test,
                                                                  meta_analysis_dictionary, dict_index_default,
                                                                  parameters_meta_analysis_dict)

    # Getting the array calculated in the division meta-analysis.
    output_array = macular_analysis_dataframes_default_test.dict_analysis_dataframes["X"]["barSpeed28,5dps"].loc[
        "cond_x_latency_ms_peak_amplitude_normalization"].values

    # Manual calculation of the expected array values of the meta-analysis division.
    denominator_array = macular_analysis_dataframes_default_test.dict_analysis_dataframes["Conditions"].loc[
        "peak_amplitude_VSDI", "barSpeed28,5dps"]
    numerator_array = macular_analysis_dataframes_default_test.dict_analysis_dataframes["X"]["barSpeed28,5dps"].loc[
        "latency_VSDI_ms"].values
    output_array_expected = numerator_array / denominator_array * 8

    # Case of meta-analysis of division with a value in the denominator and an array in the numerator.
    assert np.array_equal(output_array, output_array_expected)

    # Initialisation of a dictionary of argument associated to 2 values in numerator, denominator and in output.
    meta_analysis_dictionary = {
        "numerator": ("Conditions", "barSpeed28,5dps", "VSDI",
                      "peak_amplitude", ""),
        "denominator": ("Conditions", "barSpeed28,5dps", "FiringRate_GanglionGainControl", "peak_amplitude", ""),
        "output": {"dimension": "Conditions", "condition": "barSpeed28,5dps", "name":
            "vsdi_ganglion_peak_amplitude_normalization"}}

    # Performing division meta-analysis with two unique values in the arguments.
    MacularAnalysisDataframes.normalization_analyzing.__wrapped__(macular_analysis_dataframes_default_test,
                                                                  meta_analysis_dictionary, dict_index_default,
                                                                  parameters_meta_analysis_dict)

    # Getting the array calculated in the division meta-analysis.
    output_array = macular_analysis_dataframes_default_test.dict_analysis_dataframes["Conditions"].loc[
        "vsdi_ganglion_peak_amplitude_normalization"].values

    # Manual calculation of the expected value of the meta-analysis division.
    denominator_array = macular_analysis_dataframes_default_test.dict_analysis_dataframes["Conditions"].loc[
        "peak_amplitude_FiringRate_GanglionGainControl", "barSpeed28,5dps"]
    numerator_array = macular_analysis_dataframes_default_test.dict_analysis_dataframes["Conditions"].loc[
        "peak_amplitude_VSDI", "barSpeed28,5dps"]
    output_array_expected = numerator_array / denominator_array * 8

    # Case of meta-analysis of division with unique values in the numerator and denominator.
    assert np.array_equal(output_array[0], output_array_expected)
    assert output_array[1] is np.nan

    # Remove to verify that these additions are the only changes made during the meta-analysis.
    macular_analysis_dataframes_default_test.dict_analysis_dataframes["X"]["barSpeed28,5dps"].drop(
        "latency_ms_peak_amplitude_normalization", inplace=True)
    macular_analysis_dataframes_default_test.dict_analysis_dataframes["X"]["barSpeed28,5dps"].drop(
        "cond_x_latency_ms_peak_amplitude_normalization", inplace=True)
    macular_analysis_dataframes_default_test.dict_analysis_dataframes["Conditions"].drop(
        "vsdi_ganglion_peak_amplitude_normalization", inplace=True)

    # Verify that the conditions dataframe is correct.
    assert macular_analysis_dataframes_default_test.dict_analysis_dataframes["Conditions"].equals(
        macular_analysis_dataframes_default.dict_analysis_dataframes["Conditions"])

    # Verify that the X, Y, and T dataframes for each condition are equal.
    for condition in macular_analysis_dataframes_default_test.dict_paths_pyb:
        assert macular_analysis_dataframes_default_test.dict_analysis_dataframes["X"][condition].equals(
            macular_analysis_dataframes_default.dict_analysis_dataframes["X"][condition])
        assert macular_analysis_dataframes_default_test.dict_analysis_dataframes["Y"][condition].equals(
            macular_analysis_dataframes_default.dict_analysis_dataframes["Y"][condition])
        assert macular_analysis_dataframes_default_test.dict_analysis_dataframes["Time"][condition].equals(
            macular_analysis_dataframes_default.dict_analysis_dataframes["Time"][condition])

