import copy
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images

from src.data_manager.MacularAnalysisDataframes import MacularAnalysisDataframes
from src.data_manager.MacularDictArray import MacularDictArray
from src.figure_manager.MacularFigureMaker import MacularFigureMaker

# Get figures for test from relative path.
path_figure_test = os.path.normpath(f"{os.getcwd()}/../data_test/figure_manager/")

# Get data for test from relative path.
path_data_test = os.path.normpath(f"{os.getcwd()}/../data_test/data_manager/")

# Initialisation of a 3x3 figure dictionary with sub-figures grouped by horizontal line.
figure_dictionary_3x3 = {
    "path_figure": "", "figsize": (52, 15), "shape": (2, 300),
    "subfigs": [[{"coordinates": (0, 0), "size": (70, 1)},
                 {"coordinates": (0, 114), "size": (70, 1)},
                 {"coordinates": (0, 230), "size": (70, 1)}],
                [{"coordinates": (1, 0), "size": (70, 1)},
                 {"coordinates": (1, 114), "size": (70, 1)},
                 {"coordinates": (1, 230), "size": (70, 1)}]
                ]
}

figure_dictionary_2x1 = {"path_figure": "", "figsize": (52, 15), "shape": (1, 200),
                         "subfigs": [{"coordinates": (0, 0), "size": (70, 1)},
                                     {"coordinates": (0, 114), "size": (70, 1)}]}

# TODO Mettre à jour
graphs_dictionaries = [[{"graph_type": "curves", "curves_number": 4,
                         "graph_data": {
                             "type": ["MacularDictArrays"] * 4,
                             "name": ["RC_RM_dSGpCP0033_barSpeed30dps"] * 3 + [
                                 "RC_RM_dSGpCP0083_barSpeed28,5dps"],
                             "measurement": ["VSDI"] * 4,
                             "index": ["temporal_centered_ms"] * 4,
                             "i_index": [31, 36, 41, 36],
                             "slicer_indices": [(7, 31, None), (7, 36, None), (7, 41, None), (7, 36, None)]}},
                        {"graph_type": "curves", "curves_number": 3,
                         "graph_data": {
                             "type": ["MacularAnalysisDataframes"],
                             "name": ["barSpeed"],
                             "dimension": ["X"],
                             "condition": ["barSpeed30dps"],
                             "analysis": ["time_to_peak_VSDI_ms"]}},
                        {"graph_type": "curves", "curves_number": 3,
                         "graph_data": {
                             "type": "MacularDictArrays",
                             "name": ["RC_RM_dSGpCP0033_barSpeed30dps"] * 3 + ["RC_RM_dSGpCP0083_barSpeed28,5dps"],
                             "measurement": "VSDI",
                             "index": "temporal_centered_ms",
                             "i_index": [31, 36, 41, 36],
                             "slicer_indices": [(7, 31, None), (7, 36, None), (7, 41, None), (7, 36, None)]}}],
                       [{},
                        {},
                        {}]]

data_dictionaries = {
    "MacularDictArrays":
        {
            "RC_RM_dSGpCP0033_barSpeed30dps": f"{path_data_test}/RC_RM_dSGpCP0033_barSpeed30dps_default_0f.pyb",
            "RC_RM_dSGpCP0083_barSpeed28,5dps": f"{path_data_test}/RC_RM_dSGpCP0083_barSpeed28,5dps_default_0f.pyb"
        },
    "MacularAnalysisDataframes":
        {
            "barSpeed": f"{path_data_test}/fully_meta_analyzed_macular_analysis_dataframe_copy.pyb"}}

# Convert the current data dictionary to obtain each of its data structures.
converted_data_dictionaries = MacularFigureMaker.convert_data_dictionary_paths_to_data_structures(
    copy.deepcopy(data_dictionaries))


def compare_figures(path_figure, name_figure, tol=0.):
    """Function for testing the comparison between the current figure and a reference figure that has already been
    saved.

    The test figure is first saved so that it can be compared with the test figure. This is a comparison between two
    images. Once this has been done, the saved test figure is deleted. The process is completed by checking the result
    of the comparison.

    Parameters
    ----------
    path_figure : str
        Path to the file where the reference figure is located.

    name_figure : str
        Name of the reference figure.

    tol : float
        Tolerance in the comparison between the two images of figures. It is set by default to no difference accepted.
    """
    # Saves the test file based on the name of the reference file.
    plt.savefig(f"{path_figure}/{name_figure}_test.png")

    # Comparison between reference and test images.
    res_test = compare_images(f"{path_figure}/{name_figure}_test.png",
                              f"{path_figure}/{name_figure}.png", tol)

    # Deletes the previously saved test figure.
    os.remove(f"{path_figure}/{name_figure}_test.png")

    # Checking for errors when comparing the two images in the figures.
    if res_test is not None:
        # Delete an image file showing the difference between the two images.
        os.remove(f"{path_figure}/{name_figure}-failed-diff.png")
        assert False


def test_init():
    pass


def test_figure_initialisation():
    # Initialising a figure dictionary with a one-dimensional list of subfigures.
    figure_dictionary_6x1 = {
        "figsize": (52, 15), "shape": (2, 300),
        "subfigs": [{"coordinates": (0, 0), "size": (70, 1)},
                    {"coordinates": (0, 114), "size": (70, 1)},
                    {"coordinates": (0, 230), "size": (70, 1)},
                    {"coordinates": (1, 0), "size": (70, 1)},
                    {"coordinates": (1, 114), "size": (70, 1)},
                    {"coordinates": (1, 230), "size": (70, 1)}]
    }

    # Creation of MacularFigureMaker.
    macular_figure = MacularFigureMaker()

    # Initialisation of the figure contained in MacularFigureMaker from the figure dictionary.
    macular_figure.figure_initialisation(figure_dictionary_6x1)

    # Verify that the figure has been created correctly and that its axes are the correct size.
    compare_figures(path_figure_test, "empty_fig")
    assert len(macular_figure.axs) == 6
    try:
        assert len(macular_figure.axs[1]) == None
    except TypeError:
        assert True

    # Initialisation of the figure contained in MacularFigureMaker from the new figure dictionary.
    macular_figure.figure_initialisation(figure_dictionary_3x3)

    # Verify that the figure has been created correctly and that its axes are the correct size.
    compare_figures(path_figure_test, "empty_fig")
    assert len(macular_figure.axs) == 2
    assert len(macular_figure.axs[0]) == 3


def test_convert_data_dictionary_paths_to_data_structures():
    # Conversion of path data into data structure in the data dictionary.
    MacularFigureMaker.convert_data_dictionary_paths_to_data_structures(data_dictionaries)

    # Test of each data structure in the data dictionary.
    assert MacularDictArray.equal(data_dictionaries["MacularDictArrays"]["RC_RM_dSGpCP0083_barSpeed28,5dps"],
                                  MacularDictArray.load(f"{path_data_test}/RC_RM_dSGpCP0083_barSpeed28,5dps_"
                                                        f"default_0f.pyb"))
    assert MacularDictArray.equal(data_dictionaries["MacularDictArrays"]["RC_RM_dSGpCP0033_barSpeed30dps"],
                                  MacularDictArray.load(f"{path_data_test}/RC_RM_dSGpCP0033_barSpeed30dps_"
                                                        f"default_0f.pyb"))
    assert MacularAnalysisDataframes.equal(data_dictionaries["MacularAnalysisDataframes"]["barSpeed"],
                                           MacularDictArray.load(
                                               f"{path_data_test}/fully_meta_analyzed_macular_analysis_"
                                               f"dataframe_copy.pyb"))


def test_clear_axs():
    # Import a macular figure maker with an empty 3x3 figure.
    with open(f"{path_figure_test}/empty_3x3_MacularFigureMaker.pyb", "rb") as file_fig:
        macular_figure_maker_3x3 = pickle.load(file_fig)

    # Loop on the axis sizes of the macular figure maker to add plots.
    for i in range(2):
        for j in range(3):
            macular_figure_maker_3x3.axs[i][j].plot([i for i in range(10)], [i for i in range(10)])

    # Cleaning of all axes.
    macular_figure_maker_3x3.clear_axs()

    # Checking that cleaning is working properly.
    compare_figures(path_figure_test, "empty_fig")


def test_make_figure():
    pass


def test_make_all_graphs():
    pass


def test_make_group_graphs():
    pass


def test_make_subfig_graphs():
    # TODO Finir assert
    with open(f"{path_figure_test}/empty_2x1_MacularFigureMaker.pyb", "rb") as file_fig:
        macular_figure_maker_2x1 = pickle.load(file_fig)

    macular_figure_maker_2x1.make_subfig_graphs(macular_figure_maker_2x1.axs[0], graphs_dictionaries[0][0].copy(),
                                                converted_data_dictionaries)


def test_adapting_dictionary_lists_length():
    # Initialisation of a dictionary of lists with a maximum size of 1.
    multiple_graph_data_dictionary_length_one = {
        "type": "MacularDictArrays",
        "name": ["RC_RM_dSGpCP0033_barSpeed30dps"],
        "measurement": "VSDI",
        "index": "temporal_centered_ms",
        "i_index": 31,
        "slicer_indices": [(7, 31, None)]}

    # Initialisation of a correctly sized dictionary of size 1.
    multiple_graph_data_dictionary_length_one_correct = {
        "type": ["MacularDictArrays"],
        "name": ["RC_RM_dSGpCP0033_barSpeed30dps"],
        "measurement": ["VSDI"],
        "index": ["temporal_centered_ms"],
        "i_index": [31],
        "slicer_indices": [(7, 31, None)]}

    # Testing the adaptation of dictionary list sizes to an expected size of 1.
    MacularFigureMaker.adapting_dictionary_lists_length(multiple_graph_data_dictionary_length_one, 1)
    assert multiple_graph_data_dictionary_length_one == multiple_graph_data_dictionary_length_one_correct

    # Initialisation of a dictionary of lists with a maximum size of 4.
    multiple_graph_data_dictionary_length_four = {
        "type": ["MacularDictArrays"],
        "name": ["RC_RM_dSGpCP0033_barSpeed30dps"] * 3 + ["RC_RM_dSGpCP0083_barSpeed28,5dps"],
        "measurement": "VSDI",
        "index": "temporal_centered_ms",
        "i_index": [31, 36, 41, 36],
        "slicer_indices": [(7, 31, None), (7, 36, None), (7, 41, None), (7, 36, None)]}

    # Testing the adaptation of dictionary list sizes to an expected size superior to 1.
    MacularFigureMaker.adapting_dictionary_lists_length(multiple_graph_data_dictionary_length_four, 4)
    assert multiple_graph_data_dictionary_length_four == graphs_dictionaries[0][0]["graph_data"]


def test_extract_multiple_data_series_from_data_structures():
    # Set up the list of data series from the dictionary defined above.
    list_data_series = MacularFigureMaker.extract_multiple_data_series_from_data_structures(
        graphs_dictionaries[0][0]["graph_data"], converted_data_dictionaries)

    i = 0
    for x_col in (31, 36, 41, 36):
        print(x_col)
        if i < 3:
            current_data_serie_correct = pd.Series(
                converted_data_dictionaries["MacularDictArrays"]["RC_RM_dSGpCP0033_barSpeed30dps"].data["VSDI"][7,
                x_col, :].round(
                    4),
                converted_data_dictionaries["MacularDictArrays"]["RC_RM_dSGpCP0033_barSpeed30dps"].index[
                    "temporal_centered_ms"][x_col].round(4))
            assert list_data_series[i].equals(current_data_serie_correct)
        else:
            current_data_serie_correct = pd.Series(
                converted_data_dictionaries["MacularDictArrays"]["RC_RM_dSGpCP0083_barSpeed28,5dps"].data["VSDI"][7,
                x_col, :].round(
                    4),
                converted_data_dictionaries["MacularDictArrays"]["RC_RM_dSGpCP0083_barSpeed28,5dps"].index[
                    "temporal_centered_ms"][x_col].round(4))
            assert list_data_series[i].equals(current_data_serie_correct)
        i += 1


def test_extract_data_serie_from_data_structures():
    # Initialisation of a graph dictionary for the MacularAnalysisDataframes case and without conditions.
    graph_dictionary_analysis_dataframes_no_cond = {"type": "MacularAnalysisDataframes", "name": "barSpeed",
                                                    "dimension": "Conditions",
                                                    "analysis": "horizontal_minimal_latency_ms"}

    # Extracting the data array from MacularAnalysisDataframes.
    data_serie_test = MacularFigureMaker.extract_data_serie_from_data_structures(
        graph_dictionary_analysis_dataframes_no_cond, converted_data_dictionaries).astype(float)
    assert data_serie_test.equals(pd.Series(([-48.141, -41.743]), (['barSpeed28,5dps', 'barSpeed30dps'])))

    # Initialisation of a graph dictionary for the MacularAnalysisDataframes case and with conditions.
    graph_dictionary_analysis_dataframes_cond = {"type": "MacularAnalysisDataframes", "name": "barSpeed",
                                                 "dimension": "X", "condition": "barSpeed30dps",
                                                 "analysis": "time_to_peak_VSDI_ms"}

    # Extracting the data array from MacularAnalysisDataframes.
    data_serie_test = MacularFigureMaker.extract_data_serie_from_data_structures(
        graph_dictionary_analysis_dataframes_cond, converted_data_dictionaries)

    # Data array correctly retrieved for the "X" VSDI time to peak.
    data_array_correct = np.array([189.4, 197.4, 203.8, 211.8, 218.2, 227.8, 234.2, 242.2, 248.6, 256.6, 264.6, 271.0,
                                   279.0, 285.4, 295.0, 299.8, 309.4, 315.8, 323.8, 331.8, 338.2, 346.2, 352.6, 362.2,
                                   368.6, 376.6, 383.0, 391.0, 399.0, 407.0, 413.4, 421.4, 429.4, 435.8, 443.8, 450.2,
                                   459.8, 466.2, 474.2, 480.6, 488.6, 496.6, 503.0, 512.6, 517.4, 527.0, 533.4, 541.4,
                                   547.8, 555.8, 563.8, 571.8, 579.8, 586.2, 594.2, 600.6, 610.2, 616.6, 624.6, 631.0,
                                   639.0, 647.0, 653.4, 661.4, 667.8, 677.4, 682.2, 690.2, 696.6, 703.0, 709.4, 714.2,
                                   720.6])

    assert data_serie_test.equals(pd.Series(data_array_correct, np.linspace(1.125, 17.325, 73).round(4)))

    # Initialisation of a graph dictionary for the MacularAnalysisDataframes case and without conditions.
    graph_dictionary_dict_array = {"type": "MacularDictArrays", "name": "RC_RM_dSGpCP0033_barSpeed30dps",
                                   "measurement": "VSDI", "index": "temporal_ms", "slicer_indices": (7, 36, None)}

    data_serie_test = MacularFigureMaker.extract_data_serie_from_data_structures(graph_dictionary_dict_array,
                                                                                 converted_data_dictionaries)

    # Data array correctly retrieved and then split to obtain only an interval of the rising part of the curve.
    data_serie_correct = pd.Series([0.0318, 0.0322, 0.0325, 0.0329, 0.0332, 0.0335, 0.0338, 0.034, 0.0343,
                                    0.0346, 0.0348, 0.035, 0.0352, 0.0354, 0.0356, 0.0358, 0.036, 0.0361,
                                    0.0363, 0.0364, 0.0365, 0.0367, 0.0368, 0.0369, 0.037],
                                   np.linspace(400.6, 439, 25).round(4))

    # Verify the calculation of the data array by comparing an interval in the rising slope of the curve.
    data_serie_test.index = data_serie_test.index
    assert data_serie_test.iloc[250:275].equals(data_serie_correct)


def test_curves_plotting():
    with open(f"{path_figure_test}/empty_2x1_MacularFigureMaker.pyb", "rb") as file_fig:
        macular_figure_maker_2x1 = pickle.load(file_fig)

    with open(f"{path_figure_test}/subfig_graphs_dictionary_data_extracted.pyb", "rb") as file_fig:
        subfig_graphs_dictionary_data_extracted = pickle.load(file_fig)

    subfig_graphs_dictionary_data_extracted["curves"] = {"labels": "", "color": ["black"], "marker": [""], "markersize":
        [20], "lineweight": [6], "linestyle": ["-"]}

    macular_figure_maker_2x1.curves_plotting(macular_figure_maker_2x1.axs[0], subfig_graphs_dictionary_data_extracted)
    plt.show()
