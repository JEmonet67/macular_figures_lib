import os

from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images

from src.figure_manager.MacularFigureMaker import MacularFigureMaker

# Get figures for test from relative path.
path_figure_test = os.path.normpath(f"{os.getcwd()}/../data_test/figure_manager/")

# Get data for test from relative path.
path_data_test = os.path.normpath(f"{os.getcwd()}/../data_test/data_manager/")


figure_dictionary = {
    "path_figure": "", "figsize": (52, 15), "shape": (2, 300),
    "subfigs": [[{"coordinates": (0, 0), "size": (70, 1)},
                 {"coordinates": (0, 114), "size": (70, 1)},
                 {"coordinates": (0, 230), "size": (70, 1)}],
                [{"coordinates": (1, 0), "size": (70, 1)},
                 {"coordinates": (1, 114), "size": (70, 1)},
                 {"coordinates": (1, 230), "size": (70, 1)}]
                ]
}

graphs_dictionaries = [[{"graph_type": "curve"},
                        {},
                        {}],
                       [{},
                        {},
                        {}]]

data_dictionaries = {"MacularDictArray": {"macular_dict_array_id": {f"{path_data_test}"}},
                   "MacularAnalysisDataframes": {"macular_analysis_dataframe_id": {f"{path_data_test}"}}}


def compare_figures(path_figure, name_figure, tol=0):
    plt.savefig(f"{path_figure}/{name_figure}_test.png")

    res_test = compare_images(f"{path_figure}/{name_figure}_test.png",
                              f"{path_figure}/{name_figure}.png", tol)

    os.remove(f"{path_figure}/{name_figure}_test.png")

    if res_test is not None:
        os.remove(f"{path_figure}/{name_figure}-failed-diff.png")
        assert False


def test_init():
    pass


def test_figure_initialisation():
    # Initialising a figure dictionary with a one-dimensional list of subfigures.
    figure_dictionary_test = {
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
    macular_figure.figure_initialisation(figure_dictionary_test)

    # Verify that the figure has been created correctly and that its axes are the correct size.
    compare_figures(path_figure_test, "empty_fig")
    assert len(macular_figure.axs) == 6
    try:
        assert len(macular_figure.axs[1]) == None
    except TypeError:
        assert True

    # Modification of the figure dictionary with two-dimensional nested subfig lists.
    figure_dictionary_test["subfigs"] = [[{"coordinates": (0, 0), "size": (70, 1)},
                                          {"coordinates": (0, 114), "size": (70, 1)},
                                          {"coordinates": (0, 230), "size": (70, 1)}],
                                         [{"coordinates": (1, 0), "size": (70, 1)},
                                          {"coordinates": (1, 114), "size": (70, 1)},
                                          {"coordinates": (1, 230), "size": (70, 1)}]]

    # Initialisation of the figure contained in MacularFigureMaker from the new figure dictionary.
    macular_figure.figure_initialisation(figure_dictionary_test)

    # Verify that the figure has been created correctly and that its axes are the correct size.
    compare_figures(path_figure_test, "empty_fig")
    assert len(macular_figure.axs) == 2
    assert len(macular_figure.axs[0]) == 3
