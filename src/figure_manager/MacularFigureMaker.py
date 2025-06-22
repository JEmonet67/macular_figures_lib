import pandas as pd
from matplotlib import pyplot as plt

from src.data_manager.DataPreprocessor import DataPreprocessor
from src.data_manager.MacularAnalysisDataframes import MacularAnalysisDataframes
from src.data_manager.MacularDictArray import MacularDictArray


class MacularFigureMaker:
    """
    Utilisation de matplotlib

    Expliquer comment implémenter et ajouter de nouveaux graphes.

    Les graphes à ploter par figure sont organisés au sein de groupes avec un point commun. Il est possible que le seul
    point commun est d'être totalement hétérogène en comparaison d'autres listes homogènes.

    Attributes
    ----------
    figure : matplotlib.figure.Figure


    axs : list


    """

    def __init__(self):
        """
        """
        self.fig = None
        self.axs = None

    def figure_initialisation(self, figure_dictionary):
        """Initialisation of a new figure and new axes associated with MacularFigureMaker.

        The structure of the new figure is entirely defined by the parameters in the figure dictionary. The figure is
        created using the subplot2grid function from the pyplot module of matplotlib. This function allows you to create
        a figure in which the positions, number and sizes of the subfigures are fully customisable. The function divides
        the area of the figure into a 2D grid. When adding subfigures, you can specify the position of the subfigures
        within this grid. You can also specify the size of the subfigure in terms of the number of rows and columns in
        the grid.

        Within this function, all information for each subfigure is present in the ‘subfigs’ key of the figure
        dictionary. This key is associated with a data structure consisting of dictionaries containing the subfigure
        parameters. The data structure can be a simple one-dimensional list or a nested two-dimensional list. This data
        structure is copied and also used to organise the ‘axs’ attribute of MacularFigureMaker. The use of these nested
        two-dimensional lists allows several subfigures intended to undergo the same processing to be grouped together.

        Parameters
        ----------
        figure_dictionary : dict
            Dictionary containing all the parameters necessary to initialise a new figure.

            The dictionary contains the following keys:
            - ‘figsize’ is the size of the figure in inches.
            - ‘shape’ is the number of rows and columns that make up the ‘subplot2grid’ grid that divides the figure.
            - ‘subfigs’ is a list or nested double list containing all the dictionaries for each figure. The two keys
            in this dictionary are the “coordinates” where the subfigure should be placed and its ‘size’.
        """
        # Initialisation of the figure with size based on the figure dictionary.
        self.figure = plt.figure(figsize=figure_dictionary["figsize"])

        # Initialisation of axes with the same list structure as that found in the subfigs figure dictionary.
        self.axs = figure_dictionary["subfigs"].copy()

        # Loop of the first dimension of the nested or non-nested list of subfigs.
        for i in range(len(figure_dictionary["subfigs"])):
            # Case of a nested list of two-dimensional subfigures.
            try:
                # Loops the second dimension of the nested list of subfigs, if there is one.
                for j in range(len(figure_dictionary["subfigs"][i])):
                    # Creation of the current subfigure with coordinates and size specified in the figure dictionary.
                    self.axs[i][j] = plt.subplot2grid(shape=figure_dictionary["shape"],
                                                      loc=figure_dictionary["subfigs"][i][j]["coordinates"],
                                                      colspan=figure_dictionary["subfigs"][i][j]["size"][0],
                                                      rowspan=figure_dictionary["subfigs"][i][j]["size"][1])

            # Case of a one-dimensional list of subfigs.
            except KeyError:
                # Creation of the current subfigure with coordinates and size specified in the figure dictionary.
                self.axs[i] = plt.subplot2grid(shape=figure_dictionary["shape"],
                                               loc=figure_dictionary["subfigs"][i]["coordinates"],
                                               colspan=figure_dictionary["subfigs"][i]["size"][0],
                                               rowspan=figure_dictionary["subfigs"][i]["size"][1])

    @staticmethod
    def convert_data_dictionary_paths_to_data_structures(data_dictionaries):
        """Import all data structures whose file paths are present in the data dictionary.

        The function replaces the character string of the file path with the data structure to which it points.

        Parameters
        ----------
        data_dictionaries : dict
            Dictionary containing all file paths to data structures to be used for the plots in the current figure.

            It consists of two keys, ‘MacularDictArrays’ and ‘MacularAnalysisDataframes’. Each of these is in turn
            associated with a dictionary linking the identifier of a data structure to its pyb file path.

        Returns
        ----------
        data_dictionaries : dict of dict
            Dictionary containing all the data structures to be used for the plots in the current figure.
        """
        # Loop through the data types and data names.
        for data_type in data_dictionaries:
            for name_data in data_dictionaries[data_type]:
                path_data = data_dictionaries[data_type][name_data]

                # Loading MacularDictArray type structures.
                if data_type == "MacularDictArrays":
                    data_dictionaries[data_type][name_data] = MacularDictArray.load(path_data)
                # Loading MacularAnalysisDataframes type structures.
                elif data_type == "MacularAnalysisDataframes":
                    data_dictionaries[data_type][name_data] = MacularAnalysisDataframes.load(path_data)

        return data_dictionaries

    def clear_axs(self):
        """Function to delete all plots from the current figure without modifying its axes so that it is available again
        for creating a new figure with the same configuration.
        """
        # Loop of the first dimension of the nested or non-nested list of subfigs.
        for i in range(len(self.axs)):
            # Case of a nested list of two-dimensional subfigures.
            try:
                # Loops the second dimension of the nested list of subfigs, if there is one.
                for j in range(len(self.axs[i])):
                    # Clearing of the current subfigure.
                    self.axs[i][j].clear()

            # Case of a one-dimensional list of subfigs.
            except KeyError:
                # Clearing of the current subfigure.
                self.axs[i].clear()

    def make_figure(self, path_figure, graphs_dictionaries, data_dictionaries):
        """Creation of a new figure with all its markers.

        Parameters
        ----------
        path_figure : dict
            Path to the file where the current figure should be saved.

        graphs_dictionaries : dict
            Dictionary containing all the parameters needed to plot all graphs in all subplots of the current figure.

        data_dictionaries : dict
            Dictionary containing all the data structures to be used for the plots in the current figure.
        """
        # Resets the figure while retaining its axes so that it is ready to receive a new figure.
        self.clear_axs()

        # Set up all graphs for each subplot in the figure.
        self.make_all_graphs(graphs_dictionaries, data_dictionaries)

        # Saving the figure.
        plt.savefig(path_figure)

    def make_all_graphs(self, graphs_dictionaries, data_dictionaries):
        """Creation of all graphs in a figure.

        Parameters
        ----------
        graphs_dictionaries : dict
            Dictionary containing all the parameters needed to plot all graphs in all subplots of the current figure.

        data_dictionaries : dict
            Dictionary containing all the data structures to be used for the plots in the current figure.
        """
        for axs_group, graphs_group_dictionaries in zip(self.axs, graphs_dictionaries):
            # Case with two-dimensional nested subfig lists.
            if isinstance(axs_group == list):
                self.make_group_graphs(axs_group, graphs_group_dictionaries, data_dictionaries)

            # Case with one-dimensional nested subfig lists.
            else:
                self.make_one_graph(axs_group, graphs_group_dictionaries, data_dictionaries)

    def make_group_graphs(self, axs_group, graphs_group_dictionaries, data_dictionaries):
        """Plotting of all graphs in a list of grouped subfigures.

        These are used to create identical plots with the same parameters for several different subfigures.

        Parameters
        ----------
        axs_group : list of list or list of matplotlib.axes.
            Ax matplotlib where to plot all the grouped graphs of the subfigure.

        graphs_group_dictionaries : dict
            Dictionary containing all the parameters required to plot all the graphs in a group of subplots.

        data_dictionaries : dict
            Dictionary containing all the data structures to be used for the plots in the current figure.
        """
        # Case with two-dimensional nested subfig lists.
        if isinstance(graphs_group_dictionaries, list):
            # Loop through both the list of axes and the list of graph dictionaries.
            for ax, graph_dictionary in zip(axs_group, graphs_group_dictionaries):
                self.make_subfig_graphs(ax, graph_dictionary, data_dictionaries)

        # Case with one-dimensional nested subfig lists.
        else:
            # Loop only on the list of axs.
            for ax in axs_group:
                self.make_subfig_graphs(ax, graphs_group_dictionaries, data_dictionaries)

    def make_subfig_graphs(self, ax, subfig_graphs_dictionary, data_dictionaries):
        """Plotting all graphs of a subplot.

        Parameters
        ----------
        ax : matplotlib.axes
            Ax matplotlib where to plot the graphs of the subfigure.

        subfig_graphs_dictionary : dict
            Dictionary containing all the parameters required to plot all graphs in the current subplot.

        data_dictionaries : dict
            Dictionary containing all the data structures to be used for the plots in the current figure.
        """
        # Dictionary containing all graphs currently implemented.
        available_graphs_dict = {
            "curves": self.curves_plotting,
            # "heatmap": self.heatmap_plotting,
            # "barplot": self.barplots_plotting
        }

        MacularFigureMaker.adapting_dictionary_lists_length(subfig_graphs_dictionary["graph_data"],
                                                            subfig_graphs_dictionary["curves_number"])
        subfig_graphs_dictionary["graph_data"] = MacularFigureMaker.extract_multiple_data_series_from_data_structures(
            subfig_graphs_dictionary["graph_data"], data_dictionaries)

        # Creation of a plot corresponding to the type of graph in the sub-figure.
        available_graphs_dict[subfig_graphs_dictionary["graph_type"]](ax, subfig_graphs_dictionary)

    @staticmethod
    def adapting_dictionary_lists_length(dictionary_lists, expected_length):
        """Function that allows you to adjust the size of all lists in a dictionary so that they are all the same
        expected length.

        The function only works to adapt lists of size 1 to the correct size or by transforming objects other than
        lists.

        Parameters
        ----------
        dictionary_lists : dict
            Dictionary containing keys, some of which are associated with lists or other objects.

        expected_length : int
            Expected length of each list of the dictionary.
        """
        # A loop through all the keys in the dictionary.
        for key in dictionary_lists:
            # Cases where the expected length is greater than 1.
            if expected_length > 1:
                # Case when the key value is not a list.
                if not isinstance(dictionary_lists[key], list):
                    dictionary_lists[key] = [dictionary_lists[key]] * expected_length

                # Case when the key value is a list of size 1.
                elif len(dictionary_lists[key]) == 1:
                    dictionary_lists[key] = (dictionary_lists[key] *expected_length)
            # Cases where the expected length is equal to 1.
            else:
                # Verify that the key value is not a list.
                if not isinstance(dictionary_lists[key], list):
                    dictionary_lists[key] = [dictionary_lists[key]]

    @staticmethod
    def extract_multiple_data_series_from_data_structures(multiple_graph_data_dictionary, data_dictionaries):
        """Function that extracts a list serie of data located in a data structure in the data dictionary using a list
        of dictionary containing the coordinates of each serie.

        Parameters
        ----------
        multiple_graph_data_dictionary : dict
            Multiple dictionary putting the coordinates of the desired data serie within data dictionaries.

            Each coordinate consists of a key associated with a part of the coordinate in the form of a list. These
            lists are all the same length and allow the coordinates of multiple data series to be stored. Each
            coordinate allows a curve to be plotted in a subgraph.

            Example of length 4 :
            {"type": ["MacularDictArrays"] * 4,
             "name": ["RC_RM_dSGpCP0033_barSpeed30dps"] * 2 + ["RC_RM_dSGpCP0083_barSpeed28,5dps"] * 2,
             "measurement": ["VSDI"] * 4,
             "index": ["temporal_centered_ms"] * 4,
             "slicer_indices": [(7, 31, None), (7, 36, None), (7, 41, None), (7, 36, None)]}

        data_dictionaries : dict
            Dictionary containing all the data structures from which the data serie can be extracted.

            It consists of two keys, ‘MacularDictArrays’ and ‘MacularAnalysisDataframes’. Each of these is in turn
            associated with a dictionary linking the identifier of a data structure to its pyb file path.

        Returns
        ----------
        data_series : list of pd.Series
            Extracted each data serie whose coordinates have been specified in input.
        """
        # Initialisation of the empty data series list.
        data_series = []

        # Loop over the length of the lists contained in the multiple graph data dictionary.
        for i in range(len(multiple_graph_data_dictionary["name"])):
            # Extraction of the current graph data dictionary.
            current_graph_data_dictionary = {key: value[i] for key, value in multiple_graph_data_dictionary.items()}
            # Increment the list of data series with the series corresponding to the current coordinates.
            data_series += [MacularFigureMaker.extract_data_serie_from_data_structures(current_graph_data_dictionary,
                                                                                       data_dictionaries)]

        print()
        return data_series

    @staticmethod
    def extract_data_serie_from_data_structures(graph_data_dictionary, data_dictionaries):
        """Function that extracts a serie of data located in a data structure in the data dictionary using a dictionary
        containing the coordinates of the serie.

        Parameters
        ----------
        graph_data_dictionary : dict
            Dictionary grouping the coordinates of the desired data serie within data dictionaries. It consists of
            invariant keys and others that depend on the type of data structure. The invariant keys are the ‘type’ of
            the data structure and its ‘name’.

            In the case of MacularAnalysisDataframes, the “dimension” of the latter and its ‘analysis’ are added. There
            is one last optional key that is only added for MacularAnalysisDataframes dimensions that are not composed
            of multiple dataframes depending on the simulation conditions. In these cases (‘X’, ‘Y’, “Time”), the key
            ‘condition’ must also be added.

            Example :
            {"type": "MacularAnalysisDataframes",
             "name": "barSpeed",
             "dimension": "Conditions",
             "analysis": "horizontal_minimal_latency_ms"}

            In the case of MacularDictArrays, we have the ‘measurement’ and the index name. The name of the index allows
            you to retrieve the corresponding index and associate it with the data array within a series. We finally
            have the indices to be used to slice the array. If you do not want to cut along a dimension, simply enter
            None at the index of that dimension.

            Example :
            {"type": "MacularDictArrays",
             "name": "RC_RM_dSGpCP0033_barSpeed30dps",
             "measurement": "VSDI",
             "index": "temporal_ms",
             "slicer_indices": (7, 36, None)}

        data_dictionaries : dict
            Dictionary containing all the data structures from which the data serie can be extracted.

            It consists of two keys, ‘MacularDictArrays’ and ‘MacularAnalysisDataframes’. Each of these is in turn
            associated with a dictionary linking the identifier of a data structure to its pyb file path.

        Returns
        ----------
        data_serie : pd.Series
            Extracted data serie whose coordinates have been specified.
        """
        # Case of MacularAnalysisDataframes data structure.
        if graph_data_dictionary["type"] == "MacularAnalysisDataframes":
            # Case of unique dataframe dimensions in MacularAnalysisDataframes.
            if (graph_data_dictionary["dimension"] == "Conditions" or graph_data_dictionary["dimension"] ==
                    "MetaConditions"):
                data_serie = data_dictionaries[graph_data_dictionary["type"]][
                    graph_data_dictionary["name"]].dict_analysis_dataframes[
                    graph_data_dictionary["dimension"]].loc[graph_data_dictionary["analysis"]]

            # Case of multiple dataframes dimensions in MacularAnalysisDataframes.
            else:
                data_serie = data_dictionaries[graph_data_dictionary["type"]][
                    graph_data_dictionary["name"]].dict_analysis_dataframes[
                    graph_data_dictionary["dimension"]][
                    graph_data_dictionary["condition"]].loc[graph_data_dictionary["analysis"]]

        # Case of MacularDictArray data structure.
        elif graph_data_dictionary["type"] == "MacularDictArrays":
            data = DataPreprocessor.array_slicing(data_dictionaries[graph_data_dictionary["type"]][
                                                      graph_data_dictionary["name"]].data[
                                                      graph_data_dictionary["measurement"]],
                                                  graph_data_dictionary["slicer_indices"]).round(4)
            index = data_dictionaries[graph_data_dictionary["type"]][graph_data_dictionary["name"]].index[
                graph_data_dictionary["index"]].round(4)
            # Case of 2D indexes.
            if index.ndim > 1:
                index = index[graph_data_dictionary["i_index"]]

            data_serie = pd.Series(data, index)

        return data_serie

    def curves_plotting(self, ax, subfig_graphs_dictionary):
        """

        Parameters
        ----------
        ax : matplotlib.axes
            Ax matplotlib where to plot the graphs of the subfigure.

        subfig_graphs_dictionary : dict
            Dictionary containing all the parameters required to plot all graphs in the current subplot.

            Le dictionnaire contient trois clés du dictionnaires qui sont obligatoires. La première est "graph_type" qui
            spécifie quel type de graphe à ploter dans la sous-figure. Le second est "graph_data" qui contient la liste
            de toutes les données à ploter. Le type de donnée dépend du type de graphique. Dans le cas où le type de
            graphique est "curves" il faut également obligatoirement ajouter la clé "curves_number" indiquant le nombre
            de courbes à ploter dans la sous-figure. En plus de tout cela, le dictionnaire contient aussi plusieurs clés
            facultatives. Les clés qui ne sont pas dans le dictionnaire donné en input ont une valeur par défaut qui leur
            est attribué.

            Les différentes clés facultatives sont :
            - xlim et ylim pour recadrer la région d'affichage du graphe. Tous deux sont définis par un tuple contenant
            les deux bords maximaux de la nouvelle région. Si l'on veut modifier qu'un seul des bords de xlim ou ylim
            on peut mettre "None" au bord que l'on ne souhaite pas changer : (8, None).
            - xticks et yticks pour choisir une autre liste ou array de valeurs pour définir les ticks des axes.
            -
        """
        n = subfig_graphs_dictionary["curves_number"]

        subfig_graphs_dictionary_default = {
            "ticks": {"xlim": (None, None), "ylim": (None, None), "xticks": None, "yticks": None, "fontsize": 75,
                      "length_major": 15, "width_major": 5, "length_minor": 12, "width_minor": 4,
                      "xticks_colors": "black", "yticks_colors": "black"},
            "labels": {"name": "", "fontsize": 75, "pad": 5, "fontweight": "bold", "color": "black"},
            "legend": {"fontsize": 65, "loc": "best", "frameon": False, "pad": 0.8},
            "spines": {"left": 6, "bottom": 6, "top": 0, "right": 0, "color_left": "black", "color_bottom": "black",
                       "color_top": "black", "color_right": "black"},
            "curves": {"labels": [""], "color": ["black"], "marker": [""], "markersize": [20], "lineweight": [6],
                       "linestyle": ["-"], "ax": ["main"]},
            "twin_axis": False # {"ticks_colors": "black}
        }

        MacularFigureMaker.adapting_dictionary_lists_length(subfig_graphs_dictionary["curves"], n)

        # Merging of default dictionary keys not present in the crop dictionary provided as input.
        params = {}
        for type_params in subfig_graphs_dictionary_default:
            params[type_params] = {key: subfig_graphs_dictionary[key]
                                if key in subfig_graphs_dictionary else subfig_graphs_dictionary_default[key] for key in
                                                            subfig_graphs_dictionary_default}

        ax.set_xlim(params["ticks"]["xlim"][0], params["ticks"]["xlim"][1])
        ax.set_ylim(params["ticks"]["ylim"][0], params["ticks"]["ylim"][1])

        ax.setxlabel(params["labels"]["name"], params["labels"]["fontsize"], params["labels"]["pad"],
                     params["labels"]["fontweight"], params["labels"]["color"])
        ax.setylabel(params["labels"]["name"], params["labels"]["fontsize"], params["labels"]["pad"],
                     params["labels"]["fontweight"], params["labels"]["color"])

        if params["ticks"]["xticks"] is not None:
            ax.set_xticks(params["ticks"]["xticks"])
            ax.set_xticklabels(params["ticks"]["xticks"])

        if params["ticks"]["xticks"] is not None:
            ax.set_yticks(params["ticks"]["xticks"])
            ax.set_yticklabels(params["ticks"]["xticks"])

        ax.tick_params(axis="x", which="major", labelsize=params["ticks"]["fontsize"], colors=params["ticks"][
            "xticks_colors"], length=params["ticks"]["length_major"], width=params["ticks"]["width_major"])
        ax.tick_params(axis="y", which="major", labelsize=params["ticks"]["fontsize"], colors=params["ticks"][
            "yticks_colors"], length=params["ticks"]["length_major"], width=params["ticks"]["width_major"])
        ax.tick_params(axis="x", which="minor", labelsize=params["ticks"]["fontsize"], colors=params["ticks"][
            "xticks_colors"], length=params["ticks"]["length_minor"], width=params["ticks"]["width_minor"])
        ax.tick_params(axis="y", which="minor", labelsize=params["ticks"]["fontsize"], colors=params["ticks"][
            "yticks_colors"], length=params["ticks"]["length_minor"], width=params["ticks"]["width_minor"])

        for position in ['left', 'bottom', 'top', 'right']:
            ax.spines[position].set(lw=params["spines"][position])
            ax.spines[position].set_color(lw=params["spines"][f"color_{position}"])

        if params["twin_axis"]:
            ax_twin = ax.twinx()
            ax_twin.tick_params(which="major", labelsize=params["ticks"]["fontsize"], colors=params["twin_axis"][
                "colors"], length=params["ticks"]["length_major"], width=params["ticks"]["width_major"])
            ax_twin.tick_params(which="minor", labelsize=params["ticks"]["fontsize"], colors=params["twin_axis"][
                "colors"], length=params["ticks"]["length_minor"], width=params["ticks"]["width_minor"])

        list_plot = []
        for i_plot in range(n):
            if params["curves"]["lineweight"] == "main":
                list_plot += [ax.plot(subfig_graphs_dictionary["graph_data"][i_plot], c=params["curves"]["color"][i_plot],
                                      lw=params["curves"]["lineweight"][i_plot], ls=params["curves"]["linestyle"][i_plot],
                                      label=params["curves"]["labels"][i_plot], marker=params["curves"]["marker"][i_plot],
                                      markersize=params["curves"]["markersize"][i_plot])]
            elif params["curves"]["lineweight"] == "twin":
                list_plot += [ax_twin.plot(subfig_graphs_dictionary["graph_data"][i_plot], c=params["curves"]["color"][i_plot],
                            lw=params["curves"]["lineweight"][i_plot], ls=params["curves"]["linestyle"][i_plot],
                            label=params["curves"]["labels"][i_plot], marker=params["curves"]["marker"][i_plot],
                            markersize=params["curves"]["markersize"][i_plot])]

        all_labels = [label.get_label() for label in list_plot]
        ax.legend(list_plot, all_labels, fontsize=params["legend"]["fontsize"], loc=params["legend"]["loc"],
                  frameon=params["legend"]["frameon"], handletextpad=params["legend"]["pad"])  # 15

        # box = ax[1].get_position()
        # box.x0 = box.x0 - 0.055
        # box.x1 = box.x1 - 0.03
        # ax[1].set_position(box)