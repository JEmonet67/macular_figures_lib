from matplotlib import pyplot as plt


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
    def convert_data_dictionary_path_to_data_structure(data_dictionaries):
        pass

    def make_all_graphs(self, graphs_dictionaries, data_dictionaries):
        for axs_group, graphs_group_dictionaries in zip(self.axs, graphs_dictionaries):
            # Case with two-dimensional nested subfig lists.
            if isinstance(axs_group == list):
                self.make_group_graphs(axs_group, graphs_group_dictionaries, data_dictionaries)

            # Case with one-dimensional nested subfig lists.
            else:
                self.make_one_graph(axs_group, graphs_group_dictionaries, data_dictionaries)

    def make_group_graphs(self, axs_group, graphs_group_dictionaries, data_dictionaries):
        """
        Il est possible d'avoir des groupes de graphs dont certains partagent des paramètres identiques et d'autres non.
        """
        if isinstance(graphs_group_dictionaries, list):
            for ax, graph_dictionary in zip(axs_group, graphs_group_dictionaries):
                self.make_subfig_graphs(ax, graph_dictionary, data_dictionaries)
        else:
            for ax in axs_group:
                self.make_subfig_graphs(ax, graphs_group_dictionaries, data_dictionaries)

    def make_subfig_graphs(self, ax, subfig_graphs_dictionary, data_dictionaries):
        # Dictionary containing all graphs currently implemented.
        available_graphs_dict = {
            "curve": self.curves_plotting,
            "heatmap": self.heatmap_plotting,
            "barplot": self.barplots_plotting,
            "curve_video": self.curves_video_making,
            "heatmap_video": self.heatmap_video_making
        }

        available_graphs_dict[subfig_graphs_dictionary["graph_type"]](ax, subfig_graphs_dictionary, data_dictionaries)
