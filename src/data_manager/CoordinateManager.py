import math

import numpy as np


class CoordinateManager:
    """Class containing all the functions enabling the management of different coordinate systems.
    """

    @staticmethod
    def id_to_coordinates(num, n_cells):
        """Conversion of a Macular identification number of a cell in the Macular coordinate system.

        Parameters
        ----------
        num : int
            Macular identification number of the cell whose coordinates are to be determined in Macular.

        n_cells : tuple
            Size of the Macular graph in cells in the form of : (number of cells in x, number of cells in y).

        Returns
        ----------
        dict_coordinates : dict
            Dictionary containing the Macular coordinates of the cell named by the Macular identification number
            given as input. The dictionary contains an ‘x’ and a ‘y’ key.
        """

        dict_coordinates = {}

        # Calculation of the x, y and z coordinates.
        z = num / n_cells[0] / n_cells[1]
        x = (z - math.floor(z)) * n_cells[0]
        y = (x - math.floor(round(x, 2))) * n_cells[1]

        # Construction of the Macular coordinate dictionary.
        dict_coordinates["z"] = math.floor(z)
        dict_coordinates["x"] = math.floor(round(x, 2))
        dict_coordinates["y"] = abs(math.floor(round(y, 2)))

        return dict_coordinates

    @staticmethod
    def get_list_time_motion_center(dict_simulation):
        """Creation of a list containing all the times at which a moving object arrives at each of the cells of the
        horizontal or vertical axis.

        Parameters
        ----------
        dict_simulation : dict
            Dictionary containing all the parameters of the Macular simulations necessary for the processing of a
            MacularDictArray.

        Returns
        ----------
        list_time_bar_center : list of float
            List of the times at which the moving object reaches the centre of each of the cells of the horizontal or
            vertical axis.
        """
        # Horizontal stimulus.
        if dict_simulation["axis"] == "horizontal":
            last_cell = dict_simulation["n_cells_x"]

        # Vertical stimulus.
        elif dict_simulation["axis"] == "vertical":
            last_cell = dict_simulation["n_cells_y"]

        # Computing the arrival times of the bar in the centre of the cell receptor field.
        list_time_bar_center = [
            np.round(((np.round(x_col * dict_simulation["dx"], 5) +
                       dict_simulation["size_bar"] / 2) / dict_simulation["speed"]), 5)
            for x_col in range(0, last_cell)]

        return list_time_bar_center

    @staticmethod
    def convert_coord_macular_to_coord_numpy(dict_coord_macular, n_cells):
        """Function to transform the coordinate of a cell in the Macular coordinate system to that of Numpy.

        The Macular coordinate system differs from the Numpy coordinate system by a 90° clockwise rotation.

        Parameters
        ----------
        dict_coord_macular : dict
            Dictionary containing the Macular coordinates to be transformed. The dictionary contains an ‘x’ and a ‘y’
            key.

        n_cells : tuple
            Size of the Macular graph in cells in the form of : (number of cells in x, number of cells in y).

        Returns
        ----------
        dict_coord_numpy : dict
            Dictionary containing the Numpy coordinates of the cell whose Macular coordinates have been provided.
            The dictionary contains an ‘x’ and a ‘y’ key.
        """
        dict_coord_numpy = {}

        dict_coord_numpy["x"] = (n_cells[1] - 1) - dict_coord_macular[1]
        dict_coord_numpy["y"] = dict_coord_macular[0]

        return dict_coord_numpy

    @staticmethod
    def convert_coord_numpy_to_coord_macular(dict_coord_numpy, n_cells):
        """Function to transform the coordinate of a cell in the Numpy coordinate system to that of Coordinates.

        The Numpy coordinate system differs from the Macular coordinate system by a 90° anti-clockwise rotation.

        Parameters
        ----------
        dict_coord_numpy : dict
            Dictionary containing the Numpy coordinates to be transformed. The dictionary contains an ‘x’ and a ‘y’
            key.

        n_cells : tuple
            Size of the Macular graph in cells in the form of : (number of cells in x, number of cells in y).

        Returns
        ----------
        dict_coord_macular : dict
            Dictionary containing the Macular coordinates of the cell whose Numpy coordinates have been provided.
            The dictionary contains an ‘x’ and a ‘y’ key.
        """
        dict_coord_macular = {}

        dict_coord_macular["x"] = dict_coord_numpy[1]
        dict_coord_macular["y"] = (n_cells[0] - 1) - dict_coord_numpy[0]

        return dict_coord_macular
