import math

import numpy as np


class CoordinateManager:
    """Summary

        Explanation

        Note
        ----------


        Parameters
        ----------
        param1 : type
            Summary param1

            Explanation param1

        Example
        ----------
        >> instruction
        result instruction

    """

    @staticmethod
    def id_to_coordinates(num, n_cells):
        dict_coordinates = {}

        z = num / n_cells[0] / n_cells[1]
        x = (z - math.floor(z)) * n_cells[0]
        y = (x - math.floor(round(x, 2))) * n_cells[1]

        dict_coordinates["z"] = math.floor(z)
        dict_coordinates["x"] = math.floor(round(x, 2))
        dict_coordinates["y"] = abs(math.floor(round(y, 2)))

        return dict_coordinates

    @staticmethod
    def get_list_time_bar_center(dict_simulation):
        """Function listing the times at which the centre of the stimulus passes through the centre of each cell."""
        # Stimulus horizontal
        if dict_simulation["axis"] == "horizontal":
            last_cell = dict_simulation["n_cells_x"]

        # Stimulus vertical
        elif dict_simulation["axis"] == "vertical":
            last_cell = dict_simulation["n_cells_y"]

        list_time_bar_center = [
            np.round(((np.round(x_col * dict_simulation["dx"], 5) +
                       dict_simulation["size_bar"] / 2) / dict_simulation["speed"]), 5)
            for x_col in range(0, last_cell)]

        return list_time_bar_center
