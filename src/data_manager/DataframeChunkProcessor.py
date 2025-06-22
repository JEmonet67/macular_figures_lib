import re

import pandas as pd
import numpy as np

from src.data_manager.CoordinateManager import CoordinateManager


class DataframeChunkProcessor:
    def __init__(self):
        # Extraction of the output, the number and the Macular cell in the name of the column of a Macular csv.
        self.output_num_celltype_reg = re.compile(r"(.*?) \(([0-9]{1,5})\) (.*)")

    @property
    def output_num_celltype_reg(self):
        return self._output_num_celltype_reg

    @output_num_celltype_reg.setter
    def output_num_celltype_reg(self, output_num_celltype_reg):
        self._output_num_celltype_reg = output_num_celltype_reg

    def get_list_num_measurements(self, path_csv_file):
        columns = pd.read_csv(path_csv_file, nrows=0).columns[1:]
        list_measurements, list_num = [], []

        # Go through column names of the Macular dataframe.
        for column in columns:
            # Extraction of outputs, numbers and cell type.
            (output, num, celltype) = self.output_num_celltype_reg.findall(column)[0]

            # Incrementation of the lists of number and measurements (output_celltype).
            list_num += [int(num)]
            list_measurements += [f"{output}_{celltype}"]

        return list_num, list_measurements

    @staticmethod
    def init_dict_measurements_array(list_measurements):

        # Create the measurements list without duplicates
        list_unique_measurements = list(set(list_measurements))
        print(f"Measurements : {list_unique_measurements}")

        # Initialisation of the empty list dictionary of measurements.
        dict_measurements_array = {measurements: []
                                      for measurements in list_unique_measurements}

        return dict_measurements_array

    @staticmethod
    def extend_dict_measurements_array(dict_measurements_array, x, y, z):
        print(f"Implementation of array of size {x}x{y}x{z}...", end="")
        for measurements in dict_measurements_array:
            dict_measurements_array[measurements] += [np.zeros([x, y, z])]

        return dict_measurements_array

    @staticmethod
    def fill_dict_measurements_array_chunk(dataframe_chunk, dict_measurements_array, list_measurements,
                                              list_num, n_cells, i_chunk):
        print("Filling...", end="")
        for i, num in enumerate(list_num):
            # Conversion of the Macular identification number into Macular coordinates
            macular_coord = CoordinateManager.id_to_coordinates(num, n_cells)
            # Insert the 3D array of the given measurement and the given cell coordinates in the measurement dictionary.
            dict_measurements_array[list_measurements[i]][i_chunk][macular_coord["x"], macular_coord["y"]] = (
                dataframe_chunk.iloc[:, i].to_numpy())

        return dict_measurements_array

    @staticmethod
    def dict_measurements_array_rotation(dict_measurements_array, axes):
        print("Rotating...", end="")
        for key in dict_measurements_array:
            dict_measurements_array[key] = np.rot90(dict_measurements_array[key], axes=axes)
        print("Done!")

        return dict_measurements_array


