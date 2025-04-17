import re

import pandas as pd
import numpy as np

from src.data_manager.CoordinateManager import CoordinateManager


class MacularDictArrayConstructor:
    """Summary
MultiDataDictArray
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

    def __init__(self):
        """Summary

            Explanation

            Parameters
            ----------
            param1 : type
                Summary param1

                Explanation param1
            Returns
            ----------
            type
                Summary

            Raises
            ----------
            type
                Summary

        """
        # Extraction of the value of the number of transient frames in the simulation contained in the MacularDictArray.
        self.transient_reg = re.compile(".*/[A-Za-z]{1,2}_[A-Za-z]{1,3}_[A-Za-z]{6}[0-9]{4}_.*_([0-9]{0,4}f?)")

        # Extraction of the output, the number and the Macular cell in the name of the column of a Macular csv.
        self.output_num_celltype_reg = re.compile(r"(.*?) \(([0-9]{1,5})\) (.*)")

    @property
    def transient_reg(self):
        return self._transient_reg

    @transient_reg.setter
    def transient_reg(self, transient_reg):
        self._transient_reg = transient_reg

    @property
    def output_num_celltype_reg(self):
        return self._output_num_celltype_reg

    @output_num_celltype_reg.setter
    def output_num_celltype_reg(self, output_num_celltype_reg):
        self._output_num_celltype_reg = output_num_celltype_reg

    @staticmethod
    def crop_dataframe(dataframe, min_index, max_index):
        # Definition of maximal time.
        if max_index == "max":
            max_index = dataframe.index[-1]

        print(f"Dataframe cropping from : {min(max(dataframe.index[0], min_index), max_index)}s "
              f"to {min(dataframe.index[-1], max_index)}s")
        # Cropping of the dataframe between the minimum and maximum indicated.
        dataframe = dataframe[(dataframe.index >= min_index) & (dataframe.index <= max_index)]
        # Re-centring of the index.
        dataframe.index = dataframe.index - min_index

        return dataframe

    @staticmethod
    def init_dict_output_celltype_array(list_output_celltype):
        list_unique_output_celltype = list(set(list_output_celltype))
        print(f"Output cell type : {list_unique_output_celltype}")
        dict_output_celltype_array = {output_celltype: []
                                      for output_celltype in list_unique_output_celltype}

        return dict_output_celltype_array

    @staticmethod
    def extend_dict_output_celltype_array(dict_output_celltype_array, x, y, z):
        print(f"Initialisation of array of size {x}x{y}x{z}...", end="")
        for output_celltype in dict_output_celltype_array:
            dict_output_celltype_array[output_celltype] += [np.zeros([x, y, z])]

        return dict_output_celltype_array

    @staticmethod
    def fill_dict_output_celltype_array_chunk(dataframe_chunk, dict_output_celltype_array, list_output_celltype,
                                              list_num, n_cells, i_chunk):
        print("Filling...", end="")
        for i, num in enumerate(list_num):
            macular_coord = CoordinateManager.id_to_coordinates(num, n_cells)
            dict_output_celltype_array[list_output_celltype[i]][i_chunk][macular_coord["x"], macular_coord["y"]] = (
                dataframe_chunk.iloc[:, i].to_numpy())

        return dict_output_celltype_array

    @staticmethod
    def dict_output_celltype_array_rotation(dict_output_celltype_array, axes):
        print("Rotating...", end="")
        for key in dict_output_celltype_array:
            dict_output_celltype_array[key] = np.rot90(dict_output_celltype_array[key], axes=axes)
        print("Done!")

        return dict_output_celltype_array

    def name_extraction(self, path_data):
        """Summary

            Explanation

            Parameters
            ----------
            param1 : type
                Summary param1

                Explanation param1
            Returns
            ----------
            type
                Summary

    def transient_extraction(self, path_csv_file):
        return int(self.transient_reg.findall(path_csv_file)[0][:-1])

    def sort_macular_dataframe(self, dataframe):
        # Sorting of the temporal index according to cell type, then output and number
        print("Sorting...", end="")
        dataframe = dataframe.sort_index(axis=1, key=lambda x: [
            (self.output_num_celltype_reg.findall(elt)[0][2],
             self.output_num_celltype_reg.findall(elt)[0][0],
             int(self.output_num_celltype_reg.findall(elt)[0][1]))
            for elt in x.tolist()])
        print("Done!")

        return dataframe

    def get_list_num_output_celltype(self, path_csv_file):
        columns = pd.read_csv(path_csv_file, nrows=0).columns[1:]
        list_output_celltype, list_num = [], []

        for column in columns:
            (output, num, celltype) = self.output_num_celltype_reg.findall(column)[0]
            list_num += [int(num)]
            list_output_celltype += [f"{output}_{celltype}"]

        return list_num, list_output_celltype

