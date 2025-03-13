import pickle

import numpy as np
import pandas as pd

from src.data_manager.CoordinateManager import CoordinateManager
from src.data_manager.DataPreprocessor import DataPreprocessor
from src.data_manager.MacularDictArrayConstructor import MacularDictArrayConstructor


class MacularDictArray:
    """Summary

        Explanation

        Note
        ----------


        Parameters
        ----------
        param1 : type
            Summary param1
index_array
            Explanation param1

        Example
        ----------
        >> instruction
        result instruction

    """

    def __init__(self, dict_simulation, dict_preprocessing):
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
        self.checking_pre_existing_file(dict_simulation, dict_preprocessing)

        if self.dict_simulation != dict_simulation or self.dict_preprocessing != dict_preprocessing:
            print("Simulation and/or Preprocessing dictionary differ...")
            user_choice = input("Which configuration should be kept ? json or pyb : ").lower()
            if user_choice == "json":
                self.update_from_simulation_dict(dict_simulation)
                self.update_from_preprocessing_dict(dict_preprocessing)
            elif user_choice == "pyb":
                pass
            else:
                raise ValueError("Incorrect configuration")

        self.save()

    @property
    def simulation_id(self):
        return self._simulation_id

    @simulation_id.setter
    def simulation_id(self, simulation_id):
        self._simulation_id = simulation_id

    @property
    def dict_simulation(self):
        return self._dict_simulation

    @dict_simulation.setter
    def dict_simulation(self, dict_simulation):
        self.update_from_simulation_dict(dict_simulation)
        self.update_from_preprocessing_dict(self.dict_preprocessing)

    @property
    def dict_preprocessing(self):
        return self._dict_preprocessing

    @dict_preprocessing.setter
    def dict_preprocessing(self, dict_preprocessing):
        self.update_from_simulation_dict(self.dict_simulation)
        self.update_from_preprocessing_dict(dict_preprocessing)

    @property
    def cond(self):
        return self._cond

    @cond.setter
    def cond(self, cond):
        print("WARNING : The 'cond' attribute cannot be modified. Instead, please modify the simulation dictionary or "
              "the simulation id.")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        print("WARNING : The 'data' attribute cannot be modified. Instead, please modify the simulation dictionary or "
              "the simulation id.")

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        print("WARNING : The 'index' attribute cannot be modified. Instead, please modify the simulation dictionary or "
              "the simulation id.")

    def __repr__(self):
        """Function to display a MacularDictArray."""
        str_to_display = f"ID : {self.simulation_id}\nCond : {self.cond}\nSimulation parameters : {self.dict_simulation}\n"
        str_to_display += f"Preprocessing parameters : {self.dict_preprocessing}\n"
        str_to_display += f"Index : {self.index}\nData : {self.data}"

        return str_to_display

    def checking_pre_existing_file(self, dict_simulation, dict_preprocessing):
        try:
            print(dict_simulation['path_data'])
            self.update_from_file(f"{dict_simulation['path_data']}.pyb")
            # TODO Ajouter un moyen de mettre à jour le fichier config si pyb est choisi?
        except (FileNotFoundError, EOFError):
            print("NO FILE FOR THE UPDATE. Using the dictionaries.")
            self.update_from_simulation_dict(dict_simulation)
            self.update_from_preprocessing_dict(dict_preprocessing)

    def update_from_simulation_dict(self, dict_simulation):
        self._simulation_id = dict_simulation["path_data"].split("/")[-1]
        self._dict_simulation = dict_simulation
        dict_array_constructor = MacularDictArrayConstructor()
        self._cond = dict_array_constructor.name_extraction(dict_simulation["path_data"])
        self._data, self._index = {}, {"default": []}
        self.set_data_index_dict_array()

    def update_from_preprocessing_dict(self, dict_preprocessing):
        self._dict_preprocessing = dict_preprocessing
        self.setup_data_dict_array_preprocessing()

    def update_from_file(self, path_pyb_file):
        """Update of a newly created or already existing MacularDictArray object
        with a MacularDictArray stored in a .pyb file."""
        print("FILE UPDATING...", end="")
        with open(path_pyb_file, "rb") as pyb_file:
            tmp_dict = pickle.load(pyb_file).__dict__
        self.__dict__.clear()
        self.__dict__.update(tmp_dict)
        print("UPDATED!")

    @classmethod
    def load(cls, path_pyb_file):
        """Import of a MacularDictArray object that already exists and is stored in a .pyb file."""
        print("FILE LOADING...", end="")
        with open(path_pyb_file, "rb") as pyb_file:
            macular_dict_array = pickle.load(pyb_file)
        print("LOADED!")

        return macular_dict_array

    def save(self):
        with open(f"{self.dict_simulation['path_data']}.pyb", "wb") as pyb_file:
            pickle.dump(self, pyb_file)

    def set_data_index_dict_array(self):
        self.extract_data_index_from_macular_csv()
        self.concatenate_data_index_dict_array()
        MacularDictArrayConstructor.dict_output_celltype_array_rotation(self.data, (0, 1))

    def extract_data_index_from_macular_csv(self):
        print("Data/Index extraction.")
        path_csv_file = f"{self.dict_simulation['path_data']}.csv"
        chunked_dataframe = pd.read_csv(path_csv_file, chunksize=2000)

        i_chunk = 0
        print("Chunk : ")
        for dataframe_chunk in chunked_dataframe:
            print(f"{i_chunk + 1}, ", end="")
            self.dataframe_chunk_processing(dataframe_chunk, path_csv_file, i_chunk)
            i_chunk += 1

    def dataframe_chunk_processing(self, dataframe_chunk, path_csv_file, i_chunk):
        dict_array_constructor = MacularDictArrayConstructor()

        dataframe_chunk = dataframe_chunk.set_index("Time")
        dataframe_chunk = MacularDictArrayConstructor.crop_dataframe(
            dataframe_chunk, dict_array_constructor.transient_extraction(
                path_csv_file, self.dict_simulation["delta_t"]), self.dict_simulation["end"])

        list_num, list_output_celltype = dict_array_constructor.get_list_num_output_celltype(path_csv_file)

        if self._data == {}:
            self._data = MacularDictArrayConstructor.init_dict_output_celltype_array(list_output_celltype)

        MacularDictArrayConstructor.extend_dict_output_celltype_array(
            self.data, self.dict_simulation["n_cells_x"], self.dict_simulation["n_cells_y"],
            dataframe_chunk.shape[0])
        MacularDictArrayConstructor.fill_dict_output_celltype_array_chunk(
            dataframe_chunk, self.data, list_output_celltype, list_num,
            (self.dict_simulation["n_cells_x"], self.dict_simulation["n_cells_y"]), i_chunk)
        print("Done!")

        self.index["default"] += [dataframe_chunk.index.to_numpy()]

    def concatenate_data_index_dict_array(self):
        self.index["default"] = np.concatenate(self.index["default"])
        for key in self.data:
            self.data[key] = np.concatenate(self.data[key], axis=-1)

    def setup_data_dict_array_preprocessing(self):
        print("Preprocessing : ", end="")
        try:
            if self.dict_preprocessing["binning"]:
                bin_size, n_bin = DataPreprocessor.computing_binning_parameters(self.index["default"],
                                                                                self.dict_preprocessing["binning"])
                self.index["default"] = DataPreprocessor.binning_index(self.index["default"], bin_size, n_bin)
                for output in self.data:
                    self.data[output] = DataPreprocessor.binning_data_array(self.data[output], bin_size, n_bin)
        except KeyError:
            pass

        try:
            if self.dict_preprocessing["VSDI"]:
                self.data["VSDI"] = DataPreprocessor.vsdi_computing(self.data)
        except KeyError:
            pass

        try:
            if self.dict_preprocessing["temporal_centering"]:
                list_time_bar_center = CoordinateManager.get_list_time_bar_center(self.dict_simulation)
                self.index[f"centered"] = DataPreprocessor.temporal_centering(
                    self.index["default"], list_time_bar_center)
        except KeyError:
            pass

        if self.dict_preprocessing["derivative"]:
            for output in self.dict_preprocessing["derivative"]:
                self.data[f"{output}_derivative"] = DataPreprocessor.derivative_computing_3d_array(
                    self.data[output], self.index["default"], self.dict_preprocessing["derivative"][output])

        print("Done!")
