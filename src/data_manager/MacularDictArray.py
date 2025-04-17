import pickle

import numpy as np
import pandas as pd

from src.data_manager.CoordinateManager import CoordinateManager
from src.data_manager.DataPreprocessor import DataPreprocessor
from src.data_manager.MacularDictArrayConstructor import MacularDictArrayConstructor


class MacularDictArray:
    """Data container of a Macular simulation.

    The creation of a MacularDictArray requires the presence of a csv file containing the dataset of a Macular
    simulation. It also requires knowledge of some of the parameters of the Macular simulation. The data contained
    in the csv are extracted, transformed into a numpy array and separated by output pair and cell type. These data
    also undergo a rotation to correct the difference in rotation between Macular and numpy. The temporal index
    is also accessed and stored.

    Then, the dataset stored in the MacularDictArray undergoes a series of transformations determined by the
    preprocessing dictionary. The transformations include centering the dataset on the center of the
    cell receptor fields, binning, calculation of the cortical VSDI signal or a derivative. All these
    processes are managed by the DataPreprocessor class.

    Once the MacularDictArray has been created and transformed, the object is saved in a binary form. The location and
    name of the file is the same as that of the ‘.csv’ but with a ‘.pyb’ extension instead. The purpose of this PYB file
    is to speed up subsequent imports.

    When attempting to create a MacularDictArray for which the .pyb file already exists, it is the pyb file that
    will be imported instead of the csv. During this import, a comparison is made between the simulation and
    preprocessing dictionaries of the MacularDictArray saved in the pyb and those given as inputs to the initialisation
    function of the MaculaDictArray. In case of a difference, it is up to the user to decide which of the two to
    prioritise. Please note that it will also be up to the user to save the new MacularDictArray.

    Attributes
    ----------
    simulation_id : str
        Unique simulation identifier.

        This identifier corresponds to the name of the simulation file. This identifier should preferably follow the
        nomenclature defined below :
        - One or two letters to differentiate between retino-cortical (RC) or retina-only (R) simulations.
        - Three or more letters to define the name of the Macular code branch (RM: RefactoredMacular)
        - Several letters defining the type of graph used (dSG: diSymGraph).
        - Several letters defining the default parameter set (pCP: pConnecParams).
        - Simulation identification number (0026).
        - Name, value and unit of the modified parameter which is the subject of a possible comparison in a
        MultiMacularDictArray (barSpeed6dps).
        - Number of frames in the transient (0f).

        Example :
        "RC_RM_dSGpCP0026_barSpeed6dps_0f"

    dict_simulation : dict
        Dictionary containing all the parameters of the Macular simulations necessary for the processing of the
        MacularDictArray.

        The mandatory parameters are:
        - Path_data: The path to the file containing the Macular simulation data to be accessed but without csv
        extension.
        - n_cells_x: Number of cells on the x-axis of the Macular graph used.
        - n_cells_y: Number of cells on the y-axis of the Macular graph used.
        - dx: Distance between Macular cells in degrees.
        - delta_t: Time between two frames of the stimulus.
        - end: Time at which the dataset is cut in order to reduce it and remove an end transient. If this time
        is ‘max’ then no cuts will be made.
        - axis: Axis of the object's movement.

        The optional parameters are:
        - speed: Speed of the moving object in degrees/s if you want to use temporal centering.
        - size_bar: Size of the bar if you want to use temporal centering.


    dict_preprocessing : dict
        Dictionary for configuring the various processes to be implemented on the simulation dataset.

        The different keys in the processing dictionary are:
        - ‘temporal_centering’ to center the time index of each cell on the moment when the bar reaches the
        center of their receptor field. Two possible values: True and False.
        - ‘binning’ to average the data of the output-cell type over a time interval that is entered as
        the value associated with the key.
        - ‘VSDI’ to calculate the voltage sensitive dye imaging signal of the cortex. Two possible values: True and
        False.
        - ‘derivative’ to calculate the derivative of the output-cell type. It is possible to add an integer value to
        integrate over a larger interval. Otherwise, an instantaneous derivative will be obtained.

    cond : str
        Name, value and unit of the modified parameter which is the subject of a possible comparison in a
        MultiMacularDictArray.

        The extraction of this parameter requires that the simulation identifier follow the correct nomenclature.

        Example :
        "barSpeed6dps"

    data : dict of numpy.array
        Dictionary associating outputs and cell types to their 3D data array.

        Each key corresponds to the name of the output linked by an underscore to the name of the cell type.

        Example :
        "FiringRate_GanglionGainControl"

    index : dict of numpy.array
        Dictionary containing the 1D arrays of indexes before or after specific transformations.

        The dictionary has a ‘default’ key to which the temporal index is associated as it is present in the
        dataset of the accessed Macular simulation and after binning. It is possible to add other
        temporal indexes such as ‘centering’ for centered indexes where the response of each cell is centered on the moment
        of arrival of the bar in the center of their receptor field.
    """

    def __init__(self, dict_simulation, dict_preprocessing):
        """Init function to make a MacularDictArray object.

        Note the presence of a check to determine whether or not a pyb file exists, in order to import it as a priority,
        followed by a check to determine whether this pyb file can be compared with the dictionaries provided as input
        for the init function (dict_simulation, dict_preprocessing).

        Parameters
        ----------
        dict_simulation : dict
            Dictionary containing all the parameters of the Macular simulations necessary for the processing of the
            MacularDictArray.
        dict_preprocessing : dict
            Dictionary for configuring the various processes to be implemented on the simulation dataset.
        """
        dict_simulation_copy = dict_simulation.copy()
        dict_preprocessing_copy = dict_preprocessing.copy()

        if "path_pyb" not in dict_simulation_copy:
            dict_simulation_copy["path_pyb"] = dict_simulation_copy["path_csv"].replace("csv", "pyb")
        print(f"\n{dict_simulation_copy['path_pyb']}")

        self.checking_pre_existing_file(dict_simulation_copy, dict_preprocessing_copy)
        self.save()

    @property
    def path_csv(self):
        """Getter for the path_csv attribute.

        The content of the path_csv attribute can be an absolute or relative path. When accessing path_csv, if
        the path is relative it is transformed into an absolute path before being returned.
        """
        if not os.path.isabs(self._path_csv):
            return os.path.normpath(f"{os.getcwd()}/{self._path_csv}")
        else:
            return self._path_csv

    @path_csv.setter
    def path_csv(self, path_csv):
        """Setter for the path_csv attribute.
        """
        self._path_csv = path_csv

    @property
    def path_pyb(self):
        """Getter for the path_pyb attribute.

        The content of the path_pyb attribute can be an absolute or relative path. When accessing path_pyb, if
        the path is relative it is transformed into an absolute path before being returned.
        """
        if not os.path.isabs(self._path_pyb):
            return os.path.normpath(f"{os.getcwd()}/{self._path_pyb}")
        else:
            return self._path_pyb

    @path_pyb.setter
    def path_pyb(self, path_pyb):
        """Setter for the path_pyb attribute.
        """
        self._path_pyb = path_pyb

    @property
    def dict_simulation(self):
        """Getter for the dict_simulation attribute."""
        return self._dict_simulation

    @dict_simulation.setter
    def dict_simulation(self, dict_simulation):
        """Setter for the dict_simulation attribute.

        The modification of dict_simulation leads to a recomputation of the data and index attributes.
        """
        self.update_from_simulation_dict(dict_simulation)
        self.update_from_preprocessing_dict(self.dict_preprocessing)

    @property
    def dict_preprocessing(self):
        """Getter for the dict_preprocessing attribute."""
        return self._dict_preprocessing

    @dict_preprocessing.setter
    def dict_preprocessing(self, dict_preprocessing):
        """Setter for the dict_preprocessing attribute.

        The modification of dict_preprocessing leads to a recomputation of the data and index attributes.
        """
        dict_simulation_with_path = self.dict_simulation.copy()
        dict_simulation_with_path["path_csv"] = self._path_csv
        dict_simulation_with_path["path_pyb"] = self._path_pyb

        self.update_from_simulation_dict(dict_simulation_with_path)
        self.update_from_preprocessing_dict(dict_preprocessing)

    @property
    def data(self):
        """Getter for the data attribute."""
        return self._data

    @data.setter
    def data(self, data):
        """Setter for the data attribute.

        The data attribute cannot be modified.
        """
        print("WARNING : The 'data' attribute cannot be modified. Instead, please modify the simulation dictionary or "
              "the simulation id.")

    @property
    def index(self):
        """Getter for the index attribute."""
        return self._index

    @index.setter
    def index(self, index):
        """Setter for the index attribute.

        The index attribute cannot be modified.
        """
        print("WARNING : The 'index' attribute cannot be modified. Instead, please modify the simulation dictionary or "
              "the simulation id.")

    def __repr__(self):
        """Function to display a MacularDictArray.

        Example :
        ID : RC_RM_dSGpCP0026_barSpeed6dps_head100_npOnes_0f
        Cond : barSpeed6dps_head100_npOnes
        Simulation parameters : {'path_data': '/user/jemonet/home/Documents/These/Code/macular_figures_lib/tests/
        data_test/RC_RM_dSGpCP0026_barSpeed6dps_head100_npOnes_0f', 'n_cells_x': 83, 'n_cells_y': 15, 'dx': 0.225,
        'delta_t': 0.0167, 'end': 'max', 'speed': 6, 'size_bar': 0.67, 'axis': 'horizontal'}
        Preprocessing parameters : {'temporal_centering': False, 'binning': False, 'VSDI': False, 'derivative': False}
        ID : 0026
        Cond : BarSpeed6dps
        Index : numpy.array()
        Data : numpy.array()
        """

        # Generation of text to be displayed.
        str_to_display = f"Path pyb : {'/'.join(self.path_pyb.split('/')[:-1])}\n"
        str_to_display += f"ID pyb : {self.path_pyb.split('/')[-1]}\n"
        str_to_display += f"Path csv : {'/'.join(self.path_csv.split('/')[:-1])}\n"
        str_to_display += f"ID csv : {self.path_csv.split('/')[-1]}\n"
        str_to_display += f"Simulation parameters : {self.dict_simulation}\n"
        str_to_display += f"Preprocessing parameters : {self.dict_preprocessing}\n"
        str_to_display += f"Index : {self.index}\nData : {self.data}"

        return str_to_display

    @classmethod
    def equal(cls, macular_dict_array1, macular_dict_array2):
        """Checking the equality between two MacularDictArray.

        Two MacularDictArray are equal if they have the same attributes and values associated with each of these
        attributes. The simulation and preprocessing dictionaries stored in the MacularDictArray must also
        strictly have the same keys and values.

        Parameters
        ----------
        macular_dict_array1 : MacularDictArray
            First MacularDictArray to compare.

        macular_dict_array2 : MacularDictArray
            Second MacularDictArray to compare.

        Returns
        ----------
        Bool
            Returns True if both MacularDictArray are equal and False otherwise.
        """
        equality = True

        # Equality between the attributes of the two MacularDictArray.
        if macular_dict_array1.__dict__.keys() == macular_dict_array2.__dict__.keys():
            # Dictionary attributes search.
            for attributes in macular_dict_array1.__dict__:
                # Case of the data and index attributes.
                if attributes == "_data" or attributes == "_index":
                    # Equality between the measurements contained in data and index.
                    equality = equality & (cls.equal_dict_array(macular_dict_array1.__dict__[attributes],
                                                                macular_dict_array2.__dict__[attributes]))
                # Case of the path_pyb attribute, which is ignored.
                elif attributes == "_path_pyb":
                    continue
                # Case of other attributes.
                else:
                    equality = equality & (
                            macular_dict_array1.__dict__[attributes] == macular_dict_array2.__dict__[attributes])
        else:
            equality = False

        return equality

    @classmethod
    def equal_dict_array(cls, dict_array1, dict_array2):
        """Function to compare equality between dictionaries of arrays such as the data and index
        attributes of MacularDictArray

        Parameters
        ----------
        dict_array1 : dict of numpy.array
            First dict of numpy.array to compare.

        dict_array2 : dict of numpy.array
            Second dict of numpy.array to compare.

        Returns
        ----------
        Bool
            Returns True if both dict of numpy.array are equal and False otherwise.
        """
        equality = True

        if dict_array1.keys() == dict_array2.keys():
            # Equality between all the arrays of both dictionaries.
            for measurement in dict_array2:
                equality = equality & np.array_equal(dict_array1[measurement], dict_array2[measurement])
        else:
            equality = False

        return equality

    def checking_pre_existing_file(self, dict_simulation, dict_preprocessing):
        """Checks that a pyb file corresponding to the file path in the
        simulation dictionary already exists.

        If the pyb file exists, it is imported into the MacularDictArray as a priority to save time and avoid
        additional processing of the csv. If it does not exist, the csv file is processed and saved in a pyb.

        Parameters
        ----------
        dict_simulation : dict
            Dictionary containing all the parameters of the Macular simulations necessary for the processing of the
            MacularDictArray.

        dict_preprocessing : dict
            Dictionary for configuring the various processes to be implemented on the simulation dataset.
        """
        try:
            # Update MacularDictArray from an existing file if possible.
            self.update_from_file(dict_simulation['path_pyb'])

        except (FileNotFoundError, EOFError):
            print("NO FILE FOR THE UPDATE. Using the dictionaries.")
            self.update_from_simulation_dict(dict_simulation)
            self.update_from_preprocessing_dict(dict_preprocessing)

    def checking_difference_file_json(self, dict_simulation, dict_preprocessing):
        """Comparison between the simulation and preprocessing dictionary contained in the imported pyb and that
        specified in the init function of MacularDictArray.

        In case of a difference between the two, it is up to the user to choose between the dictionaries
        contained in the pyb or the one entered as input for the init function. This is a security measure.

        Parameters
        ----------
        dict_simulation : dict
            Dictionary containing all the parameters of the Macular simulations necessary for the processing of the
            MacularDictArray.

        dict_preprocessing : dict
            Dictionary for configuring the various processes to be implemented on the simulation dataset.

        Raises
        ----------
        ValueError
            The value error is raised in the event of an incorrect response from the user.
        """
        # Removal of the remaining path_data parameter
        dict_simulation_no_path = dict_simulation.copy()
        del dict_simulation_no_path["path_csv"]
        del dict_simulation_no_path["path_pyb"]

        # Checking difference between each dictionary and path_csv
        if (self.dict_simulation != dict_simulation_no_path or self.dict_preprocessing != dict_preprocessing
                or self._path_csv != dict_simulation["path_csv"]):
            print("Simulation and/or Preprocessing dictionary differ...")
            user_choice = input("Which configuration should be kept ? json or pyb : ").lower()
            # Conservation of the json file.
            if user_choice == "json":
                self.update_from_simulation_dict(dict_simulation)
                self.update_from_preprocessing_dict(dict_preprocessing)
            # Conservation of the pyb file.
            elif user_choice == "pyb":
                pass
            # Incorrect user response.
            else:
                raise ValueError("Incorrect configuration")

            return 1
        return 0

    def update_from_simulation_dict(self, dict_simulation):
        """Updating the MacularDictArray from a simulation dictionary (dict_simulation).

        The update concerns the value of the attributes dict_simulation, simulation_id, cond, data and index.

        Parameters
        ----------
        dict_simulation : dict
            Dictionary containing all the parameters of the Macular simulations necessary for the processing of the
            MacularDictArray.
        """
        self._path_pyb = dict_simulation["path_pyb"]
        self._path_csv = dict_simulation["path_csv"]
        self._dict_simulation = dict_simulation
        self._data, self._index = {}, {"default": []}
        del self._dict_simulation["path_pyb"]
        del self._dict_simulation["path_csv"]
        self._data, self._index = {}, {"temporal": [], "spatial_x": np.array([]), "spatial_y": np.array([])}
        self.setup_data_index_dict_array()

    def update_from_preprocessing_dict(self, dict_preprocessing):
        """Updating the MacularDictArray from a preprocessing dictionary (dict_preprocessing).

        The update concerns the value of the attributes dict_preprocessing, data and index.

        Parameters
        ----------
        dict_preprocessing : dict
            Dictionary for configuring the various processes to be implemented on the simulation dataset.
        """
        self._dict_preprocessing = dict_preprocessing
        self.setup_data_dict_array_preprocessing()

    def update_from_file(self, path_pyb):
        """Update of a newly created or already existing MacularDictArray object with a MacularDictArray stored in a
        pyb file.

        Parameters
        ----------
        path_pyb_file : str
            Path to the pyb file containing the dataset.
        """
        print("FILE UPDATING...", end="")
        with open(path_pyb, "rb") as pyb_file:
            tmp_dict = pickle.load(pyb_file).__dict__
        self.__dict__.clear()
        self.__dict__.update(tmp_dict)
        print("UPDATED!")

    @classmethod
    def load(cls, path_pyb):
        """Import of a MacularDictArray object that already exists and is stored in a pyb file.

        Parameters
        ----------
        path_pyb_file : str
            Path to the pyb file containing the dataset.
        """
        print("FILE LOADING...", end="")
        with open(path_pyb, "rb") as pyb_file:
            macular_dict_array = pickle.load(pyb_file)
        print("LOADED!")

        return macular_dict_array

    def save(self):
        """Saving the MacularDictArray in a pyb file whose path and name correspond to that
        present in the attribute of the simulation dictionary.
        """
        print(1)
        with open(f"{self.path_pyb}", "wb") as pyb_file:
            pickle.dump(self, pyb_file)

    def setup_data_index_dict_array(self):
        """Setting up output-cell dictionaries type associated with a dataset in the form of numpy.arrays.

        This process first requires extracting the dataset and its index from the csv of the Macular simulation. This
        extraction is done on pieces of pandas dataframe, which therefore requires concatenation at the end.
        The spatial orientation of the dataset within the numpy array is also different, which requires a
        correction.
        """
        self.setup_spatial_index("x")
        self.setup_spatial_index("y")
        self.extract_data_index_from_macular_csv()
        self.concatenate_data_index_dict_array()
        MacularDictArrayConstructor.dict_measurements_array_rotation(self.data, (0, 1))

    def extract_data_index_from_macular_csv(self):
        """Function allowing the extraction of the dataset and index contained in a Macular csv.

        The data contained in the csv Macular is read in chunks of dataframe of 2000 lines. This
        choice accelerates the reading of large datasets by parallelising them. As a result, the datasets
        and the index obtained after extraction are also subdivided and combined into a list.
        """
        print("\nData/Index extraction.")
        # Import of the data contained in the csv into a segmented dataframe.
        chunked_dataframe = pd.read_csv(self.path_csv, chunksize=2000)

        i_chunk = 0
        print("Chunk : ")
        for dataframe_chunk in chunked_dataframe:
            print(f"{i_chunk + 1}, ", end="")
            self.dataframe_chunk_processing(dataframe_chunk, i_chunk)
            i_chunk += 1

    def dataframe_chunk_processing(self, dataframe_chunk, path_csv_file, i_chunk):
        """Restructuring of a chunk of pandas dataframe into numpy array dictionaries for the index and
        the dataset.

        The dataframe is first modified so that its index is the ‘Time’ column and to remove the entire ‘transient’ part
        of the simulation, if there is one. The dictionary of the data attribute is configured to contain the output-
        cell type within keys associated with empty lists. A numpy array of size (n_cells_x, n_cells_y, size_chunk) is
        added to the list and then filled with the data from the chunk dataframe. Finally, the index of the chunk
        dataframe is added to the empty list associated with the ‘default’ key of the index attribute.

        All these operations are carried out using the MacularDictArrayConstructor class.

        Parameters
        ----------
        dataframe_chunk : pandas.io.parsers.readers.TextFileReader
            Morceau de dataframe de 2000 lignes à restructurer.

        path_csv_file : str
            Path to the csv file containing the dataset.

        i_chunk : int
            Current chunk number.
        """
        dict_array_constructor = MacularDictArrayConstructor()

        dataframe_chunk = dataframe_chunk.set_index("Time")
        dataframe_chunk = MacularDictArrayConstructor.crop_dataframe(
            dataframe_chunk, dict_array_constructor.transient_extraction(
                path_csv_file, self.dict_simulation["delta_t"]), self.dict_simulation["end"])

        list_num, list_measurements = dict_array_constructor.get_list_num_measurements(self.path_csv)

        if self._data == {}:
            self._data = MacularDictArrayConstructor.init_dict_measurements_array(list_measurements)
        MacularDictArrayConstructor.extend_dict_measurements_array(
            self.data, self.dict_simulation["n_cells_x"], self.dict_simulation["n_cells_y"],
            dataframe_chunk.shape[0])
        MacularDictArrayConstructor.fill_dict_measurements_array_chunk(
            dataframe_chunk, self.data, list_measurements, list_num,
            (self.dict_simulation["n_cells_x"], self.dict_simulation["n_cells_y"]), i_chunk)
        print("Done!")

        self.index["temporal"] += [dataframe_chunk.index.to_numpy()]

    def concatenate_data_index_dict_array(self):
        """
        Concatenation of datasets and index separated into chunks within a list.
        """
        self.index["temporal"] = np.concatenate(self.index["temporal"])
        for key in self.data:
            self.data[key] = np.concatenate(self.data[key], axis=-1)

    def setup_spatial_index(self, name_axis):
        """Function calculating the spatial index of the MacularDictArray for a given axis (x or y).

        Parameters
        ----------
        name_axis : str
            Name of the axis for which the index is to be calculated. The two possible values are ‘x’ and ‘y’.
        """
        self.index[f"spatial_{name_axis}"] = np.array([i_cell * self.dict_simulation["dx"] for i_cell in
                                                  range(self.dict_simulation[f"n_cells_{name_axis}"])])

    def setup_data_dict_array_preprocessing(self):
        """Implementation of all the procedures for transforming the dataset indicated in the dictionary of
        preprocessing.

        All these transformation processes are managed by the DataPreprocessor class.

        The different processes are:
        - ‘temporal_centering’ to center the time index of each cell on the moment when the bar reaches the
        center of their receptor field. Two possible values: True and False.
        - ‘binning’ to average the data of the output-cell type over a time interval that is entered as
        the value associated with the key.
        - ‘VSDI’ to calculate the voltage sensitive dye imaging signal of the cortex. Two possible values: True and
        False.
        - ‘derivative’ to calculate the derivative of the output-cell type. It is possible to add an integer value to
        integrate over a larger interval. Otherwise, an instantaneous derivative will be obtained.
        """
        print("Preprocessing : ", end="")
        try:
            if self.dict_preprocessing["binning"]:
                print(f"Binning {self.dict_preprocessing['binning']}s...", end="")
                bin_size, n_bin = DataPreprocessor.computing_binning_parameters(self.index["temporal"],
                                                                                self.dict_preprocessing["binning"])
                self.index["temporal"] = DataPreprocessor.binning_index(self.index["temporal"], bin_size, n_bin)
                for measurement in self.data:
                    self.data[measurement] = DataPreprocessor.binning_data_array(self.data[measurement], bin_size,
                                                                                 n_bin)
        except KeyError:
            pass

        try:
            if self.dict_preprocessing["VSDI"]:
                self.data["VSDI"] = DataPreprocessor.vsdi_computing(self.data)
        except KeyError:
            pass

        try:
            if self.dict_preprocessing["temporal_centering"]:
                print("Temporal centering...", end="")
                list_time_bar_center = CoordinateManager.get_list_time_motion_center(self.dict_simulation)
                self.index[f"temporal_centered"] = DataPreprocessor.temporal_centering(
                    self.index["temporal"], list_time_bar_center)
        except KeyError:
            pass

        # Spatial centering of x-axis index array.
        try:
            if self.dict_preprocessing["spatial_x_centering"]:
                print("Spatial x centering...", end="")
                self.index[f"spatial_x_centered"] = DataPreprocessor.spatial_centering(
                    self.index["spatial_x"], self.dict_simulation["n_cells_x"], self.dict_simulation["dx"])
        except KeyError:
            pass

        # Spatial centering of y-axis index array.
        try:
            if self.dict_preprocessing["spatial_y_centering"]:
                print("Spatial y centering...", end="")
                self.index[f"spatial_y_centered"] = DataPreprocessor.spatial_centering(
                    self.index["spatial_y"], self.dict_simulation["n_cells_y"], self.dict_simulation["dx"])
        except KeyError:
            pass

        # Computation of the array of data derivatives.
        try:
            if self.dict_preprocessing["derivative"]:
                print("Derivating...", end="")
                for measurement in self.dict_preprocessing["derivative"]:
                    self.data[f"{measurement}_derivative"] = DataPreprocessor.derivative_computing_3d_array(
                        self.data[measurement], self.index["temporal"],
                        self.dict_preprocessing["derivative"][measurement])
        except KeyError:
            pass

        print("Done!")

    def copy(self, path_pyb=""):
        """Function used to copy a MacularDictArray.

        The copy is performed deeply by also copying all the objects included in the MacularDictArray. It is also
        possible to specify a new .pyb file path to be used in the copy of the MacularDictArray.

        Parameters
        ----------
        path_pyb : str
            Path to file with .pyb extension where to save the MacularDictArray object in a binary file.

        Returns
        ----------
        macular_dict_array_copy : MacularDictArray
            Returns the copy of the current MacularDictArray.
        """
        macular_dict_array_copy = copy.deepcopy(self)

        if path_pyb:
            macular_dict_array_copy.path_pyb = path_pyb

        return macular_dict_array_copy


    @classmethod
    def make_multiple_macular_dict_array(cls, multiple_dicts_simulations, multiple_dicts_preprocessings):
        """Class method to create several MultiDictArray in succession.

        MultiDictArrays are created from dictionaries of simulation and preprocessing dictionaries. Each specific
        parameter of the MacularDictArray is associated with conditions (keys) whose names are not important here. In addition, it is
        possible to create a key named ‘global’ which allows you to group together all the parameters whose values are
        shared by all the MacularDictArray to be created.

        Parameters
        ----------
        multiple_dicts_simulations : dict of dict
            Dictionary associating simulation condition names with simulation dictionaries. The dictionary may
            also contain a ‘global’ key containing parameters shared between all
            simulations. The simulation dictionary cannot be empty or contains only a ‘global’ key. There must be
            at least one difference, such as the pyb file name.

        multiple_dicts_preprocessings : dict of dict
            Dictionary associating preprocessing condition names with preprocessing dictionaries. The dictionary may
            also contain a ‘global’ key containing parameters shared between all preprocessing of MacularDictArray. The
            preprocessing dictionary can be empty or contains only a ‘global’ key.
        """
        for condition in multiple_dicts_simulations:
            if condition != "global":
                # Loading simulation dictionary parameters shared between simulations.
                try:
                    dict_simulation = multiple_dicts_simulations["global"]
                except KeyError:
                    dict_simulation = {}

                # Loading preprocessing dictionary parameters shared between simulations.
                try:
                    dict_preprocessing = multiple_dicts_preprocessings["global"]
                except KeyError:
                    dict_preprocessing = {}

                # Loading simulation-specific simulation dictionary parameters.
                for param_simulation in multiple_dicts_simulations[condition]:
                    dict_simulation[param_simulation] = multiple_dicts_simulations[condition][param_simulation]

                # Loading preprocessing-specific simulation dictionary parameters if there is one.
                try:
                    for preprocess in multiple_dicts_preprocessings[condition]:
                        dict_preprocessing[preprocess] = multiple_dicts_preprocessings[condition][preprocess]
                except KeyError:
                    pass

                MacularDictArray(dict_simulation, dict_preprocessing)
