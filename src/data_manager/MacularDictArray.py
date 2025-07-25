import os.path
import pickle
import copy
import re

import numpy as np
import pandas as pd

from src.data_manager.CoordinateManager import CoordinateManager
from src.data_manager.DataPreprocessor import DataPreprocessor
from src.data_manager.DataframeHelpers import DataframeHelpers
from src.data_manager.DataframeChunkProcessor import DataframeChunkProcessor


class MacularDictArray:
    """Data container of a Macular simulation.

    The creation of a MacularDictArray requires the presence of a csv file containing the measurements of a Macular
    simulation. It also requires knowledge of some of the parameters of the Macular simulation. The measurements
    contained in the csv are parsed into a numpy arrays and separated by pair of output and cell type.
    These measurements also undergo a rotation to correct the difference in rotation between Macular and numpy. The
    temporal index is also accessed and stored.

    Then, the data stored in the MacularDictArray undergoes a series of transformations determined by the
    preprocessing dictionary. The transformations include centering the data on the center of the cell receptor fields,
    centering spatial indexes, binning, calculation of the cortical VSDI signal or a derivative. All these processes are
    managed by the DataPreprocessor class.

    Once the MacularDictArray has been created and transformed, the object is saved in a binary form. The location and
    name of the file is the same as that of the ‘.csv’ but with a ‘.pyb’ extension instead. The purpose of this PYB file
    is to speed up subsequent imports.

    When attempting to create a MacularDictArray for which the .pyb file already exists, it is the pyb file that
    will be imported instead of the csv. During this import, a comparison is made between the simulation and
    preprocessing dictionaries of the MacularDictArray saved in the pyb and those given as inputs to the initialisation
    function of the MaculaDictArray. In case of a difference, it is up to the user to decide which of the two to
    prioritise. Please note that it will also be up to the user to save the new MacularDictArray.

    The name of the csv file used to create a MacularDictArray can be a unique identifier that respects the following
    recommended nomenclature :
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

    The MacularDictArray class has been designed to allow users to add their own processes to prepare data for analysis.
    To do this, simply add a new key:value pair to the preprocessing dictionary and then create a new function that
    performs the process and will be called in the setup_data_dict_array_preprocessing function. The user must also pay
    attention to where this process is placed in the order of processes executed.

    Attributes
    ----------
    path_csv : str
        The path of the csv file containing the Macular simulation data to be accessed.

        The path can be absolute or relative.

        Example :
        "/user/jemonet/macular_figures_lib/example/RC_RM_dSGpCP0026_barSpeed6dps_0f.csv"

    path_pyb : str
        Path to file with .pyb (python binary) extension where to save the MacularDictArray object in a binary file.

        The path can be absolute or relative. If the .pyb file already exists, it will be possible to import it directly
        instead of processing the csv file again. The path_pyb attribute can differ between two MacularDictArray because
        it only affects the save location, not the processing or the data.

        Example :
        "/user/jemonet/macular_figures_lib/example/RC_RM_dSGpCP0026_barSpeed6dps_0f.pyb"

    dict_simulation : dict
        Dictionary containing all the parameters of the Macular simulations necessary for the processing of the
        MacularDictArray.

        This is a copy of the simulation dictionary used to create the MacularDictArray. This copy differs
        in that the file paths path_csv and path_pyb are absent. For more information on the other parameters,
        refer to the description of the __init__ function.

    dict_preprocessing : dict
        Dictionary for configuring the various processes to be implemented on the simulation data.

        This is a copy of the preprocessing dictionary used to create the MacularDictArray. For more information on the
        different parameters, refer to the description of the __init__ function.

    data : dict of numpy.array
        Dictionary associating measurements (outputs-cell types) to their 3D data array.

        Each key corresponds to the name of the output linked by an underscore to the name of the cell type. This rule
        primarily concerns measurements made by Macular and may differ for measurements calculated during preprocessing.
        This is particularly the case for VSDI, which is simply called ‘VSDI’ because it is associated with two
        different cell types.

        Example :
        "FiringRate_GanglionGainControl"

    index : dict of numpy.array
        Dictionary containing the 1D arrays of indexes before or after specific transformations.

        The index dictionary can contain the following index keys :
        - 'temporal' (default key) in which the temporal index (in second) is associated as it is
        present in the data of the accessed Macular simulation and after binning
        - ‘spatial_x’ in degrees (default key) which is the indexes for the x-axis.
        - ‘spatial_y’ in degrees (default key) which is the indexes for the y-axis.
        - 'temporal_centered' (optional) for centered time indexes where the response of each cell is centered on the
        moment of arrival of the bar in the center of their receptor field.
        - ‘spatial_x_centered’ in degrees (default key) which is the indexes for the x-axis centered on the middle cell
        of the x-axis of the grid cell.
        - ‘spatial_y_centered’ in degrees (default key) which is the indexes for the y-axis  centered on the middle cell
        of the y-axis of the grid cell.
        - 'temporal_ms' is the temporal index in milliseconds.
        - 'spatial_x_mm_retina' is the spatial x-axis index in millimeters of retina.
        - 'spatial_y_mm_retina' is the spatial y-axis index in millimeters of retina.
        - 'spatial_x_mm_cortex' is the spatial x-axis index in millimeters of cortex.
        - 'spatial_y_mm_cortex' is the spatial y-axis index in millimeters of cortex.
        - 'temporal_centered_ms' is the centered temporal index in milliseconds.
        - 'spatial_x_centered_mm_retina' is the centered spatial x-axis index in millimeters of retina.
        - 'spatial_y_centered_mm_retina' is the centered spatial y-axis index in millimeters of retina.
        - 'spatial_x_centered_mm_cortex' is the centered spatial x-axis index in millimeters of cortex.
        - 'spatial_y_centered_mm_cortex' is the centered spatial y-axis index in millimeters of cortex.

        It is possible to add any new index based on conversion of temporal or spatial indexes.

    transient_reg : re.Pattern
        Regular expression to extract the value of the number of transient frames in the Macular simulation path if
        the name of the simulation follow the recommended nomenclature.
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

            The mandatory parameters are:
            - path_csv : The path of the csv file containing the Macular simulation data to be accessed.
            - n_cells_x: Number of cells on the x-axis of the Macular graph used.
            - n_cells_y: Number of cells on the y-axis of the Macular graph used.
            - dx: Distance between Macular cells in degrees.
            - delta_t: Time between two frames of the stimulus.
            - end: Time at which the data is cut in order to remove a transient at the beginning and a part at the end.
            If this time is ‘max’ then no cuts will be made.

            The optional parameters are:
            - path_pyb : Path to file with .pyb (python binary) extension where to save the MacularDictArray object in a
            binary file.
            - speed: Speed of the moving object in degrees/s if you want to use temporal centering.
            - size_bar: Size of the bar if you want to use temporal centering.
            - transient: Duration at the start of the simulation to be removed, it can be in number of frames or in
            seconds.
            - axis: Axis of the object's movement ("horizontal" or "vertical") if there is one.

            Note : It is possible to enter a simulation dictionary containing only the optional parameter ‘path_pyb’
            in order to import the pre-existing pyb file without having to specify all the parameter values. In this
            case, no comparison is made between the json and the pyb file.

        dict_preprocessing : dict
            Dictionary for configuring the various processes to be implemented on the simulation data.

            The different keys in the processing dictionary are:
            - ‘temporal_centering’ to center the time index of each cell on the moment when the bar reaches the
            center of their receptor field. Two possible values: True and False.
            - 'spatial_x_centering' to center the x-axis index on the center of the grid cell length. Two possible
            values: True and False.
            - 'spatial_y_centering' to center the y-axis index on the center of the grid cell width. Two possible
            values: True and False.
            - ‘binning’ to average the data of the measurements over a time interval that is entered as
            the value associated with the key.
            - ‘VSDI’ to calculate the voltage sensitive dye imaging signal of the cortex. Two possible values: True and
            False.
            - ‘derivative’ to calculate the derivative of the measurements. It is possible to add an integer value
            to integrate over a larger interval. Otherwise, an instantaneous derivative will be obtained. The
            ‘derivative’ key is associated with a dictionary that must contain the measurements to be processed as keys
            and the size of the derivative interval as a value.
            - 'edge' to crop the edges of arrays of all measurements in MacularDictArray. The value can be a tuple to
            crop differently in horizontal and vertical axes: (x_edge, y_edge) or an int to crop everywhere the same.
            The x_edge and the y_edge of the tuple can also be tuples to crop asymmetrically the two edges of each axis.
            - 'temporal_index_ms' to add a temporal index expressed in milliseconds. The value is the ratio to convert,
            so 1000.
            - 'spatial_index_mm_retina' to add a spatial index expressed in mm for retina. The value is the ratio to
            convert degrees in millimetres of retina.
            - 'spatial_index_mm_cortex' to add a spatial index expressed in mm for cortex. The value is the ratio to
            convert degrees in millimetres of cortex.
        """
        self._transient_reg = re.compile(".*/[A-Za-z]{1,2}_[A-Za-z]{1,3}_[A-Za-z]{6}[0-9]{4}_.*_([0-9]{0,4}f?)")

        dict_simulation_copy = dict_simulation.copy()
        dict_preprocessing_copy = dict_preprocessing.copy()
        dict_preprocessing_copy = self.cleaning_dict_preprocessing(dict_preprocessing_copy)


        if "path_pyb" not in dict_simulation_copy:
            dict_simulation_copy["path_pyb"] = dict_simulation_copy["path_csv"].replace("csv", "pyb")
        print(f"\n{dict_simulation_copy['path_pyb']}")

        self.managing_pre_existing_file(dict_simulation_copy, dict_preprocessing_copy)
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

        Raises
        ----------
        AttributeError
            The attribute error is raised if the user try to modify data.
        """
        raise AttributeError("The attribute data can't be modified.")

    @property
    def index(self):
        """Getter for the index attribute."""
        return self._index

    @index.setter
    def index(self, index):
        """Setter for the index attribute.

        The index attribute cannot be modified.

        Raises
        ----------
        AttributeError
            The attribute error is raised if the user try to modify index.
        """
        raise AttributeError("The attribute index can't be modified.")

    @property
    def transient_reg(self):
        """Getter for the transient_reg attribute."""
        return self._transient_reg

    @transient_reg.setter
    def transient_reg(self, transient_reg):
        """Setter for the transient_reg attribute."""
        self._transient_reg = transient_reg

    def __repr__(self):
        """Function to display a MacularDictArray.

        Example :
        Path pyb : /home/jemonet/Documents/These/Code/macular_figures_lib/tests/data_test/data_manager
        ID pyb : RC_RM_dSGpCP0026_barSpeed6dps_head100_npOnes_0f.pyb
        Path csv : /home/jemonet/Documents/These/Code/macular_figures_lib/tests/data_test/data_manager
        ID csv : RC_RM_dSGpCP0026_barSpeed6dps_head100_npOnes_0f.csv
        Simulation parameters : {'n_cells_x': 83, 'n_cells_y': 15, 'dx': 0.225, 'delta_t': 0.0167, 'end': 'max',
        'speed': 6, 'size_bar': 0.67, 'axis': 'horizontal'}
        Preprocessing parameters : {}
        Index : {"temporal": numpy.array(), "spatial_x": numpy.array(), "spatial_y": numpy.array()}
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
        attributes. Only the path_pyb attribute can differ between the two MacularDictArrays. The simulation and
        preprocessing dictionaries stored in the MacularDictArray must also strictly have the same keys and values.

        Parameters
        ----------
        macular_dict_array1 : MacularDictArray
            First MacularDictArray to compare.

        macular_dict_array2 : MacularDictArray
            Second MacularDictArray to compare.

        Returns
        ----------
        equality : Bool
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
        equality : Bool
            Returns True if both dict of numpy.array are equal and False otherwise.
        """
        equality = True

        if dict_array1.keys() == dict_array2.keys():
            # Equality between all the arrays of both dictionaries.
            for measurement in dict_array2:
                equality = equality & np.array_equal(dict_array1[measurement], dict_array2[measurement],
                                                     equal_nan=True)
        else:
            equality = False

        return equality

    @staticmethod
    def cleaning_dict_preprocessing(dict_preprocessing):
        """Cleans the preprocessing dictionary by removing all keys associated with a value of False.

        The purpose of this cleanup is to take into account that preprocesses missing from the preprocess dictionary are
        equivalent to preprocesses that are present but with a value set to False.

        Parameters
        ----------
        dict_preprocessing : dict
            Dictionary for configuring the various processes to be implemented on the simulation data.

        Returns
        ----------
        dict_preprocessing_cleaned : dict
            Preprocessing dictionary with no keys associated with False values.
        """
        dict_preprocessing_cleaned = dict_preprocessing.copy()

        for preprocess in dict_preprocessing:
            if not dict_preprocessing[preprocess]:
                del dict_preprocessing_cleaned[preprocess]

        return dict_preprocessing_cleaned

    def managing_pre_existing_file(self, dict_simulation, dict_preprocessing):
        """Managing that a pyb file corresponding to the file path in the
        simulation dictionary already exists.

        If the pyb file exists, it is imported into the MacularDictArray as a priority to save time and avoid
        additional processing of the csv. If it does not exist, the csv file is processed and saved in a pyb.

        Parameters
        ----------
        dict_simulation : dict
            Dictionary containing all the parameters of the Macular simulations necessary for the processing of the
            MacularDictArray.

        dict_preprocessing : dict
            Dictionary for configuring the various processes to be implemented on the simulation data.
        """
        try:
            # Update MacularDictArray from an existing file if possible.
            self.update_from_file(dict_simulation['path_pyb'])

            # The comparison with json only occurs if the simulation dictionary does not contain only the pyb path.
            if len(dict_simulation.keys()) > 1:
                self.checking_difference_file_json(dict_simulation, dict_preprocessing)

        except (FileNotFoundError, EOFError):
            # Construction of a MacularDictArray from the dictionaries if no file exists.
            print("NO FILE FOR THE UPDATE. Using the dictionaries.")
            self.update_from_simulation_dict(dict_simulation)
            self.update_from_preprocessing_dict(dict_preprocessing)

    def checking_difference_file_json(self, dict_simulation, dict_preprocessing):
        """Comparison between the simulation and preprocessing dictionary contained in the imported pyb and that
        specified in the init function of MacularDictArray.

        The verification is carried out on the elements of the simulation and processing dictionaries, but also on
        the path of the csv file. In case of a difference between the two, it is up to the user to choose between the
        dictionaries contained in the pyb or the one entered as input for the init function.

        Parameters
        ----------
        dict_simulation : dict
            Dictionary containing all the parameters of the Macular simulations necessary for the processing of the
            MacularDictArray.

        dict_preprocessing : dict
            Dictionary for configuring the various processes to be implemented on the simulation data.

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

        The update concerns the value of the attributes dict_simulation, simulation_id, data and index.

        Parameters
        ----------
        dict_simulation : dict
            Dictionary containing all the parameters of the Macular simulations necessary for the processing of the
            MacularDictArray.
        """
        self._path_pyb = dict_simulation["path_pyb"]
        self._path_csv = dict_simulation["path_csv"]
        self._dict_simulation = dict_simulation
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
            Dictionary for configuring the various processes to be implemented on the simulation data.
        """
        self._dict_preprocessing = dict_preprocessing
        self._dict_preprocessing = self.cleaning_dict_preprocessing(dict_preprocessing)

        self.setup_data_dict_array_preprocessing()

    def update_from_file(self, path_pyb):
        """Update of a newly created or already existing MacularDictArray object with a MacularDictArray stored in a
        pyb file.

        Parameters
        ----------
        path_pyb : str
            Path to file with .pyb extension where a MacularDictArray object is saved in binary.

            The path can be absolute or relative.

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
        path_pyb : str
            Path to file with .pyb extension where a MacularDictArray object is saved in binary.

            The path can be absolute or relative.        """
        print("FILE LOADING...", end="")
        with open(path_pyb, "rb") as pyb_file:
            macular_dict_array = pickle.load(pyb_file)
        print("LOADED!")

        return macular_dict_array

    def save(self):
        """Saving the MacularDictArray in a pyb (python binary) file whose path and name correspond to that
        present in the attribute of the simulation dictionary.
        """
        with open(f"{self.path_pyb}", "wb") as pyb_file:
            pickle.dump(self, pyb_file)

    def setup_data_index_dict_array(self):
        """Setting up measurements (output-cell type) dictionaries with data in the form of numpy.arrays.

        This process first requires extracting the data and its index from the csv of the Macular simulation. This
        extraction is done on pieces of pandas dataframe, which therefore requires concatenation at the end.
        The spatial orientation of the data within the numpy array is also different, which requires a
        correction.
        """
        self.setup_spatial_index("x")
        self.setup_spatial_index("y")
        self.extract_data_index_from_macular_csv()
        self.concatenate_data_index_dict_array()
        DataframeChunkProcessor.dict_measurements_array_rotation(self.data, (0, 1))

    def extract_data_index_from_macular_csv(self):
        """Function allowing the extraction of the data and index contained in a Macular csv.

        The data contained in the csv Macular is read in chunks of dataframe of 2000 lines. This
        choice accelerates the reading of large datasets by parallelising them. As a result, the datasets
        and the index obtained after extraction are also subdivided and combined into a list.
        """
        print("\nData/Index extraction.")
        # Import of the data contained in the csv into a segmented dataframe.
        chunked_dataframe = pd.read_csv(self.path_csv, chunksize=2000)

        i_chunk = 0
        print("Chunk : ")
        # Processing of data frame segments
        for dataframe_chunk in chunked_dataframe:
            print(f"{i_chunk + 1}, ", end="")
            self.dataframe_chunk_processing(dataframe_chunk, i_chunk)
            i_chunk += 1

    def dataframe_chunk_processing(self, dataframe_chunk, i_chunk):
        """Restructuring of a chunk of pandas dataframe into numpy array dictionaries for the index and
        the data.

        The dataframe is first modified so that its index is the ‘Time’ column and to remove the entire ‘transient’ part
        of the simulation, if there is one. The dictionary of the data attribute is configured to contain the
        measurements names within keys associated with empty lists. A numpy array of size (n_cells_x, n_cells_y,
        size_chunk) is added to the list and then filled with the data from the chunk dataframe. Finally, the index of
        the chunk dataframe is added to the empty list associated with the ‘temporal’ key of the index attribute.

        All these operations are carried out using the DataframeChunkProcessor class.

        Parameters
        ----------
        dataframe_chunk : pandas.io.parsers.readers.TextFileReader
            Portion of a dataframe of 2000 lines to be restructured.

        i_chunk : int
            Current chunk number.
        """
        dataframe_chunk_processor = DataframeChunkProcessor()

        # Transient computing
        transient = self.transient_computing()
        # Shaping of the dataframe fragment.
        dataframe_chunk = dataframe_chunk.set_index("Time")
        dataframe_chunk = DataframeHelpers.crop_dataframe_rows(dataframe_chunk, transient,
                                                                self.dict_simulation["end"])

        list_num, list_measurements = dataframe_chunk_processor.get_list_num_measurements(self.path_csv)

        # Implementation of data and index arrays.
        if self._data == {}:
            self._data = DataframeChunkProcessor.init_dict_measurements_array(list_measurements)
        DataframeChunkProcessor.extend_dict_measurements_array(
            self.data, self.dict_simulation["n_cells_x"], self.dict_simulation["n_cells_y"],
            dataframe_chunk.shape[0])
        DataframeChunkProcessor.fill_dict_measurements_array_chunk(
            dataframe_chunk, self.data, list_measurements, list_num,
            (self.dict_simulation["n_cells_x"], self.dict_simulation["n_cells_y"]), i_chunk)
        print("Done!")

        self.index["temporal"] += [dataframe_chunk.index.to_numpy()]

    def transient_computing(self):
        """Function to calculate the value of the transient to be removed from the data set.

        The transient is taken primarily from the simulation dictionary. If it is not defined there, we get the
        transient value from the file name. If it is not in either of these, it takes the default value (0). In the
        simulation dictionary, the transient can be given in seconds or frames. In both cases, to specify, the transient
        value is supplemented with the letters s (seconds) or f (frames).

        Returns
        ----------
        transient : float
            Returns the value of the calculated transient.
        """

        transient = 0

        # Case of the transient obtained in the simulation dictionary.
        if "transient" in self.dict_simulation:
            transient = float(self.dict_simulation["transient"][:-1])
            if self.dict_simulation["transient"][-1] == "f":
                transient = transient * self.dict_simulation["delta_t"]

        # Case of the transient extracted from the name of the csv file.
        elif (self.transient_reg.findall(self.path_csv) and
              self.transient_reg.findall(self.path_csv) != [""]):
            transient = self.transient_extraction() * self.dict_simulation["delta_t"]

        return transient

    def transient_extraction(self):
        """
        Extracting the transient value from the CSV file path if it follows the nomenclature.
        """
        return int(self.transient_reg.findall(self.path_csv)[0][:-1])

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
                                                       range(self.dict_simulation[f"n_cells_{name_axis}"])]).round(5)

    def setup_data_dict_array_preprocessing(self):
        """Implementation of all the procedures for transforming the data indicated in the dictionary of
        preprocessing.

        All these transformation processes are managed by the DataPreprocessor class. The processes are carried out in a
        predefined order to ensure they run smoothly.

        The different processes are in the order :
        - ‘binning’ to average the data of the measurements over a time interval that is entered as
        the value associated with the key.
        - 'edge' to crop the edges of arrays of all measurements in MacularDictArray. The value can be a tuple to
        crop differently in horizontal and vertical axes: (x_edge, y_edge) or an int to crop everywhere the same.
        The x_edge and the y_edge of the tuple can also be tuples to crop asymmetrically the two edges of each axis.
        - ‘VSDI’ to calculate the voltage sensitive dye imaging signal of the cortex. Two possible values: True and
        False.
        - ‘derivative’ to calculate the derivative of the measurements. It is possible to add an integer value
        to integrate over a larger interval. Otherwise, an instantaneous derivative will be obtained. The
        ‘derivative’ key is associated with a dictionary that must contain the measurements to be processed as keys
        and the size of the derivative interval as a value.
        - ‘temporal_centering’ to center the time index of each cell on the moment when the bar reaches the
        center of their receptor field. Two possible values: True and False.
        - 'spatial_x_centering' to center the x-axis index on the center of the grid cell length. Two possible values:
        True and False.
        - 'spatial_y_centering' to center the y-axis index on the center of the grid cell width. Two possible values:
        True and False.

        It is also possible to add any new index based on spatial or temporal indexes units conversion. To do
        this, simply add a key with the prefix ‘spatial_index’ or ‘temporal_index’ followed by the suffix that will be
        used and added to the end of the index name. This key is associated with the value of the coefficient that you
        want to multiply by the index to convert its unit. For example, ‘temporal_index_ms’: 1000 adds a new index
        ‘temporal_ms’ in milliseconds. In addition, all "temporal" indexes will be used to create as many new indexes.
        Thus, if ‘temporal_centering’ is active, this will create a ‘temporal_centered_ms’ index. For this reason, unit
        conversion processing must always be performed last in order to process all possible new spatial or temporal
        measurements before.
        """
        print("Preprocessing : ", end="")

        process_order = ("binning", "edge", "VSDI", "derivative", "temporal_centering", "spatial_x_centering",
                         "spatial_y_centering", "mean_sections")

        for preprocess in process_order:
            # If the preprocess does not exist, move on to the next one.
            try:
                self.dict_preprocessing[preprocess]
            except KeyError:
                continue

            # Binning of data and index arrays.
            if preprocess == "binning":
                print(f"Binning {self.dict_preprocessing['binning']}s...", end="")
                self.binning_preprocess()

            # Crop of x and y edges
            elif preprocess == "edge":
                print("Edge cropping...", end="")
                self.edge_cropping_preprocess()

            # Computation of the array of data VSDI.
            elif preprocess == "VSDI":
                print("VSDI computing...", end="")
                self.data["VSDI"] = DataPreprocessor.vsdi_computing(self.data)

            # Computation of the array of data derivatives.
            elif preprocess == "derivative":
                print("Derivating...", end="")
                self.derivating_preprocess()

            # Temporal centering of index array.
            elif preprocess == "temporal_centering":
                print("Temporal centering...", end="")
                self.temporal_centering_preprocess()

            # Spatial centering of x-axis index array.
            elif preprocess == "spatial_x_centering":
                print("Spatial x centering...", end="")
                self.index[f"spatial_x_centered"] = DataPreprocessor.spatial_centering(
                    self.index["spatial_x"], self.dict_simulation["n_cells_x"])

            # Spatial centering of y-axis index array.
            elif preprocess == "spatial_y_centering":
                print("Spatial y centering...", end="")
                self.index[f"spatial_y_centered"] = DataPreprocessor.spatial_centering(
                    self.index["spatial_y"], self.dict_simulation["n_cells_y"])

            # Mean sectioning along one axis of index array.
            elif preprocess == "mean_sections":
                    print(f"Mean sectioning...", end="")
                    self.mean_sectioning_preprocess()

        # All indexes units conversion
        print(f"Units converting...", end="")
        self.make_all_indexes_units_conversion_preprocess()

        print("Done!")

    def binning_preprocess(self):
        """Function to perform binning of all MacularDictArray measurements as well as the time index.

        The binning interval is defined by the ‘binning’ key in the preprocessing dictionary.
        """
        bin_size, n_bin = DataPreprocessor.computing_binning_parameters(self.index["temporal"],
                                                                        self.dict_preprocessing["binning"])
        self.index["temporal"] = DataPreprocessor.binning_unidimensional(self.index["temporal"], bin_size, n_bin)
        for measurement in self.data:
            self.data[measurement] = DataPreprocessor.binning_tridimensional(self.data[measurement], bin_size,
                                                                             n_bin)

    def edge_cropping_preprocess(self):
        """Function to remove the edges of the cell area from the MacularDictArray.

        The crop is performed according to the ‘edge’ key in the preprocessing dictionary. If this key is associated
        to an int, the value of the int will be used to crop all edges equally. If the value is a tuple, cropping will
        be performed differently for the horizontal (x) and vertical (y) axes. If the value is a tuple of tuples, then
        cropping will also be performed asymmetrically between the two edges of the x or y axes.

        All cropping values to be applied to each edge of the spatial area are added to a dictionary containing the
        keys: x_min_edge, x_max_edge, y_min_edge and y_max_edge.
        """
        dict_edges = CoordinateManager.edge_to_dict_edge(self.dict_preprocessing["edge"])

        for measurement in self.data:
            self.data[measurement] = DataPreprocessor.array_edge_cropping(self.data[measurement], dict_edges.copy())
        self.index["spatial_x"] = self.index["spatial_x"][dict_edges["x_min_edge"]:
                                                          len(self.index["spatial_x"]) - dict_edges["x_max_edge"]]
        self.index["spatial_y"] = self.index["spatial_y"][dict_edges["y_min_edge"]:
                                                          len(self.index["spatial_y"]) - dict_edges["y_max_edge"]]

    def derivating_preprocess(self):
        """Function for calculating the derivative of given measurements.

        Derivatives are made based on the information contained in the ‘derivative’ key in the preprocessing dictionary.
        The value of this key is a dictionary with pairs associating the name of the measurement to be derived and,
        as value, the derivative window.
        """
        for measurement in self.dict_preprocessing["derivative"]:
            self.data[f"{measurement}_derivative"] = DataPreprocessor.derivative_computing_3d_array(
                self.data[measurement], self.index["temporal"],
                self.dict_preprocessing["derivative"][measurement])

    def temporal_centering_preprocess(self):
        """Function that centers the temporal index on the time when the bar is located in the centre of the receiver
        field of each cell.

        Time centering is performed according to the ‘temporal_centering’ key in the preprocessing dictionary. The
        centering computing produces a two-dimensional array of centred times for each cell on the object's movement
        axis.

        If edges have been cropped from the current MacularDictArray, it is necessary to crop these edges also these
        edges in the list of arrival times in the bar in the centre of the receiver fields. All cropping values to be
        applied to each edge of the spatial area are added to a dictionary containing the keys: x_min_edge, x_max_edge,
        y_min_edge and y_max_edge
        """
        # List of arrival times of the bar on all cells of the MacularDictArray before any cropping.
        list_time_bar_center = CoordinateManager.get_list_time_motion_center(self.dict_simulation)

        # Cases where the edges of the MacularDictArray space area have been cropped.
        try:
            dict_edges = CoordinateManager.edge_to_dict_edge(self.dict_preprocessing["edge"])
            if self.dict_simulation["axis"] == "horizontal":
                list_time_bar_center = list_time_bar_center[dict_edges["x_min_edge"]:
                                                            len(list_time_bar_center) - dict_edges["x_max_edge"]]
            elif self.dict_simulation["axis"] == "vertical":
                list_time_bar_center = list_time_bar_center[dict_edges["y_min_edge"]:
                                                            len(list_time_bar_center) - dict_edges["y_max_edge"]]
        # Case without edges cropped.
        except KeyError:
            pass

        # Centering the time index within its assigned index.
        self.index[f"temporal_centered"] = DataPreprocessor.temporal_centering(
            self.index["temporal"], list_time_bar_center)

    def mean_sectioning_preprocess(self):
        """Function using the 3D array of values along one of these axes.

        Mean sectioning is performed based on the ‘mean_sections’ key in the preprocessing dictionary. It produces a
        two-dimensional array in which the removed axis has been averaged. The ‘mean_sections’ key is associated with a
        dictionary containing the names of the axes to be averaged. The corresponding values are lists of other
        dictionaries containing the parameters of the mean section. Each element of the list will be the source of a
        new array in the MacularDictArray. The parameter dictionary must contain the ‘measurement’ key, which is the
        measurement to be averaged, the ‘cropping_type’ which defines how the array should be cropped or not in
        preparation for averaging, and the ‘cropping_dict’ which groups the parameters specific to this cropping.

        The function begins by scanning the names of the axes to be averaged and the measurements. For each of them,
        it transforms the array into a imbricated array. This is a 2D array with the dimensions of the two axes not
        averaged, and each position stores a 1D array corresponding to the axis to be averaged. This allows to isolate
        values according to the axis to be averaged. This imbricated array is then cropped or not, depending on the
        user's input. Only the isolated array corresponding to the axis to be averaged is cropped. Different cropping
        methods are possible. The imbricated array structure was chosen because some of these methods can result in
        different array sizes coexisting within the same array. Finally, the isolated array from the imbricated array is
        averaged to obtain the average section.
        """
        # Loop on the axes to be averaged.
        for averaging_axis in self.dict_preprocessing["mean_sections"]:
            # Loop through the measurements of each axis to be averaged.
            for mean_section_dictionary in self.dict_preprocessing["mean_sections"][averaging_axis]:
                # Creation of a imbricated array isolating the axis to be averaged.
                imbricated_array_data = DataPreprocessor.transform_3d_array_to_imbricated_arrays(
                self.data[mean_section_dictionary["measurement"]], averaging_axis)

                # Cut the axis to be averaged within the imbricated array to average only a portion.
                imbricated_array_data = DataPreprocessor.crop_imbricated_array(
                    imbricated_array_data, mean_section_dictionary["cropping_type"],
                    mean_section_dictionary["cropping_dict"])

                # Average of the current axis transforming the imbricated array into a 2D array.
                self.data[f"{averaging_axis}_mean_section_{mean_section_dictionary['cropping_type']}".strip("_")] = (
                    DataPreprocessor.imbricated_arrays_axis_averaging(imbricated_array_data))

    def make_all_indexes_units_conversion_preprocess(self):
        """Function that creates dictionary entries corresponding to conversions of temporal and spatial indexes.

        All new entries are taken from the preprocessing dictionary. These are all keys beginning with ‘temporal_index’
        or ‘spatial_index’. These keys end with a suffix that is found in the name of the new index created. Depending
        on whether ‘temporal’ or ‘spatial’ has been specified, all indexes of this type will undergo a unit conversion
        to produce a new index.

        For example, ‘temporal_index_mm’ will create a ‘temporal_mm’ index, but also a ‘temporal_centered_ms’ index if
        ‘centering’ is enabled.
        """
        index_copy = self.index.copy()

        for preprocess in self.dict_preprocessing:
            if "index" in preprocess:
                type_index = preprocess.split("_")[0]
                suffix_index = preprocess.replace(f"{type_index}_index_", "")
                self._index.update(DataPreprocessor.conversion_specific_arrays_unit_dict_array(
                    index_copy, type_index, suffix_index, self.dict_preprocessing[preprocess]))

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
        parameter of the MacularDictArray is associated with conditions (keys) whose names are not important here. In
        addition, it is possible to create a key named ‘global’ which allows you to group together all the parameters
        whose values are shared by all the MacularDictArray to be created. The simulation and processing dictionaries
        used to create each MacularDictArray are constructed by merging the condition and global dictionaries.

        Parameters
        ----------
        multiple_dicts_simulations : dict of dict
            Dictionary associating simulation condition names with simulation dictionaries.

            The dictionary may  also contain a ‘global’ key containing parameters shared between all simulations. The
            simulation dictionary cannot be empty or contains only a ‘global’ key. There must be at least one
            difference, such as the pyb file name.

            Example :
            {
                "global": {"n_cells_x": 83, "n_cells_y": 15, "dx": 0.225, "delta_t": 0.0167,
                            "end": "max", "size_bar": 0.67, "axis": "horizontal"},
                "barSpeed6dps": {"path_pyb": path/to/file.pyb, "path_csv": path/to/file.csv, "speed":6}
                "barSpeed30dps": {"path_pyb": path/to/file.pyb, "path_csv": path/to/file.csv, "speed":30}
            }


        multiple_dicts_preprocessings : dict of dict
            Dictionary associating preprocessing condition names with preprocessing dictionaries.

            The dictionary may also contain a ‘global’ key containing parameters shared between all preprocessing of
            MacularDictArray. The preprocessing dictionary can be empty or contains only a ‘global’ key.

        Returns
        ----------
        multi_macular_dict_array : dict of MacularDictArray
            Returns a dictionary associating specific conditions with MacularDictArray.
        """
        multi_macular_dict_array = {}

        for condition in multiple_dicts_simulations:
            if condition != "global":
                # Loading simulation dictionary parameters shared between simulations.
                try:
                    dict_simulation = multiple_dicts_simulations["global"].copy()
                except KeyError:
                    dict_simulation = {}

                # Loading preprocessing dictionary parameters shared between simulations.
                try:
                    dict_preprocessing = multiple_dicts_preprocessings["global"].copy()
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

                multi_macular_dict_array[condition] = MacularDictArray(dict_simulation, dict_preprocessing)

        return multi_macular_dict_array
