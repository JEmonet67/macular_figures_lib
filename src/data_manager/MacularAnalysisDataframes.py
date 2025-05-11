import re
import copy

import numpy as np
import pandas as pd

from src.data_manager.MacularDictArray import MacularDictArray


class MacularAnalysisDataframes:
    """Summary

    Explanation

    Attributes
    ----------
    dict_paths_pyb : dict of str
        Summary attr1

    dict_analysis_dataframes : dict of pd.DataFrame
        Summary attr1

    multiple_dicts_analysis : dict of dict
        Summary attr1

    multiple_dicts_simulations : dict of dict
        Summary attr1

    multiple_dicts_preprocessings : dict of dict
        Summary attr1

    condition_reg : re.Pattern
        Regular expression to extract the name of the condition, its value and its unit from the keys of the
        MacularDictArray multiple dictionary.

        By default, the regular expression entered allows you to read conditions that follow a ‘NameValueUnit’ format.

    Example
    ----------
    >>> instruction
    result instruction

    """

    def __init__(self, multiple_dicts_simulations, multiple_dicts_preprocessings, multiple_dicts_analysis):
        """Summary

        Explanation

        Parameters
        ----------
        multiple_dicts_simulations : dict of dict
            Dictionary associating simulation dictionaries with multiple conditions

            The dictionary may  also contain a ‘global’ key containing parameters shared between all simulations. The
            simulation dictionary cannot be empty or contains only a ‘global’ key. There must be at least one
            difference, such as the pyb file name. It is possible to enter simulation dictionaries that only contain the
            path of the pyb file.

        multiple_dicts_preprocessings : dict of dict
            Dictionary associating preprocessing dictionaries with multiple conditions

            The dictionary may also contain a ‘global’ key containing parameters shared between all preprocessing of
            MacularDictArray. The preprocessing dictionary can be empty or contains only a ‘global’ key.

        multiple_dicts_analysis : dict of dict
            Dictionary associating the name of the analysis dataframe with the dictionary of analyses to be performed
            on it.

            The dictionary of analyses to be performed on each dataframe contains the name of the analysis associated
            with its value, which can be a dictionary if you want a different value for each condition present in the
            MacularAnalysisDataframe.
        """
        # Create or import the corresponding batch of Macular Dict Array.
        multi_macular_dict_array = MacularDictArray.make_multiple_macular_dict_array(multiple_dicts_simulations,
                                                                                     multiple_dicts_preprocessings)

        # Create and clean the multiple_dicts_analysis attributes.
        self.multiple_dicts_analysis = multiple_dicts_analysis
        self.multiple_dicts_analysis = self.cleaning_multiple_dicts_features(multiple_dicts_analysis)

        # Create and clean the multiple_dicts_simulations attributes.
        self._multiple_dicts_simulations = multiple_dicts_simulations
        self._multiple_dicts_simulations = self.cleaning_multiple_dicts_features(multiple_dicts_simulations)

        # Create and clean the multiple_dicts_preprocessings attributes.
        self._multiple_dicts_preprocessings = multiple_dicts_preprocessings
        self._multiple_dicts_preprocessings = self.cleaning_multiple_dicts_features(multiple_dicts_preprocessings)

        # Create dict_paths_pyb attributes to store each path_pyb associated to its condition.
        self.dict_paths_pyb = {}
        for condition in multi_macular_dict_array:
            self.dict_paths_pyb[condition] = multi_macular_dict_array[condition].path_pyb
            # Delete path_pyb attribute from simulations dictionaries.
            del self._multiple_dicts_simulations[condition]["path_pyb"]

        # Regular expression to extract the name, value and unit of a condition with "NameValueUnit" format.
        self.condition_reg = re.compile("(^[A-Za-z]+)(-?[0-9]{1,4},?[0-9]{0,4})([A-Za-z]+$)")

        # Create the dataframes specified in the analysis dictionary.
        t_index = self.get_maximal_index_multi_macular_dict_array(multi_macular_dict_array, "temporal")
        x_index = self.get_maximal_index_multi_macular_dict_array(multi_macular_dict_array, "spatial_x")
        y_index = self.get_maximal_index_multi_macular_dict_array(multi_macular_dict_array, "spatial_y")
        self.initialize_dict_analysis_dataframes(x_index, y_index, t_index)

        # Make analysis

    @property
    def dict_paths_pyb(self):
        """Getter for the dict_paths_pyb attribute.
        """
        return self._dict_paths_pyb

    @dict_paths_pyb.setter
    def dict_paths_pyb(self, dict_paths_pyb):
        """Setter for the dict_paths_pyb attribute.
        """
        self._dict_paths_pyb = dict_paths_pyb

    @property
    def dict_analysis_dataframes(self):
        """Getter for the dict_analysis_dataframes attribute.
        """
        return self._dict_analysis_dataframes

    @dict_analysis_dataframes.setter
    def dict_analysis_dataframes(self, dict_analysis_dataframes):
        """Setter for the dict_paths_pyb attribute.
        """
        self._dict_analysis_dataframes = dict_analysis_dataframes

    @property
    def multiple_dicts_analysis(self):
        """Getter for the multiple_dicts_analysis attribute.
        """
        return self._multiple_dicts_analysis

    @multiple_dicts_analysis.setter
    def multiple_dicts_analysis(self, multiple_dicts_analysis):
        """Setter for the multiple_dicts_analysis attribute.
        """
        self._multiple_dicts_analysis = self.cleaning_multiple_dicts_features(multiple_dicts_analysis)

    @property
    def multiple_dicts_preprocessings(self):
        """Getter for the multiple_dicts_preprocessings attribute.
        """
        return self._multiple_dicts_preprocessings

    @multiple_dicts_preprocessings.setter
    def multiple_dicts_preprocessings(self, multiple_dicts_preprocessings):
        """Setter for the multiple_dicts_preprocessings attribute.
        """
        raise AttributeError("The attribute multiple_dicts_preprocessings can't be modified.")

    @property
    def multiple_dicts_simulations(self):
        """Getter for the multiple_dicts_simulations attribute.
        """
        return self._multiple_dicts_simulations

    @multiple_dicts_simulations.setter
    def multiple_dicts_simulations(self, multiple_dicts_simulations):
        """Setter for the multiple_dicts_simulations attribute.
        """
        raise AttributeError("The attribute multiple_dicts_simulations can't be modified.")

    @property
    def condition_reg(self):
        """Getter for the name_value_unit_reg attribute.
        """
        return self._condition_reg

    @condition_reg.setter
    def condition_reg(self, condition_reg):
        """Setter for the name_value_unit_reg attribute.

        The setter includes the compilation of the regular expression entered as a parameter.

        Parameters
        ----------
        condition_reg : str
            Character string corresponding to a regular expression for sorting condition names.

        """
        self._condition_reg = re.compile(condition_reg)

    @staticmethod
    def get_maximal_index_multi_macular_dict_array(multi_macular_dict_array, name_index):
        """Creation of an empty dictionary of analysis dataframes.

        Parameters
        ----------
        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with MacularDictArray.

        name_index : str
            Name of the MacularDictArray index from which the index with the maximum size must be extracted. The names
            can be ‘temporal’, ‘spatial_x’ and ‘spatial_y’.

        Returns
        ----------
        index : np.array()
            Numpy array of the index with the largest size within the MacularDictArray.
        """
        maximal = 0
        index = np.array([0])
        for name_dict_array in multi_macular_dict_array:
            dict_array = multi_macular_dict_array[name_dict_array]
            if dict_array.index[name_index].shape[0] > maximal:
                maximal = dict_array.index[name_index].shape[0]
                index = dict_array.index[name_index].round(5)

        return index

    def initialize_dict_analysis_dataframes(self, x_index=np.array([0]), y_index=np.array([0]), t_index=np.array([0])):
        """Creation of an empty dictionary of analysis dataframes.

        The analysis dataframe dictionary associates each analysis dataframe name with an empty dataframe
        that only has a named index. When this is created, only analysis dataframes containing analyses to be performed
        on them are taken into account. Other dataframes associated with empty analysis dictionaries are discarded. The
        dataframe of conditions is by default sorted alphabetically or according to another specific order.

        Parameters
        ----------
        x_index : np.array
            Index of the x-axis used for the x-axis dataframe.

            This parameter is facultative.

        y_index : np.array
            Index of the y-axis used for the y-axis dataframe.

            This parameter is facultative.

        t_index : np.array
            Index of the t-axis used for the t-axis dataframe.

            This parameter is facultative.
"""
        self.dict_analysis_dataframes = {}

        # Creates a list of the names of analysis dataframes present in the analysis dictionary.
        names_dataframes = list(self.multiple_dicts_analysis.keys())

        # Initialise only dataframes that are present.
        for name_dataframe in names_dataframes:
            # Create conditions dataframe.
            if name_dataframe == "Conditions":
                sorted_conditions = self.dataframe_conditions_sorting()
                self.dict_analysis_dataframes[name_dataframe] = self.initialize_analysis_dataframe(
                    sorted_conditions, name_dataframe)
                self.setup_conditions_values_to_condition_dataframe()
            # Create y-axis dataframe.
            elif name_dataframe == "X":
                self.dict_analysis_dataframes[name_dataframe] = {condition: self.initialize_analysis_dataframe(
                    x_index, name_dataframe) for condition in self.dict_paths_pyb.keys()}
            # Create x-axis dataframe.
            elif name_dataframe == "Y":
                self.dict_analysis_dataframes[name_dataframe] = {condition: self.initialize_analysis_dataframe(
                    y_index, name_dataframe) for condition in self.dict_paths_pyb.keys()}
            # Create t-axis dataframe.
            elif name_dataframe == "Time":
                self.dict_analysis_dataframes[name_dataframe] = {condition: self.initialize_analysis_dataframe(
                    t_index, name_dataframe) for condition in self.dict_paths_pyb.keys()}

    @staticmethod
    def initialize_analysis_dataframe(columns, name_columns):
        """Create an empty analysis dataframe with only a named index.

        Parameters
        ----------
        index : list of str
            List of items to be placed as the index of the DataFrame.

        name_index : str
            Name of the Dataframe index.

        Returns
        ----------
        analysis_dataframe : pd.DataFrame
            Returns an initialised, empty analysis dataframe with a named index.
        """
        analysis_dataframe = pd.DataFrame(columns=columns)
        analysis_dataframe.columns.name = name_columns

        return analysis_dataframe

    def dataframe_conditions_sorting(self):
        """Sorting the conditions of a multiple MacularDictArray.

        Sorting is performed on the list of conditions for the various MacularDictArray. Three types of sorting are used
        here. The first ‘default’ sort is in alphabetical order. The second sort is performed according to a pre-sorted
        list of each condition, which is entered in the multiple analysis dictionaries. Finally, the last sort is
        performed on the names of conditions that must follow a specific format: ‘NameValueUnit’.

        In the case of multiple conditions, each NameValueUnit must also be separated from the others by an underscore.
        Sorting is then performed with priority from left to right.

        Example :
        >> ["barSeed6dps_wAmaBip1Hz", "barSeed1dps_wAmaBip1Hz", "barSeed6dps_wAmaBip0.2Hz"]
        ["barSeed1dps_wAmaBip1Hz", "barSeed6dps_wAmaBip0", "barSeed6dps_wAmaBip1Hz"]

        Returns
        ----------
        sorted_conditions : list
            Sorted list of the different conditions present in the multiple MacularDictArray.
        """
        # Case of default sorting by alphabetical order.
        sorted_conditions = list(self.dict_paths_pyb.keys())
        sorted_conditions.sort()

        # Cases where specific sorting is required.
        try:
            # Case of an ordered list of conditions.
            if isinstance(self.multiple_dicts_analysis["Conditions"]["sorting"], list):
                sorted_conditions = self.multiple_dicts_analysis["Conditions"]["sorting"]

            # Case of automatic sorting by the name of each condition according to a format.
            elif isinstance(self.multiple_dicts_analysis["Conditions"]["sorting"], str):
                # Automatic sorting by the ‘NameValueUnit’ format.
                if self.multiple_dicts_analysis["Conditions"]["sorting"] == "NameValueUnit":
                    sorted_conditions = self.name_value_unit_sorting_conditions()
        except KeyError:
            pass

        return sorted_conditions

    def name_value_unit_sorting_conditions(self):
        """Function sorting condition names according to the ‘NameValueUnit’ naming convention.

        Sorting is first performed on the condition name.

        Example :
        conditions = ["barSpeed80dps", "barSpeed23dps", "barSpeed28dps",
        > barSpeed6dps

        Returns
        ----------
        sorted_conditions : list
            List containing conditions sorted by condition name and value.
        """
        sorted_conditions = list(self.dict_paths_pyb.keys())

        sorted_conditions.sort(key=lambda x: [(self.condition_reg.findall(x.split("_")[i])[0][0],
                                               float(self.condition_reg.findall(x.split("_")[i])[0][1].replace(
                                                   ",", "."))) for i in range(len(x.split("_")))])

        return sorted_conditions

    @staticmethod
    def cleaning_multiple_dicts_features(multiple_dicts_features):
        """Cleans the analysis dictionary by removing all keys associated with a value of False.

        The purpose of this cleanup is to take into account that analysis missing from the analysis dictionary are
        equivalent to analysis that are present but with a value set to False.

        Parameters
        ----------
        multiple_dicts_features : dict of dict
            Dictionary associating features dictionaries with dictionnaries of parameters.

            These dictionaries can be illustrated by the dictionaries multiple_dicts_analysis or
            multiple_dicts_simulations. In the first case, the features are dataframes, each of which can have different
            analysis parameters. In the second case, the features are the different conditions of the MacularDictArray,
            each of which has different simulation parameters.

        Returns
        ----------
        multiple_dicts_analysis_cleaned : dict of dict
            Multiple analysis dictionary with no keys associated with False values.
        """
        # Deep copy of the multiple_dicts_features.
        multiple_dicts_features_cleaned = copy.deepcopy(multiple_dicts_features)

        for dataframe in multiple_dicts_features:
            # Removal of false features.
            for feature in multiple_dicts_features[dataframe]:
                if not multiple_dicts_features[dataframe][feature]:
                    del multiple_dicts_features_cleaned[dataframe][feature]

            # Removed empty features dictionaries.
            if not multiple_dicts_features_cleaned[dataframe]:
                del multiple_dicts_features_cleaned[dataframe]

        return multiple_dicts_features_cleaned

    def setup_conditions_values_to_condition_dataframe(self):
        """Set up columns in the condition dataframe containing the different conditions used in the
        multiple MacularDictArray.

        The name of each column corresponds to the name of the condition followed by its unit in brackets: "barSpeed
        (dps)". The column is then filled with the different values taken by the condition between the MacularDictArray
        present on each line. If a condition does not already exist among the columns, it is added. However, if it
        already exists, the value of the current MacularDictArray row is filled with its value.
        """
        for conditions in self.dict_analysis_dataframes["Conditions"].columns:
            for condition in conditions.split("_"):
                name, value, unit = self.condition_reg.findall(condition)[0]
                if f"{name} ({unit})" not in self.dict_analysis_dataframes["Conditions"].index:
                    self.dict_analysis_dataframes["Conditions"].loc[f"{name} ({unit})"] = ""
                self.dict_analysis_dataframes["Conditions"].loc[f"{name} ({unit})", conditions] = float(
                    value.replace(",", "."))
