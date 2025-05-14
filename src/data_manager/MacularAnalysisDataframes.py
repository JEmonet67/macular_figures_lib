import re
import copy

import numpy as np
import pandas as pd

from src.data_manager.MacularDictArray import MacularDictArray
from src.data_manager.SpatialAnalyser import SpatialAnalyser


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

        # Analysis preparation
        multiple_dicts_analysis_substituted, dict_sort_order = self.setup_multiple_dicts_analysis(
            multi_macular_dict_array)

        # Make analysis
        self.make_spatial_dataframes_analysis(multiple_dicts_analysis_substituted["X"], "X",
                                              multi_macular_dict_array, dict_sort_order)

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

    def setup_multiple_dicts_analysis(self, multi_macular_dict_array):
        """Function for preparing a copy of the multiple analysis dictionaries for analysis.

        The multiple analysis dictionaries is first copied before being modified to replace all aliases ‘all_conditions’
        and “all_measurements” with a name containing all conditions or measurements separated by ‘:’. A second
        dictionary is also constructed from the multiple analysis dictionaries, associating each dimension and analyses
        with a list of conditions and measurements for each MacularDictArray. These lists are sorted alphabetically,
        except for the first element, which corresponds to the case ‘all_conditions’ or ‘all_measurements’ if present.
        This allows global cases to be applied before specific ones.

        Parameters
        ----------
        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with MacularDictArray.

        Returns
        ----------
        multiple_dicts_analysis_substituted : dict of dict
            Multiple analysis dictionary with "all_conditions" and "all_measurements" aliases substituted.

        dict_sort_order : dict of dict
            Dictionary containing ordered lists of condition groups and measurements for all analyses and dimensions of
            multiple analysis dictionaries.
        """
        # Create a copy of the multiple analysis dictionaries.
        multiple_dicts_analysis_copy = copy.deepcopy(self.multiple_dicts_analysis)

        # Obtaining the names of conditions and measurements present in the MacularDictArray used.
        levels_multiple_dictionaries = self.get_levels_of_multi_macular_dict_array(multi_macular_dict_array)

        # Substitution of all aliases ‘all_conditions’ and ‘all_measurements’ in the multiple analysis dictionaries.
        multiple_dicts_analysis_substituted = self.substituting_all_alias_in_multiple_analysis_dictionaries(
            multiple_dicts_analysis_copy, levels_multiple_dictionaries)

        # Creation of a dictionary containing the conditions and their measurements sorted.
        dict_sort_order = self.creating_sort_order_from_multiple_dicts_analysis(multiple_dicts_analysis_substituted,
                                                                                levels_multiple_dictionaries)

        return multiple_dicts_analysis_substituted, dict_sort_order

    def substituting_all_alias_in_multiple_analysis_dictionaries(self, multiple_dicts_analysis,
                                                                 levels_multiple_dictionaries):
        """Function replacing all aliases ‘all_conditions’ and ‘all_measurements’ in multiple analysis dictionaries.

        Parameters
        ----------
        multiple_dicts_analysis : dict of MacularDictArray
            Copy of the multiple analysis dictionaries contained in the current MacularAnalysisDataframes.

        levels_multiple_dictionaries : list of str or dict of str
            List containing all the names of conditions and measurements found in the associated MacularDictArray
            separated by ‘:’.

        Returns
        ----------
        multiple_dicts_analysis : dict of dict
            Multiple analysis dictionary with "all_conditions" and "all_measurements" aliases substituted.
        """
        for dimension in multiple_dicts_analysis:
            for analysis in multiple_dicts_analysis[dimension]:
                self.substituting_all_alias_in_analysis_dictionary(multiple_dicts_analysis[dimension], analysis,
                                                                   levels_multiple_dictionaries)

        return multiple_dicts_analysis

    def substituting_all_alias_in_analysis_dictionary(self, multiple_dicts_analysis_unidimensional, name_level,
                                                      levels_multiple_dictionaries, n=0):
        """Recursive function allowing the aliases ‘all_conditions’ and ‘all_measurements’ to be substituted within one
        analysis dictionary of a dimension of the MacularAnalysisDataframe (X, Y, Time, Conditions).

        The recursion of the function is only performed on the two levels of the analysis dictionary: the conditions
        and measurements. In the case of ‘all_measurements’, all measurements associated with the current
        MacularDictArray are taken into account. This means that if several conditions are associated with the same
        ‘all_measurements’, these conditions must have MacularDictArray with exactly the same measurements.

        If the conditions and measurements associated with ‘all_conditions’ and ‘all_measurements’ already exist in
        the dictionary, the latter will be replaced by the dictionary associated with “all_conditions” or
        ‘all_measurements’. An alert message is displayed when this happens to inform the user.

        Parameters
        ----------
        multiple_dicts_analysis_unidimensional : dict of dict
            Dictionary for analysing a given dimension (X, Y, Time or Conditions) associating analyses or conditions
            (depending on the level of recursion) with dictionaries of conditions or measurements. These measurements
            can also be a Boolean, int or float parameter.

        name_level : str
            Name of the architectural level in which we are located in the analysis dictionary.

        levels_multiple_dictionaries : list of str or dict of str
            List containing all the names of conditions and measurements found in the associated MacularDictArray
            separated by ‘:’.

        n : int
            Hierarchical level counter in the analysis dictionary that increases with recursion. It is always less than
            or equal to 2.

        """
        # Case of the ‘condition’ level of the analysis dictionary structure.
        if isinstance(levels_multiple_dictionaries[n], str):
            levels_current_dictionary = levels_multiple_dictionaries[n]
        # Case of the ‘measurement’ level of the analysis dictionary structure.
        elif isinstance(levels_multiple_dictionaries[n], dict):
            levels_current_dictionary = levels_multiple_dictionaries[n][name_level.split(":")[0]]

        # Increment the hierarchical level counter.
        n += 1

        # Substitution only on analyses with an additional hierarchical level in their analysis dictionary.
        if isinstance(multiple_dicts_analysis_unidimensional[name_level], dict):
            # Loop through all names of lower levels in the analysis dictionary hierarchy.
            for low_level in multiple_dicts_analysis_unidimensional[name_level].copy():
                # The name of the level is kept or adapted if it was ‘all_conditions’ or ‘all_measurements’.
                corrected_low_level = low_level
                if "all_" in low_level:
                    corrected_low_level = levels_current_dictionary
                    # Error message when the adapted level of ‘all_conditions’ or ‘all_measurements’ already exists.
                    if corrected_low_level in multiple_dicts_analysis_unidimensional[name_level]:
                        print(f"Warning: substitution of existing term : {corrected_low_level}")
                    multiple_dicts_analysis_unidimensional[name_level][corrected_low_level] = (
                        multiple_dicts_analysis_unidimensional)[name_level][low_level]
                    del multiple_dicts_analysis_unidimensional[name_level][low_level]

                # If you are in the first level of the dictionary hierarchy, you enter the next level.
                if n < 2:
                    self.substituting_all_alias_in_analysis_dictionary(
                        multiple_dicts_analysis_unidimensional[name_level], corrected_low_level,
                        levels_multiple_dictionaries, n)
                else:
                    pass

    def creating_sort_order_from_multiple_dicts_analysis(self, multiple_dicts_analysis_substituted,
                                                         levels_multiple_dictionaries):
        """Function to create a dictionary containing ordered lists of conditions and measurements to be analysed for
        each dimension and analysis.

        These lists allow each dimension and analysis to analyse each group of conditions and measurements in
        alphabetical order. There is also an exception for the first item in the list, which will always correspond to
        the conditions or measurements corresponding to aliases ‘all_conditions’ and ‘all_measurements’. The aim is to
        start with the global cases before move on to the specific cases.

        The dictionary is initially structured with two hierarchical levels corresponding to the dimension of the
        Macular Analysis Dataframes to be used and the analysis. For a given dimension and analysis, there is a
        dictionary with the keys ‘conditions’ and ‘measurements’. The ‘conditions’ key is directly associated with the
        list of conditions in the multi_macular_dict_array. The ‘measurements’ key, on the other hand, is associated
        with another dictionary composed of keys corresponding to the conditions in the multi_macular_dict_array
        associated with the lists of measurements found in each of the MacularDictArray.

        Creating an ordered list within this dictionary requires that the analysis dictionary be composed of a double
        dictionary with the conditions and then the measurements as keys. Otherwise, the common analysis or analysis
        group is ignored.

        Parameters
        ----------
        multiple_dicts_analysis_substituted : dict of dict
            Multiple analysis dictionary with "all_conditions" and "all_measurements" aliases substituted.

        levels_multiple_dictionaries : list of str or dict of str
            List containing all the names of conditions and measurements found in the associated MacularDictArray
            separated by ‘:’.

        Returns
        ----------
        dict_sort_order : dict of dict
            Dictionary containing ordered lists of condition groups and measurements for all analyses and dimensions of
            multiple analysis dictionaries.
        """
        dict_sort_order = {}
        for dimension in multiple_dicts_analysis_substituted:
            dict_sort_order[dimension] = {}
            for analysis in multiple_dicts_analysis_substituted[dimension]:
                # Verify that the current analysis dictionary is indeed a dictionary.
                if isinstance(multiple_dicts_analysis_substituted[dimension][analysis], dict):
                    # Sort the list of condition names.
                    sorted_list_conditions = self.creating_sort_order_from_dict_analysis(
                        multiple_dicts_analysis_substituted[dimension][analysis],
                        levels_multiple_dictionaries[0])
                    dict_sort_order[dimension][analysis] = {"conditions": sorted_list_conditions, "measurements": {}}
                    for grouped_conditions in multiple_dicts_analysis_substituted[dimension][analysis]:
                        # Verify that the current analysis dictionary is a double dictionary.
                        if isinstance(multiple_dicts_analysis_substituted[dimension][analysis][grouped_conditions],
                                      dict):
                            # Sort the list of measurement names depending on the current condition.
                            sorted_list_measurements = self.creating_sort_order_from_dict_analysis(
                                multiple_dicts_analysis_substituted[dimension][analysis][grouped_conditions],
                                levels_multiple_dictionaries[1][grouped_conditions.split(":")[0]])
                            dict_sort_order[dimension][analysis]["measurements"][grouped_conditions] = (
                                sorted_list_measurements)
                        # If the current analysis dictionary is not a double dictionary, it is removed.
                        else:
                            del dict_sort_order[dimension][analysis]

        return dict_sort_order

    @staticmethod
    def creating_sort_order_from_dict_analysis(dict_analysis, first_element):
        """Function that collects all names from a hierarchical level of the multiple analysis dictionaries and sorts
        them into an ordered list.

        The hierarchical level processed can be that of condition groups or measurements. Names can therefore be
        composed of a succession of conditions or measurements separated by ‘:’.

        The sorting is done alphabetically by the names of the elements. In addition, it is possible to place a given
        input element at the start of the sorted list if it is present in the keys of the analysis dictionary.

        Parameters
        ----------
        dict_analysis : dict of dict
            Dictionary corresponding to a hierarchical level in the multiple analysis dictionaries.

        first_element : str
            Name of a hierarchical level in the multiple analysis dictionaries to be placed at the top of the ordered
            list.

        Returns
        ----------
        sorted_list_elements : list of str
            Ordered list of names at a hierarchical level in the multiple analysis dictionaries.
        """
        # Creates and sorts the sorted list of elements in the current hierarchy of the analysis dictionary.
        sorted_list_elements = [grouped_elements for grouped_elements in dict_analysis]
        sorted_list_elements.sort()
        if first_element in sorted_list_elements:
            # Deletes the item to be placed at the start of the list.
            sorted_list_elements.pop(sorted_list_elements.index(first_element))

            # Adds the item to be placed at the start of the list to the first position.
            sorted_list_elements.insert(0, first_element)

        return sorted_list_elements

    # def make_conditions_dataframes_analysis(self, multi_macular_dict_array):
    #     conditions_analyser = ConditionsAnalyser()
    #     for analysis in self.multiple_dicts_analysis["Conditions"]:
    #         pass

    # def make_temporal_dataframes_analysis(self, multi_macular_dict_array):
    #     for analysis in self.multiple_dicts_analysis["Time"]:
    #         pass

    def make_spatial_dataframes_analysis(self, multiple_dicts_analysis_unidimensional, dimension,
                                         multi_macular_dict_array, dict_sort_order):
        """Function used to perform all MacularAnalysisDataframe analyses to be carried out in the spatial dimension
        (X and Y).

        The names of all analyses in the multiple analysis dictionaries are scanned and identified. For each of them, a
        conditional block allows the analysis function used later in the analysis execution to be adapted.

        Parameters
        ----------
        multiple_dicts_analysis_unidimensional : dict of dict
            Dictionary for analysing a given dimension (X, Y, Time or Conditions) associating analyses or conditions
            (depending on the level of recursion) with dictionaries of conditions or measurements. These measurements
            can also be a Boolean, int or float parameter.

        dimension : str
            Dimension of the MacularAnalysisDataframes in which the result of the current analysis is stored.

        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with MacularDictArray.

        dict_sort_order : dict of dict
            Dictionary containing ordered lists of condition groups and measurements for all analyses and dimensions of
            multiple analysis dictionaries.
        """
        # Performs all analyses listed in the current analysis dictionary.
        for analysis in multiple_dicts_analysis_unidimensional:
            if analysis == "activation_time":
                analysis_function = self.activation_time_analyzing
            elif analysis == "latency":
                analysis_function = self.latency_analyzing
            elif analysis == "time_to_peak":
                analysis_function = self.time_to_peak_analyzing
            elif analysis == "delay_to_peak":
                analysis_function = self.peak_delay_analyzing

            self.make_analysis(analysis_function, multi_macular_dict_array, multiple_dicts_analysis_unidimensional,
                               dimension, analysis, dict_sort_order)

    def get_levels_of_multi_macular_dict_array(self, multi_macular_dict_array):
        """Extract and format all condition names and their measurements from a multi macular dict array within strings
        of characters separated by ‘:’.

        The measurements that come into play could vary depending on the MacularDictArray. This is why the structure
        used to store the names of the measurements is a dictionary associating the conditions of the MacularDictArray
        with their measurements.

        Examples : "barSpeed27dps:barSpeed30dps" or "BipolarResponse_BipolarGainControl:VSDI"

        Parameters
        ----------
        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with MacularDictArray.

        Returns
        ----------
        levels_multiple_dictionaries : list of str or dict of str
            List containing all the names of conditions and measurements found in the associated MacularDictArray
            separated by ‘:’.

            The first element is a character string with conditions names separated by ":". The second is a dictionary
            associating the conditions of a multiple macular dict array as keys with values corresponding to the names
            of the measurements present in each of the MacularDictArray. The names of the measurements are also
            separated by ‘:’.
        """
        # Create character string containing all the conditions of a multiple MacularDictArray separated by ‘:’.
        all_conditions = ":".join(sorted([condition for condition in self.dict_paths_pyb]))

        # Create dictionary associating each multiple MacularDictArray conditions with their "all_conditions".
        all_measurements = {
            condition: ":".join(sorted([measure for measure in multi_macular_dict_array[condition].data]))
            for condition in self.dict_paths_pyb}

        levels_multiple_dictionaries = [all_conditions, all_measurements]

        return levels_multiple_dictionaries

    def make_analysis(self, analysis_function, multi_macular_dict_array, multiple_dicts_analysis_unidimensional,
                      dimension, analysis, dict_sort_order):
        """Function for performing one analysis from the multiple analysis dictionaries.

        Depending on the type of analysis being performed, the name of the analysis and the associated function are
        modified. The global analysis is performed by carrying out specific analyses for groups of common analyses.
        Each of these groups combines conditions and measurements that share the same parameter values for the current
        analysis.

        Before analysis, the multiple analysis dictionaries is corrected to replace all instances of the terms
        ‘all_conditions’ or “all_measurements” in the current analysis with character strings containing all conditions
        or measurements from the MacularDictArray processed in the MacularAnalysisDataframes. These conditions and
        measurements are separated by ‘:’.

        Parameters
        ----------
        analysis_function : function
            Analysis function to apply to calculate the current analysis.

            This analysis function changes between each analysis, so they must all have the same three input
            arguments: the data, the index and the analysis parameters.

        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with MacularDictArray.

        multiple_dicts_analysis_unidimensional : dict of dict
            Dictionary for analysing a given dimension (X, Y, Time or Conditions) associating analyses or conditions
            (depending on the level of recursion) with dictionaries of conditions or measurements. These measurements
            can also be a Boolean, int or float parameter.

        dimension : str
            Dimension of the MacularAnalysisDataframes in which the result of the current analysis is stored.

        analysis : str
            Name of the current analysis.

        dict_sort_order : dict of dict
            Dictionary containing ordered lists of condition groups and measurements for all analyses and dimensions of
            multiple analysis dictionaries.
        """
        # Double loop allowing to browse the conditions and measurements of common analysis groups.
        for grouped_conditions in dict_sort_order[dimension][analysis]["conditions"]:
            for grouped_measurements in dict_sort_order[dimension][analysis]["measurements"][grouped_conditions]:
                # Extract the list of condition/measurements pairs to be analysed with the same analysis parameters.
                common_analysis_group_list = self.common_analysis_group_parser(grouped_conditions, grouped_measurements)
                # Analysis of conditions/measurements for a common analysis group sharing the same analysis parameters.
                self.make_common_group_analysis(analysis_function, multi_macular_dict_array, common_analysis_group_list,
                                                dimension, analysis, multiple_dicts_analysis_unidimensional[analysis]
                                                [grouped_conditions][grouped_measurements])

        # TODO Faire de cette fonction une propriété pour chaque fonction d'analyses ?
        # TODO Ici soit je parse pour structurer en une collection ou structure de données les groupes d'analyses avant de les faire un par un.


    @staticmethod
    def common_analysis_group_parser(grouped_conditions, grouped_measurements):
        """Function that transforms the names of conditions and measurements in a group of common analyses into a list
        of pairs of conditions and measurements that share one or more identical analyses.

        The sets of condition names or measurements in the common analysis group (grouped_conditions and
        grouped_measurements) are separated by ‘:’. To extract each of them, separate them using this symbol.

        Examples : "barSpeed27dps:barSpeed30dps" or "BipolarResponse_BipolarGainControl:VSDI"

        Parameters
        ----------
        grouped_conditions : str
            Names of conditions in a common analysis group.

        grouped_measurements : str
            Names of measurements in a common analysis group.

        Returns
        ----------
        common_analysis_group_list : list of tuples
            List of all tuples pairs of conditions and measurements in a group of common analyses.
        """
        common_analysis_group_list = []

        # Loop on the conditions and measurements of the common analysis group.
        for condition in grouped_conditions.split(":"):
            for measurement in grouped_measurements.split(":"):
                # Added pair condition, measurement in progress.
                common_analysis_group_list += [(condition, measurement)]

        return common_analysis_group_list

    def make_common_group_analysis(self, analysis_function, multi_macular_dict_array, common_analysis_group_list,
                                   dimension, analysis, parameters_analysis_dict):
        """Function that performs a given analysis within a common group of analyses

        A common analysis group is a bunch of conditions and measurements that share one or more identical analyses
        with the same parameters. This group can be represented by a list of pairs of conditions and measurements on
        which to perform these same analyses.

        Each analysis is stored in the dataframe of the dimension given as input (Conditions, X, Y, Time) in the row
        named by its name and that of the measurement on which it is made, for example: ‘Activation_time_VSDI’. It
        is possible to add any suffix to this to differentiate between several similar analyses; this suffix must be
        associated with the keyword ‘flag’ in the parameter dictionary of the current analysis.

        Parameters
        ----------
        analysis_function : function
            Analysis function to apply to calculate the current analysis.

            This analysis function changes between each analysis, so they must all have the same three input
            arguments: the data, the index and the analysis parameters.

        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with MacularDictArray.

        common_analysis_group_list : list of tuples
            List of all tuples pairs of conditions and measurements in a group of common analyses.

        dimension : str
            Dimension of the MacularAnalysisDataframes in which the result of the current analysis is stored.

        analysis : str
            Name of the current analysis.

        parameters_analysis_dict : dict
            Dictionary containing all parameters of the current analysis.

            The parameters vary depending on the analysis, except for the optional ‘flag’ parameter, which can always be
            present. It corresponds to a suffix used to specify the name of the dataframe row that will be added.
        """
        # Managing the presence of a "flag" in the analysis dictionary.
        try:
            str_parameters_analysis = f"_{parameters_analysis_dict['flag']}"
        except KeyError:
            str_parameters_analysis = ""

        # Loop of conditions and measurements of the common analysis group.
        for condition, measurement in common_analysis_group_list:
            # Defines the name of the line where the current analysis is stored.
            dataframe_row = f"{analysis}_{measurement}{str_parameters_analysis}"
            # Conducting an analysis of a given condition and measurement.
            self.dict_analysis_dataframes[dimension][condition].loc[dataframe_row] = analysis_function(
                multi_macular_dict_array[condition].data[measurement], multi_macular_dict_array[condition].index
                , parameters_analysis_dict)

    @staticmethod
    def activation_time_analyzing(data, index, parameters_analysis_dict):
        """Function that analyses activation time based on a single spatial dimension.

        The activation time is calculated in the 3D array and the index of a measurement of a condition. It is obtained
        in the form of a 2D array from which only the desired X or Y position can be taken to obtain a 1D array based
        on a single spatial dimension.

        Parameters
        ----------
        data : np.array
            3D array containing the values of a measurement for a given condition.

        index : dict of np.array
            Dictionary containing all the indexes of a MacularDictArray in the form of a 1D array.

        parameters_analysis_dict : dict
            Dictionary of parameters to be used for activation time analysis. It must contain the threshold,
            the name of the index to be taken from the dictionary (allows switching from the s index to the ms index),
            and the x or y position to be analysed.

        Returns
        ----------
        activation_time_1d_array : np.array
            1D array of activation times along a single spatial axis.
        """
        # Calculation of the 2D array of activation times.
        activation_time_2d_array = SpatialAnalyser.activation_time_computing(data,
                                                                             index[parameters_analysis_dict["index"]],
                                                                             parameters_analysis_dict["threshold"])

        # Extracting a single spatial dimension from the activation time array.
        if "x" in parameters_analysis_dict:
            activation_time_1d_array = activation_time_2d_array[:, parameters_analysis_dict["x"]]
        elif "y" in parameters_analysis_dict:
            activation_time_1d_array = activation_time_2d_array[parameters_analysis_dict["y"], :]

        return activation_time_1d_array

    @staticmethod
    def latency_analyzing(data, index, parameters_analysis_dict):
        """Function that analyses latency based on a single spatial dimension.

        The latency is calculated in the 3D array and a centered index of a measurement of a condition. It is obtained
        in the form of a 2D array from which only the desired X or Y position can be taken to obtain a 1D array based
        on a single spatial dimension.

        To use the centered index, the user have to specify the name of the time index in the analysis parameter
        dictionary associated with the ‘index’ key. For example: “index”: ‘temporal_centered’. This index will consist
        of several indexes in the form of a list, as each spatial position will be associated with a different reference
        time. In the most basic case, this time is the moment when the moving object is at the center of the cell
        receptive field.

        Parameters
        ----------
        data : np.array
            3D array containing the values of a measurement for a given condition.

        index : dict of np.array
            Dictionary containing all the indexes of a MacularDictArray in the form of a 1D array.

        parameters_analysis_dict : dict
            Dictionary of parameters to be used for latency analysis. It must contain the threshold, the axis of the
            object's movement, the name of the index to be taken from the dictionary (allows switching from the s index
            to the ms index), and the x or y position to be analysed.

        Returns
        ----------
        activation_time_1d_array : np.array
            1D array of latency along a single spatial axis.
        """
        # Calculation of the 2D array of latency.
        latency_2d_array = SpatialAnalyser.latency_computing(data, index[parameters_analysis_dict["index"]],
                                                             parameters_analysis_dict["threshold"],
                                                             parameters_analysis_dict["axis"])

        # Extracting a single spatial dimension from the latency array.
        if "x" in parameters_analysis_dict:
            latency_1d_array = latency_2d_array[:, parameters_analysis_dict["x"]]
        elif "y" in parameters_analysis_dict:
            latency_1d_array = latency_2d_array[parameters_analysis_dict["y"], :]

        return latency_1d_array

    @staticmethod
    def time_to_peak_analyzing(data, index, parameters_analysis_dict):
        """Function that analyses time to peak based on a single spatial dimension.

        The time to peak is calculated in the 3D array and the index of a measurement of a condition. It is obtained
        in the form of a 2D array from which only the desired X or Y position can be taken to obtain a 1D array based
        on a single spatial dimension.

        Parameters
        ----------
        data : np.array
            3D array containing the values of a measurement for a given condition.

        index : dict of np.array
            Dictionary containing all the indexes of a MacularDictArray in the form of a 1D array.

        parameters_analysis_dict : dict
            Dictionary of parameters to be used for time to peak analysis. It must contain the name of the index to be
            taken from the dictionary (allows switching from the s index to the ms index), and the x or y position to be
            analysed.

        Returns
        ----------
        time_to_peak_1d_array : np.array
            1D array of time to peak along a single spatial axis.
        """
        # Calculation of the 2D array of time to peak.
        time_to_peak_2d_array = SpatialAnalyser.time_to_peak_computing(data, index[parameters_analysis_dict["index"]])

        # Extracting a single spatial dimension from the time to peak array.
        if "x" in parameters_analysis_dict:
            time_to_peak_1d_array = time_to_peak_2d_array[:, parameters_analysis_dict["x"]]
        elif "y" in parameters_analysis_dict:
            time_to_peak_1d_array = time_to_peak_2d_array[parameters_analysis_dict["y"], :]

        return time_to_peak_1d_array

    @staticmethod
    def peak_delay_analyzing(data, index, parameters_analysis_dict):
        """Function that analyses delay to peak based on a single spatial dimension.

        The delay to peak is calculated in the 3D array and a centered index of a measurement of a condition. It is
        obtained in the form of a 2D array from which only the desired X or Y position can be taken to obtain a 1D array
        based on a single spatial dimension.

        To use the centered index, the user have to specify the name of the time index in the analysis parameter
        dictionary associated with the ‘index’ key. For example: “index”: ‘temporal_centered’. This index will consist
        of several indexes in the form of a list, as each spatial position will be associated with a different reference
        time. In the most basic case, this time is the moment when the moving object is at the center of the cell
        receptive field.

        Parameters
        ----------
        data : np.array
            3D array containing the values of a measurement for a given condition.

        index : dict of np.array
            Dictionary containing all the indexes of a MacularDictArray in the form of a 1D array.

        parameters_analysis_dict : dict
            Dictionary of parameters to be used for delay to peak analysis. It must contain the threshold, the axis of
            the object's movement, the name of the index to be taken from the dictionary (allows switching from the s
            index to the ms index), and the x or y position to be analysed.

        Returns
        ----------
        activation_time_1d_array : np.array
            1D array of delay to peak along a single spatial axis.
        """
        # Calculation of the 2D array of delay_to_peak.
        delay_to_peak_2d_array = SpatialAnalyser.peak_delay_computing(data,
                                                                      index[parameters_analysis_dict["index"]],
                                                                      parameters_analysis_dict["axis"])

        # Extracting a single spatial dimension from the delay_to_peak array.
        if "x" in parameters_analysis_dict:
            delay_to_peak_1d_array = delay_to_peak_2d_array[:, parameters_analysis_dict["x"]]
        elif "y" in parameters_analysis_dict:
            delay_to_peak_1d_array = delay_to_peak_2d_array[parameters_analysis_dict["y"], :]

        return delay_to_peak_1d_array

    @staticmethod
    def peak_amplitude_analyzing(data, index, parameters_analysis_dict):
        """Function that analyses peak amplitude based on a single spatial dimension.

        The amplitude is calculated in the 3D array and the index of a measurement of a condition. It is obtained
        in the form of a 2D array from which only the desired X or Y position can be taken to obtain a 1D array based
        on a single spatial dimension.

        Parameters
        ----------
        data : np.array
            3D array containing the values of a measurement for a given condition.

        index : dict of np.array
            Dictionary containing all the indexes of a MacularDictArray in the form of a 1D array.

        parameters_analysis_dict : dict
            Dictionary of parameters to be used for amplitude analysis. It must contain only the x or y position to be
            analysed.

        Returns
        ----------
        amplitude_1d_array : np.array
            1D array of amplitude along a single spatial axis.
        """
        # Calculation of the 2D array of amplitude.
        amplitude_2d_array = SpatialAnalyser.peak_amplitude_computing(data)

        # Extracting a single spatial dimension from the amplitude array.
        if "x" in parameters_analysis_dict:
            amplitude_1d_array = amplitude_2d_array[:, parameters_analysis_dict["x"]]
        elif "y" in parameters_analysis_dict:
            amplitude_1d_array = amplitude_2d_array[parameters_analysis_dict["y"], :]

        return amplitude_1d_array
