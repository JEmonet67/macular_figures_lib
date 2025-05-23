import re
import copy
from functools import wraps

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

    levels_multiple_dictionaries : list of str or dict of str
        Container grouping all the names of conditions and measurements found in the associated multiple Macular Dict
        Array separated by ‘:’.

        The first element is a character string with conditions names separated by ":". The second is a dictionary
        associating the conditions of a multiple macular dict array as keys with values corresponding to the names
        of the measurements present in each of the MacularDictArray. The names of the measurements are also
        separated by ‘:’.

    Example
    ----------
    >>> instruction
    result instruction

    """

    def __init__(self, multi_macular_dict_array, multiple_dicts_analysis):
        """Summary

        Explanation

        Parameters
        ----------
        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with different MacularDictArray.

            Each MacularDictArray is defined by a set of data, indexes, a dictionary for configuring the simulation,
            and another for the pre-processing it has undergone.

        multiple_dicts_analysis : dict of dict
            Dictionaries containing all analyses to be performed for each dimension of the MacularAnalysisDataframes.

            The dictionary consists of a series of dictionaries included in the previous one, each representing a
            hierarchical level of the analysis to be performed. The keys of the first dictionary are those of the
            dimensions of the MacularAnalysisDataframes (‘Conditions’, ‘X’, ‘Y’, ‘Time’). The keys of the second
            dictionary are those of the analyses to be performed on the given dimension. For each of these analyses,
            there is a list of dictionaries, each representing a group of common analyses. These groups of common
            analyses are a conditions and measurements for which the same analysis is performed with the same
            parameters. The order in which the different dictionaries of common analysis groups are arranged in the
            list also defines the order in which they will be performed. Thus, if two common analysis groups modify the
            same line of a dataframe, it will be the last group in the list that will leave its value.

            The dictionaries for these groups of common analyses must contain a ‘conditions’ and ‘measurements’ key,
            both associated with all the names of the conditions and measurements included in that group. These names
            are in the form of a string where each name is separated by ‘:’ as in, for example:
            ‘barSpeed15dps:barSpeed30dps’ or ‘FiringRate_GanglionGainControl:VSDI’. The dictionary also contains a
            “params” key associated with a final dictionary level containing the parameters to be used for the analysis.
            The parameters to be used depend on the analysis, but all have a ‘flag’ parameter that allows add a suffix
            behind the name of the analysis used to create a new line in the corresponding dataframe in order to
            differentiate it from other similar analyses. For example, to differentiate two ‘activation_time’ with two
            different thresholds, two flags can be used:‘flag’:‘threshold0,1’ and ‘flag’:‘threshold0,05’ which gives two
            column names in the dataframe: ‘activation_time_threshold0,1’ and 'activation_time_threshold0,05’.

            There is a special case with the ‘MetaAnalysis’ key in the multiple analysis dictionary. This dictionary is
            used to perform analyses using other analyses already performed within the MacularAnalysisDataframes. In
            this case, the analysis key (second dictionary level) is associated with an additional dictionary level
            used for meta-analysis. This dictionary level contains a first key ‘params’ containing any parameters that
            do not depend on existing analyses. In addition, there are keys associated with the various arguments
            required to perform the meta-analysis function. These arguments are associated with a list of groups of
            common analyses in the form of dictionaries. These are all the analyses to be retrieved and used in the
            meta-analysis calculation. Their dictionary contains keys ‘conditions’, ‘measurements’, “dimensions” and
            ‘analyses’. Each of these is associated with a string of all the names of the common analysis group
            separated by ‘:’.

            Global aliases can be used for each element used in the common analysis group. These aliases allow you to
            retrieve all possible elements (conditions, measurements, dimensions, analyses). In the case of
            ‘measurements’ and 'analyses', the possible elements vary depending on the dimensions and conditions present
            in the multiple analysis group. Aliases must contain the prefix ‘all_’ followed by the element from which
            you want to retrieve everything. These aliases are substituted within the MacularAnalysisDataframes by the
            getter of the multiple analysis dictionary. Be careful to only group together analyses that share the same
            configurations.

            In certain specific cases, the analysis key may be associated only with a Boolean, an int or a float, as with
            the ‘sorting’ analysis for sorting condition names.
        """
        # Create and clean the multiple_dicts_analysis attributes.
        self.multiple_dicts_analysis = multiple_dicts_analysis
        self.multiple_dicts_analysis = self.cleaning_multiple_dicts_features(multiple_dicts_analysis)

        # Create and clean the multiple_dicts_simulations attributes.
        self._multiple_dicts_simulations = self.cleaning_multiple_dicts_features(
            {condition: multi_macular_dict_array[condition].dict_simulation for condition in multi_macular_dict_array})

        # Create and clean the multiple_dicts_preprocessings attributes.
        self._multiple_dicts_preprocessings = self.cleaning_multiple_dicts_features(
            {condition: multi_macular_dict_array[condition].dict_preprocessing for condition in
             multi_macular_dict_array})

        # Create dict_paths_pyb attributes to store each path_pyb associated to its condition.
        self._dict_paths_pyb = {}
        for condition in multi_macular_dict_array:
            self._dict_paths_pyb[condition] = multi_macular_dict_array[condition].path_pyb

        # Extract the conditions/measurements levels present in the MacularAnalysisDataframes.
        self._analysis_dataframes_levels = self.get_levels_of_multi_macular_dict_array(multi_macular_dict_array)

        # Regular expression to extract the name, value and unit of a condition with "NameValueUnit" format.
        self.condition_reg = re.compile("(^[A-Za-z]+)(-?[0-9]{1,4},?[0-9]{0,4})([A-Za-z]+$)")

        # Create the dataframes specified in the analysis dictionary.
        t_index = self.get_maximal_index_multi_macular_dict_array(multi_macular_dict_array, "temporal")
        x_index = self.get_maximal_index_multi_macular_dict_array(multi_macular_dict_array, "spatial_x")
        y_index = self.get_maximal_index_multi_macular_dict_array(multi_macular_dict_array, "spatial_y")
        self.initialize_dict_analysis_dataframes(x_index, y_index, t_index)

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
        raise AttributeError("The attribute dict_paths_pyb can't be modified.")

    @property
    def analysis_dataframes_levels(self):
        """Getter for the analysis_dataframes_levels attribute.
        """
        return self._analysis_dataframes_levels

    @analysis_dataframes_levels.setter
    def analysis_dataframes_levels(self, analysis_dataframes_levels):
        """Setter for the analysis_dataframes_levels attribute.
        """
        raise AttributeError("The attribute analysis_dataframes_levels can't be modified.")

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

        This getter allows you to keep the original multiple analysis dictionary while substituting the aliases ‘all_’
        it contains whenever you need to use the multiple analysis dictionary.
        """
        # Create a copy of the multiple analysis dictionaries.
        multiple_dicts_analysis_substituted = copy.deepcopy(self._multiple_dicts_analysis)

        # Substitution of all aliases ‘all_conditions’ and ‘all_measurements’ in the multiple analysis dictionaries.
        multiple_dicts_analysis_substituted = self.substituting_all_alias_in_multiple_analysis_dictionaries(
            multiple_dicts_analysis_substituted)

        return multiple_dicts_analysis_substituted

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
            for feature in multiple_dicts_features[dataframe]:
                # Removal of false list in features.
                if isinstance(multiple_dicts_features_cleaned[dataframe][feature], list):
                    for i_sub_feature in range(len(multiple_dicts_features_cleaned[dataframe][feature])):
                        if not multiple_dicts_features_cleaned[dataframe][feature][i_sub_feature]:
                            del multiple_dicts_features_cleaned[dataframe][feature][i_sub_feature]

                # Removal of false features.
                if not multiple_dicts_features_cleaned[dataframe][feature]:
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
        dict_levels_multi_macular_dict_array : dict of str or dict
            Dictionary grouping all the names of conditions and measurements found in the associated multiple Macular
            Dict Array separated by ‘:’.

            The first element "conditions" is a character string with conditions names separated by ":". The second
            "measurements" is a dictionary associating the conditions of a multiple macular dict array as keys with
            values corresponding to the names of the measurements present in each of the MacularDictArray. The names of
            the measurements are also separated by ‘:’.
        """
        # Create character string containing all the conditions of a multiple MacularDictArray separated by ‘:’.
        all_conditions = ":".join(sorted([condition for condition in self.dict_paths_pyb]))

        # Create dictionary associating each multiple MacularDictArray conditions with their "all_conditions".
        all_measurements = {
            condition: ":".join(sorted([measure for measure in multi_macular_dict_array[condition].data]))
            for condition in self.dict_paths_pyb}

        dict_levels_multi_macular_dict_array = {"conditions": all_conditions, "measurements": all_measurements}

        return dict_levels_multi_macular_dict_array

    def get_levels_of_macular_analysis_dataframes(self):
        """Extract and format all dimensions names and their analyses from a MacularAnalysisDictionary within strings
        of characters separated by ‘:’.

        The analyses that come into play could vary depending on the dimension of MacularAnalysisDictionary. This is why
        the structure used to store the names of the analyses is a dictionary associating the dimensions of the
        MacularAnalysisDictionary with their analyses.

        Examples : "X:Y" or "peak_amplitude:activation_time"

        Returns
        ----------
        dict_levels_macular_analysis_dataframes_correct : dict of str or dict
            Dictionary grouping all the names of dimensions and analyses found in the current MacularAnalysisDictionary
            separated by ‘:’.

            The first element "dimensions" is a character string with dimensions names separated by ":". The second
            "analyses" is a dictionary associating the dimensions of a multiple macular dict array as keys with
            values corresponding to the names of the analysis present in each of the MacularAnalysisDictionary
            dimensions. The names of the analyses are also separated by ‘:’.
        """
        # Create character string containing all the conditions of a multiple MacularDictArray separated by ‘:’.
        all_dimensions = ":".join(sorted(self.dict_analysis_dataframes.keys()))

        # Create dictionary associating each multiple MacularDictArray conditions with their "all_conditions".
        all_analyses = {
            dimension: ":".join(sorted(list(self.dict_analysis_dataframes[dimension].index)))
            if dimension == "Conditions"
            else {condition: ":".join(sorted(list(self.dict_analysis_dataframes[dimension][condition].index)))
                  for condition in self.dict_analysis_dataframes[dimension]}
            for dimension in self.dict_analysis_dataframes.keys()}

        dict_levels_macular_analysis_dataframes_correct = {"dimensions": all_dimensions, "analyses": all_analyses}

        return dict_levels_macular_analysis_dataframes_correct

    def substituting_all_alias_in_multiple_analysis_dictionaries(self, multiple_dicts_analysis):
        """Function replacing all aliases ‘all_conditions’ and ‘all_measurements’ in multiple analysis dictionaries.

        Parameters
        ----------
        multiple_dicts_analysis : dict of MacularDictArray
            Copy of the multiple analysis dictionaries contained in the current MacularAnalysisDataframes.

        Returns
        ----------
        multiple_dicts_analysis : dict of dict
            Multiple analysis dictionary with "all_conditions" and "all_measurements" aliases substituted.
        """
        # Double loop to browse the dimensions and analyses contained in the multiple analysis dictionary.
        for dimension in multiple_dicts_analysis:
            for analysis in multiple_dicts_analysis[dimension]:
                # Case of meta-analyses.
                if dimension == "MetaAnalysis":
                    # Loop on the arguments of the meta-analysis function.
                    for meta_analysis_arguments in multiple_dicts_analysis["MetaAnalysis"][analysis]:
                        if meta_analysis_arguments is not "params":
                            # Verify that common analysis groups are organised within a list.
                            if isinstance(multiple_dicts_analysis[dimension][analysis][meta_analysis_arguments], list):
                                # Loop on common analysis groups.
                                for common_group_analysis in (
                                        multiple_dicts_analysis)[dimension][analysis][meta_analysis_arguments]:
                                    # Verification that we are dealing with the dictionary of a common analysis groups.
                                    if isinstance(common_group_analysis, dict):
                                        self.substituting_all_alias_in_common_analysis_group_dictionary(common_group_analysis)
                else:
                    # Verify that common analysis groups are organised within a list.
                    if isinstance(multiple_dicts_analysis[dimension][analysis], list):
                        # Loop on common analysis groups.
                        for common_group_analysis in multiple_dicts_analysis[dimension][analysis]:
                            # Verification that we are dealing with the dictionary of a common analysis groups.
                            if isinstance(common_group_analysis, dict):
                                self.substituting_all_alias_in_common_analysis_group_dictionary(common_group_analysis)

        return multiple_dicts_analysis

    def substituting_all_alias_in_common_analysis_group_dictionary(self, common_analysis_group_dictionary):
        """Function that allows global aliases in the dictionary of a group of common analyses to be substituted.

        Global aliases are aliases that are used when you do not want to define specific elements of a hierarchical
        level of the analysis and instead want to consider all possible elements. The different hierarchical levels are
        conditions, measures, dimensions and analyses. Global aliases are all designed with the hierarchical level
        preceded by the suffix ‘all_’. So we have ‘all_conditions’, ‘all_measurements’, ‘all_dimensions’ and
        ‘all_analyses’. In the cases of “all_measurements” and ‘all_analyses’, the names retrieved depend on the
        dimension or condition considered.

        Parameters
        ----------
        common_analysis_group_dictionary : dict
            Dictionary for a group of common analyses. It must contain a ‘conditions’ key and a “measurements” key
            associated with the measurement or conditions included in the group of common analyses, separated by ‘:’.
            In the case of a group of common analyses for meta-analyses, it must also contain the ‘dimensions’ and
            'analyses' keys associated with the dimension(s) and analyses of the group, again separated by ‘:’. Each of
            these keys may be associated with an alias allowing all names associated with this hierarchical level of
            the MacularAnalysisDataframes to be included.
            Example : {"dimensions": "X", "conditions": "all_conditions", "measurements": "all_measurements",
                          "analyses": "peak_amplitude:activation_time:time_to_peak"}
        """
        # Replace the alias for all conditions in a multiple MacularDictArray.
        if common_analysis_group_dictionary["conditions"] == "all_conditions":
            common_analysis_group_dictionary["conditions"] = self.analysis_dataframes_levels["conditions"]

        # Replace the alias for all measurements in a multiple MacularDictArray.
        if common_analysis_group_dictionary["measurements"] == "all_measurements":
            common_analysis_group_dictionary["measurements"] = self.analysis_dataframes_levels["measurements"][common_analysis_group_dictionary[
                "conditions"].split(":")[0]]

        # Substitution of the alias for all dimensions in the case of the Meta-analyses analysis dictionary.
        try:
            if common_analysis_group_dictionary["dimensions"] == "all_dimensions":
                common_analysis_group_dictionary["dimensions"] = self.analysis_dataframes_levels["dimensions"]
        except KeyError:
            pass

        # Substitution of the alias for all analyses in the case of the Meta-analyses analysis dictionary.
        try:
            if common_analysis_group_dictionary["analyses"] == "all_analyses":
                if common_analysis_group_dictionary["dimensions"].split(":")[0] == "Conditions":
                    common_analysis_group_dictionary["analyses"] = self.analysis_dataframes_levels["analyses"][common_analysis_group_dictionary[
                        "dimensions"].split(":")[0]]
                else:
                    common_analysis_group_dictionary["analyses"] = self.analysis_dataframes_levels["analyses"][common_analysis_group_dictionary[
                        "dimensions"].split(":")[0]][common_analysis_group_dictionary["conditions"].split(":")[0]]
        except KeyError:
            pass

    def make_spatial_dataframes_analysis(self, dimension, multi_macular_dict_array):
        """Function used to perform all MacularAnalysisDataframe analyses to be carried out in the spatial dimension
        (X and Y).

        The names of all analyses in the multiple analysis dictionaries are scanned and identified. For each of them, a
        conditional block allows the corresponding analysis function to be executed. All these functions take as inputs
        the current MacularAnalysisDataframes, the multiple macular dict array, the dimension, and the current analysis.

        Parameters
        ----------
        dimension : str
            Dimension of the MacularAnalysisDataframes in which the result of the current analysis is stored
            ("X" or "Y").

        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with different MacularDictArray.
        """
        # Dictionary containing all spatial analyses currently implemented.
        available_spatial_analyses_dict = {
            "activation_time": self.activation_time_analyzing,
            "latency": self.latency_analyzing,
            "time_to_peak": self.time_to_peak_analyzing,
            "peak_delay": self.peak_delay_analyzing,
            "peak_amplitude": self.peak_amplitude_analyzing
        }

        # Performs all analyses listed in the current analysis dictionary.
        for analysis in self.multiple_dicts_analysis[dimension]:
            if analysis in available_spatial_analyses_dict:
                available_spatial_analyses_dict[analysis](self, multi_macular_dict_array, dimension, analysis)

        return sorted_list_elements

    def make_conditions_dataframes_analysis(self, multi_macular_dict_array):
        """Function used to perform all MacularAnalysisDataframe analyses to be carried out in the conditions dimension.

        The names of all analyses in the multiple analysis dictionaries are scanned and identified. For each of them, a
        conditional block allows the corresponding analysis function to be executed. All these functions take as inputs
        the current MacularAnalysisDataframes, the multiple macular dict array, the dimension, and the current analysis.

        Parameters
        ----------
        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with different MacularDictArray.
        """
        dimension = "Conditions"

        # Dictionary containing all conditions analyses currently implemented.
        available_spatial_analyses_dict = {
            # "maximal_latency": self.maximal_latency_analyzing,
            # "anticipation_range": self.anticipation_range_analyzing,
            # "short_range_anticipation_speed": self.short_range_anticipation_speed_analyzing,
            # "long_range_anticipation_speed": self.long_range_anticipation_speed_analyzing,
            # "peak_speed": self.peak_speed_analyzing,
            # "stationary_peak_delay": self.stationary_peak_delay_analyzing,
            "peak_amplitude": self.peak_amplitude_analyzing
        }

        # Performs all analyses listed in the current analysis dictionary.
        for analysis in self.multiple_dicts_analysis[dimension]:
            if analysis in available_spatial_analyses_dict:
                available_spatial_analyses_dict[analysis](self, multi_macular_dict_array, dimension, analysis)


    @staticmethod
    def analysis(analysis_function):
        """Decorator for functions used to perform a specific analysis of a multiple analysis dictionary.

        Parameters
        ----------
        analysis_function : function
            Analysis function to apply to calculate the current analysis.

            This analysis function changes between each analysis, so they must all have the same three input
            arguments: the data, the index and the analysis parameters.
        """

        @wraps(analysis_function)
        def modified_analysis_function(macular_analysis_dataframes, multi_macular_dict_array, dimension, analysis):
            """Function applied within the decorator, prior to the analysis function, to process each group of common
            analyses and analyse each of their pairs of conditions/measurements.

            Depending on the type of analysis being performed, the name of the analysis and the associated function are
            modified. The global analysis is performed by carrying out specific analyses for groups of common analyses.
            Each of these groups combines conditions and measurements that share the same parameter values for the
            current analysis. These common analysis groups are presented in the form of dictionaries contained in a
            list. The dictionaries must contain a ‘conditions’ key, “measurements” associated with the names of the
            conditions and measurements in the group separated by ‘:’ and a ‘params’ key containing all the parameters
            to be used for the current analysis.

            The function goes through the list of common analysis groups in the order decided by the user. This order
            is important because if two common analysis groups act on one or more identical rows of dataframes, then
            the last common analysis group processed will leave its value.

            Parameters
            ----------
            macular_analysis_dataframes : MacularAnalysisDataframes
                Macular analysis dataframes that the user want to do one analyse in one of its dimensions.

            multi_macular_dict_array : dict of MacularDictArray
                Dictionary associating specific conditions with MacularDictArray.

            dimension : str
                Dimension of the MacularAnalysisDataframes in which the result of the current analysis is stored.

            analysis : str
                Name of the current analysis.
            """
            # Loop allowing to browse the conditions and measurements of common analysis groups.
            for common_analysis_group_dict in macular_analysis_dataframes.multiple_dicts_analysis[dimension][analysis]:
                grouped_conditions = common_analysis_group_dict["conditions"]
                grouped_measurements = common_analysis_group_dict["measurements"]
                # Extract the list of condition/measurements pairs to be analysed with the same parameters.
                common_analysis_group_generator = macular_analysis_dataframes.common_analysis_group_parser(
                    grouped_conditions, grouped_measurements)
                # Analysis of conditions/measurements for a common analysis group sharing the same parameters.
                macular_analysis_dataframes.make_common_group_analysis(
                    analysis_function, multi_macular_dict_array,
                    common_analysis_group_generator, dimension, analysis, common_analysis_group_dict["params"])

        return modified_analysis_function

    @staticmethod
    def common_analysis_group_parser(grouped_conditions, grouped_measurements):
        """Function that transforms the names of conditions and measurements in a group of common analyses into a
        generator of pairs of conditions and measurements that share one or more identical analyses.

        The sets of condition names or measurements in the common analysis group (grouped_conditions and
        grouped_measurements) are separated by ‘:’. To extract each of them, separate them using this symbol.

        Examples : "barSpeed27dps:barSpeed30dps" or "BipolarResponse_BipolarGainControl:VSDI"

        Parameters
        ----------
        grouped_conditions : str
            Names of conditions in a common analysis group.

        grouped_measurements : str
            Names of measurements in a common analysis group.
        """
        # Loop on the conditions and measurements of the common analysis group.
        for condition in grouped_conditions.split(":"):
            for measurement in grouped_measurements.split(":"):
                # New pair condition, measurement for analysis in generator.
                yield condition, measurement

    def make_common_group_analysis(self, analysis_function, multi_macular_dict_array, common_analysis_group_generator,
                                   dimension, analysis, parameters_analysis_dict):
        """Function that performs a given analysis within a common group of analyses

        A common analysis group is a bunch of conditions and measurements that share one or more identical analyses
        with the same parameters. This group can be represented by a generator of pairs of conditions and measurements
        on which to perform these same analyses.

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

        common_analysis_group_generator : list of tuples
            Generator of all tuples pairs of conditions and measurements in a group of common analyses.

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
            if parameters_analysis_dict['flag'] != "":
                str_parameters_analysis = f"_{parameters_analysis_dict['flag']}"
            else:
                str_parameters_analysis = ""
        except KeyError:
            str_parameters_analysis = ""

        # Loop of conditions and measurements of the common analysis group.
        for condition, measurement in common_analysis_group_generator:
            # Defines the name of the line where the current analysis is stored.
            dataframe_row = f"{analysis}_{measurement}{str_parameters_analysis}"
            # Conducting an analysis of a given condition and measurement in the conditions dataframe.
            if dimension == "Conditions":
                self.dict_analysis_dataframes[dimension].loc[dataframe_row, condition] = analysis_function(
                    multi_macular_dict_array[condition].data[measurement], multi_macular_dict_array[condition].index
                    , parameters_analysis_dict)
            # Conducting an analysis of a given condition and measurement in spatio-temporal dataframes.
            else:
                self.dict_analysis_dataframes[dimension][condition].loc[dataframe_row] = analysis_function(
                    multi_macular_dict_array[condition].data[measurement], multi_macular_dict_array[condition].index
                    , parameters_analysis_dict)

    @staticmethod
    @analysis
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
    @analysis
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
    @analysis
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
    @analysis
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
            Dictionary of parameters to be used for delay to peak analysis. It must contain the axis of the object's
            movement, the name of the index to be taken from the dictionary (allows switching from the s index to the ms
            index), and the x or y position to be analysed.

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
    @analysis
    def peak_amplitude_analyzing(data, index, parameters_analysis_dict):
        """Function that analyses peak amplitude based on a single spatial or conditions dimension.

        The amplitude is calculated in the 3D array and the index of a measurement of a condition. It is obtained
        in the form of a 2D array from which only the desired X or Y positions can be taken. This can be a 1D array with
        a single spatial dimension or a single value at a specific position.

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
        amplitude : np.array or float
            1D array of amplitude along a single spatial axis or value of peak amplitude at a specific spatial position.
        """
        # Calculation of the 2D array of amplitude.
        amplitude_2d_array = SpatialAnalyser.peak_amplitude_computing(data)

        # Extracting a single spatial dimension from the amplitude array.
        if "x" in parameters_analysis_dict and "y" not in parameters_analysis_dict:
            amplitude = amplitude_2d_array[:, parameters_analysis_dict["x"]]
        elif "x" not in parameters_analysis_dict and "y" in parameters_analysis_dict:
            amplitude = amplitude_2d_array[parameters_analysis_dict["y"], :]

        # Extracting a single spatial position from the amplitude array.
        elif "x" in parameters_analysis_dict and "y" in parameters_analysis_dict:
            amplitude = amplitude_2d_array[parameters_analysis_dict["y"], parameters_analysis_dict["x"]]

        return amplitude
