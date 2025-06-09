import re
import copy
from functools import wraps

import numpy as np
import pandas as pd

from src.data_manager.MacularDictArray import MacularDictArray
from src.data_manager.MetaAnalyser import MetaAnalyser
from src.data_manager.SpatialAnalyser import SpatialAnalyser
from src.data_manager.ConditionsAnalyser import ConditionsAnalyser


class MacularAnalysisDataframes:
    """Summary

    Explanation

    Attributes
    ----------
    dict_paths_pyb : dict of str
        Dictionnaire associant les conditions d'un MacularDictArray multiple avec le path de leur fichier pyb.

    dict_analysis_dataframes : dict of pd.DataFrame
        Summary attr1

    multiple_dicts_analysis : dict of dict
        Dictionaries containing all analyses to be performed for each dimension of the MacularAnalysisDataframes.

        The dictionary consists of a series of dictionaries included in the previous one, each representing a
        hierarchical level of the analysis to be performed. The keys of the first dictionary are those of the
        dimensions of the MacularAnalysisDataframes (‘Conditions’, ‘X’, ‘Y’, ‘Time’). The keys of the second
        dictionary are those of the analyses to be performed on the given dimension. For each of these analyses,
        there is a list of dictionaries, each representing a group of common analyses. These groups of common
        analyses are a conditions and measurements for which the same analysis is performed with the same parameters.
        The order in which the different dictionaries of common analysis groups are arranged in the list also defines
        the order in which they will be performed. Thus, if two common analysis groups modify the same line of a
        dataframe, it will be the last group in the list that will leave its value.

        The dictionaries for these groups of common analyses must contain a ‘conditions’ and ‘measurements’ key, both
        associated with all the names of the conditions and measurements included in that group. These names are in
        the form of a string where each name is separated by ‘:’ as in, for example: ‘barSpeed15dps:barSpeed30dps’
         or ‘FiringRate_GanglionGainControl:VSDI’. The dictionary also contains a “params” key associated with a
        final dictionary level containing the parameters to be used for the analysis. The parameters to be used
        depend on the analysis, but all have a ‘flag’ parameter that allows add a suffix behind the name of the
        analysis used to create a new line in the corresponding dataframe in order to differentiate it from other
        similar analyses. For example, to differentiate two ‘activation_time’ with two different thresholds, two
        flags can be used:‘flag’:‘threshold0,1’ and ‘flag’:‘threshold0,05’ which gives two column names in the
        dataframe: ‘activation_time_threshold0,1’ and 'activation_time_threshold0,05’.

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

    condition_reg : re.Pattern
        Regular expression to extract the name of the condition, its value and its unit from the keys of the
        MacularDictArray multiple dictionary.

        This pattern is primarily used to sort conditions in the ‘Conditions’ dataframe of MacularAnalysisDataframes.
        By default, the regular expression entered allows you to read conditions that follow a ‘NameValueUnit’ format.

    analysis_dataframes_levels : dict of str or dict of dict
        Container grouping all the names of conditions, measurements, dimensions and analyses found in the
        MacularAnalysisDataframes or associated multiple Macular Dict Array, separated by ‘:’.

        The first key "conditions" is associated to a character string with conditions names separated by ":". The
        second key "measurements" is a dictionary associating the conditions of a multiple macular dict array as keys
        with values corresponding to the names of the measurements present in each of the MacularDictArray. The names
        of the measurements are also separated by ‘:’. La troisième clé "dimensions" est associée à une chaîne de caractères
        avec toutes les dataframes de dimension contenu dans le MacularAnalysisDataframes actuel. La dernière clé "analyses"
        est aussi un dictionnaire contenant des les noms des dimensions associés à un second dictionnaire avec des pairs
        de noms de conditions et des noms des analyses dans le dataframe de cette dimension et de cette condition.

        Example :
        {
        'conditions': 'barSpeed6dps:barSpeed15dps:barSpeed30dps',
        'measurements': {
        'barSpeed6dps': 'BipolarResponse_BipolarGainControl:VSDI',
        'barSpeed15dps': 'FiringRate_GanglionGainControl:VSDI',
        'barSpeed30dps': 'VSDI',
        },
        'dimensions': 'Conditions:Time:X:Y',
        'analyses': {
            'Conditions': 'barSpeed (dps)',
            'X': {'barSpeed6dps': '', 'barSpeed15dps': '', 'barSpeed30dps': ''},
            'Y': {'barSpeed6dps': '', 'barSpeed15dps': '', 'barSpeed30dps': ''},
            'Time': {'barSpeed6dps': '', 'barSpeed15dps': '', 'barSpeed30dps': ''}
            }
        }

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

        # Implementation of the MacularAnalysisDataframes index dictionary.
        dict_index = self.setup_index_dictionary(multi_macular_dict_array)

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
            Dictionary associating specific conditions with different MacularDictArray.

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
        x_index : np.ndarray
            Index of the x-axis used for the x-axis dataframe.

            This parameter is facultative.

        y_index : np.ndarray
            Index of the y-axis used for the y-axis dataframe.

            This parameter is facultative.

        t_index : np.ndarray
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
            # Creation of the overall conditions dataframe.
            elif name_dataframe == "MetaAnalysis":
                name_dataframe = "MetaConditions"
                sorted_conditions = ["overall"]
                self.dict_analysis_dataframes[name_dataframe] = self.initialize_analysis_dataframe(
                    sorted_conditions, name_dataframe)
                self.setup_conditions_values_to_condition_dataframe()

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
            Dictionary associating specific conditions with different MacularDictArray.

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
            if dimension == "Conditions" or dimension == "MetaConditions"
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
                    # Loop on the common meta-analysis groups list.
                    for common_meta_analysis_group in multiple_dicts_analysis["MetaAnalysis"][analysis]:
                        # Loop on the arguments of the meta-analysis function.
                        for meta_analysis_arguments in common_meta_analysis_group:
                            if meta_analysis_arguments != "params":
                                # Substitution of aliases in the common analysis group for each meta-analysis argument.
                                self.substituting_all_alias_in_common_analysis_group_dictionary(
                                    common_meta_analysis_group[meta_analysis_arguments])
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
        conditions, measurements, dimensions and analyses. Global aliases are all designed with the hierarchical level
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
            common_analysis_group_dictionary["measurements"] = self.analysis_dataframes_levels["measurements"][
                common_analysis_group_dictionary["conditions"].split(":")[0]]

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
                    common_analysis_group_dictionary["analyses"] = self.analysis_dataframes_levels["analyses"][
                        common_analysis_group_dictionary[
                            "dimensions"].split(":")[0]]
                else:
                    common_analysis_group_dictionary["analyses"] = \
                        self.analysis_dataframes_levels["analyses"][common_analysis_group_dictionary[
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

        # Performs all spatial analyses listed in the current analysis dictionary.
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
            "peak_amplitude": self.peak_amplitude_analyzing
        }

        # Performs all conditions analyses listed in the current analysis dictionary.
        for analysis in self.multiple_dicts_analysis[dimension]:
            if analysis in available_spatial_analyses_dict:
                available_spatial_analyses_dict[analysis](self, multi_macular_dict_array, dimension, analysis)

    # TODO test
    def make_meta_analysis_dataframes_analysis(self, dict_index):
        """Function used to perform all MacularAnalysisDataframe meta-analyses.

        The names of all meta-analyses type in the multiple analysis dictionaries are scanned and identified. For each
        of them, a conditional block allows the corresponding analysis function to be executed. All these functions
        take as inputs the current MacularAnalysisDataframes, the dimension, and the current analysis.
        """
        # Dictionary containing all meta-analyses type currently implemented.
        available_spatial_analyses_dict = {
            "peak_speed": self.peak_speed_analyzing,
            "stationary_peak_delay": self.stationary_peak_delay_analyzing,
            "linear_fit": self.linear_fit_analyzing,
            "anticipation_fit": self.anticipation_fit_analyzing,
            "maximal_latency": self.maximal_latency_analyzing,
            "normalization": self.normalization_analyzing,
            "subtraction": self.subtraction_analyzing
        }

        # Performs all meta-analyses type listed in the current analysis dictionary.
        for meta_analysis_type in self.multiple_dicts_analysis["MetaAnalysis"]:
            if meta_analysis_type in available_spatial_analyses_dict:
                available_spatial_analyses_dict[meta_analysis_type](self, meta_analysis_type, dict_index)

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
                Dictionary associating specific conditions with different MacularDictArray.

            dimension : str
                Dimension of the MacularAnalysisDataframes in which the result of the current analysis is stored.

            analysis : str
                Name of the current analysis.

            Returns
            ----------
            analysis_function : function
                Decorated analysis function to apply to calculate the current analysis.
            """
            # Loop allowing to browse the conditions and measurements of common analysis groups.
            for common_analysis_group_dict in macular_analysis_dataframes.multiple_dicts_analysis[dimension][analysis]:
                # Extract the list of condition/measurements pairs to be analysed with the same parameters.
                common_analysis_group_generator = MacularAnalysisDataframes.common_analysis_group_parser(
                    [common_analysis_group_dict["conditions"], common_analysis_group_dict["measurements"]])
                # Analysis of conditions/measurements for a common analysis group sharing the same parameters.
                macular_analysis_dataframes.make_common_group_analysis(
                    analysis_function, multi_macular_dict_array,
                    common_analysis_group_generator, dimension, analysis, common_analysis_group_dict["params"])

        return modified_analysis_function

    @staticmethod
    def common_analysis_group_parser(list_grouped_levels, list_analysis_levels=None, n=0):
        """Recursive function for creating a unique association generator for hierarchical levels.

        The goal is to create all possible unique tuples of hierarchical level names from a given number of different
        names for each hierarchical level. These different names are provided in the grouped level list. Each item in
        the list is a different name for a given level separated by ‘:’. All levels are classified in a hierarchical
        order defined within this list of grouped levels.

        Parameters
        ----------
        list_grouped_levels : list of str
            List containing different hierarchical level names separated by ‘:’. For example: ‘X:Y’ or
            ‘barSpeed30dps:barSpeed27dps’. The number of hierarchical levels contained in this list will define the
            depth of recursion.

        list_analysis_levels : list of str
            List containing all the names of an association of a hierarchical level. This list is constructed at each
            hierarchical level and is reset between each association of a different hierarchical level. The list is
            transformed into a tuple once it is complete.

        n : int
            Hierarchical level counter in the analysis dictionary that increases with recursion. It is always equal to
            the length of the grouped levels list.
        """
        # Loop on the current hierarchical level of the current analysis.
        for level in list_grouped_levels[n].split(":"):
            # Initialisation of the list of analysis levels when at level 0.
            if not n:
                list_analysis_levels = []
            # Increment the list of analysis levels with the current level.
            new_list_current_analysis_levels = list_analysis_levels + [level]

            # Call the recursive function to go down one level if you are not at the last level.
            if n < len(list_grouped_levels) - 1:
                yield from MacularAnalysisDataframes.common_analysis_group_parser(list_grouped_levels,
                                                                                  new_list_current_analysis_levels,
                                                                                  n + 1)
            else:
                # Returns the analysis level tuple to the generator once the last level has been reached.
                yield tuple(new_list_current_analysis_levels)

    def make_common_group_analysis(self, analysis_function, multi_macular_dict_array, common_analysis_group_generator,
                                   dimension, analysis, common_parameters_analysis_dict):
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
            Dictionary associating specific conditions with different MacularDictArray.

        common_analysis_group_generator : list of tuples
            Generator of all tuples pairs of conditions and measurements in a group of common analyses.

        dimension : str
            Dimension of the MacularAnalysisDataframes in which the result of the current analysis is stored.

        analysis : str
            Name of the current analysis.

        common_parameters_analysis_dict : dict
            Dictionary containing all parameters of the current analysis.

            The parameters vary depending on the analysis, except for the optional ‘flag’ parameter, which can always be
            present. It corresponds to a suffix used to specify the name of the dataframe row that will be added.
        """
        # Loop of conditions and measurements of the common analysis group.
        for condition, measurement in common_analysis_group_generator:
            # Defines the name of the line where the current analysis is stored.
            dataframe_row = f"{analysis}_{measurement}_{common_parameters_analysis_dict['flag']}".strip("_")
            # Conducting an analysis of a given condition and measurement in the conditions dataframe.
            if dimension == "Conditions":
                self.dict_analysis_dataframes[dimension].loc[dataframe_row, condition] = analysis_function(
                    multi_macular_dict_array[condition].data[measurement], multi_macular_dict_array[condition].index
                    , common_parameters_analysis_dict)
            # Conducting an analysis of a given condition and measurement in spatio-temporal dataframes.
            else:
                self.dict_analysis_dataframes[dimension][condition].loc[dataframe_row] = analysis_function(
                    multi_macular_dict_array[condition].data[measurement], multi_macular_dict_array[condition].index
                    , common_parameters_analysis_dict)

    @staticmethod
    @analysis
    def activation_time_analyzing(data, index, parameters_analysis_dict):
        """Function that analyses activation time based on a single spatial dimension.

        The activation time is calculated in the 3D array and the index of a measurement of a condition. It is obtained
        in the form of a 2D array. From this array it's possible to only take the desired X or Y position to obtain a
        1D array based on a single spatial dimension (spatial analysis decorator). The activation time can be calculated
        either from a fixed threshold value or dynamically by adjusting the threshold value for each spatial position.

        Parameters
        ----------
        data : np.ndarray
            3D array containing the values of a measurement for a given condition.

        index : dict of np.ndarray
            Dictionary containing all the indexes of a MacularDictArray in the form of a 1D array.

        parameters_analysis_dict : dict
            Dictionary of parameters to be used for activation time analysis. It must contain the threshold, the type of
            threshold (dynamic or static), the name of the index to be taken from the dictionary (allows switching from
            the s index to the ms index), and the x or y position to be analysed.

        Returns
        ----------
        activation_time_array : np.ndarray
            1D array of activation times along a single spatial axis.
        """
        # Switch between static or dynamic thresholding.
        if parameters_analysis_dict["threshold_type"] == "static":
            threshold = parameters_analysis_dict["threshold"]

        elif parameters_analysis_dict["threshold_type"] == "dynamic":
            threshold = SpatialAnalyser.dynamic_threshold_computing(data, parameters_analysis_dict["threshold"])

        # Calculation of the 2D array of activation times.
        activation_time_2d_array = SpatialAnalyser.activation_time_computing(data,
                                                                             index[parameters_analysis_dict["index"]],
                                                                             threshold)

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
        on a single spatial dimension. The latency can be calculated either from a fixed threshold value or dynamically
        by adjusting the threshold value for each spatial position.

        To use the centered index, the user have to specify the name of the time index in the analysis parameter
        dictionary associated with the ‘index’ key. For example: “index”: ‘temporal_centered’. This index will consist
        of several indexes in the form of a list, as each spatial position will be associated with a different reference
        time. In the most basic case, this time is the moment when the moving object is at the center of the cell
        receptive field.

        Parameters
        ----------
        data : np.ndarray
            3D array containing the values of a measurement for a given condition.

        index : dict of np.ndarray
            Dictionary containing all the indexes of a MacularDictArray in the form of a 1D array.

        parameters_analysis_dict : dict
            Dictionary of parameters to be used for latency analysis. It must contain the threshold, the type of
            threshold (dynamic or static),the axis of the object's movement, the name of the index to be taken from the
            dictionary (allows switching from the s index to the ms index), and the x or y position to be analysed.

        Returns
        ----------
        latency_1d_array : np.ndarray
            1D array of latency along a single spatial axis.
        """
        # Switch between static or dynamic thresholding.
        if parameters_analysis_dict["threshold_type"] == "static":
            threshold = parameters_analysis_dict["threshold"]

        elif parameters_analysis_dict["threshold_type"] == "dynamic":
            threshold = SpatialAnalyser.dynamic_threshold_computing(data, parameters_analysis_dict["threshold"])

        # Calculation of the 2D array of latency.
        latency_2d_array = SpatialAnalyser.latency_computing(data, index[parameters_analysis_dict["index"]],
                                                             threshold, parameters_analysis_dict["axis"])

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
        data : np.ndarray
            3D array containing the values of a measurement for a given condition.

        index : dict of np.ndarray
            Dictionary containing all the indexes of a MacularDictArray in the form of a 1D array.

        parameters_analysis_dict : dict
            Dictionary of parameters to be used for time to peak analysis. It must contain the name of the index to be
            taken from the dictionary (allows switching from the s index to the ms index), and the x or y position to be
            analysed.

        Returns
        ----------
        time_to_peak_1d_array : np.ndarray
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
        data : np.ndarray
            3D array containing the values of a measurement for a given condition.

        index : dict of np.ndarray
            Dictionary containing all the indexes of a MacularDictArray in the form of a 1D array.

        parameters_analysis_dict : dict
            Dictionary of parameters to be used for delay to peak analysis. It must contain the axis of the object's
            movement, the name of the index to be taken from the dictionary (allows switching from the s index to the ms
            index), and the x or y position to be analysed.

        Returns
        ----------
        activation_time_1d_array : np.ndarray
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
        data : np.ndarray
            3D array containing the values of a measurement for a given condition.

        index : dict of np.ndarray
            Dictionary containing all the indexes of a MacularDictArray in the form of a 1D array.

        parameters_analysis_dict : dict
            Dictionary of parameters to be used for amplitude analysis. It must contain only the x and/or y position to
            be analysed.

        Returns
        ----------
        amplitude : np.ndarray or float
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

    # TODO
    @staticmethod
    def meta_analysis(meta_analysis_function):
        """Decorator for functions used to perform a specific meta-analysis of a multiple analysis dictionary.

        Parameters
        ----------
        meta_analysis_function : function
            Analysis function to apply to calculate the current meta-analysis.

            This meta-analysis function changes between each meta-analysis, so they must all have the same three input
            arguments: the data, the index and the analysis parameters. However, the data may be in different forms
            depending on the function of the meta-analysis.
        """

        @wraps(meta_analysis_function)
        def modified_meta_analysis_function(macular_analysis_dataframes, meta_analysis_type, dict_index):
            # TODO Test, relire commentaires
            """Function applied within the decorator, prior to the meta-analysis function, to process each group of common
            analyses and analyse each of their pairs of conditions/measurements.

            Depending on the type of meta-analysis being performed, the name of the meta-analysis and the associated
            function are modified. The global analysis is performed by carrying out specific analyses for groups
            of common analyses. Each of these groups combines conditions and measurements that share the same
            parameter values for the current analysis. These common analysis groups are presented in the form of
            dictionaries contained in a list. The dictionaries must contain a ‘conditions’ key, “measurements” associated with the names of the
            conditions and measurements in the group separated by ‘:’ and a ‘params’ key containing all the parameters
            to be used for the current analysis.

            The function goes through the list of common analysis groups in the order decided by the user. This order
            is important because if two common analysis groups act on one or more identical rows of dataframes, then
            the last common analysis group processed will leave its value.

            Lors d'une méta-analyse, l'ensemble des analyses contenues dans les listes des groupes communs d'analyses
            associés à chaque arguments de la fonction de Meta-analyses sont récupérées une par une. L'ordre dans lequel sont
            renseignées les différentes analyses utilisées par chaque argument est donc crucial. Si la méta-analyse nécessite
            plusieurs arguments, le nombre d'analyses contenues dans leurs groupes d'analyses communes respectives doivent
            être identiques. Il est aussi possible qu'un argument soit associé à une unique analyse, dans ce cas elle sera
            ré-utilisée pour toutes les méta-analyses s'il y en a plus d'une.

            Chaque analyses d'un MacularAnalysisDataframes peut être définie par les 4 niveaux hiérarchiques d'un
            MacularAnalysisDataframes. On a la dimension ("X", "Y", "Conditions"), la condition ("barSpeed30dps",
            "ampGang30Hz"), la mesure ("VSDI", "FiringRate_GanglionGainControl") et le type de l'analyse ("latency",
            "peak_amplitude"). Il faut donc utiliser ces 4 niveaux pour localiser et extraire une analyse. A cela
            s'ajoute aussi le nom du flag s'il y en a un.

            # Le dictionnaire de méta-analyse est parser pour créer une liste de méta-analyses communes dont la
            # longueur doit être la même pour chaque analyses utilisée dans la méta-analyse en cours.
            # Si il y a une seule analyse elle sera répétée pour tous les groupes de méta-analyses.
            # Les groupes de méta-analyses communes sont sous la forme d'un tuple
            # (dimension, condition, mesure, analyse, étiquette)
            # Les groupes de méta-analyses communes fonctionnent comme les groupes d'analyses communes, elles regroupent un
            # ensemble de méta-analyses identiques avec les mêmes paramètres et à réaliser sur tout un lot d'input.
            # Seul ces inputs varient. Le but est donc de récupérer tout ces inputs avant de les traiter identiquement.

            Le dictionnaire de méta-analyse peut contenir deux manières de définir des outputs, la manière utilisée dépend
            avant tout de la méta-analyse considérée. Certaine méta-analyses n'auront pas besoin de définir complètement
            un nouvel output.

            Parler de la structure des méta-analyses arguments et de la répartition des méta-analyses au sein de leurs
            listes.

            Parler de la différence entre l'argument output qui sert à spécifier un output totalement différent tandis
            que si on le met pas ça veut dire que le type de méta-analyse prend un output parmi les arguments des analyses.

            Parameters
            ----------
            macular_analysis_dataframes : MacularAnalysisDataframes
                Macular analysis dataframes that the user want to do one meta-analysis. This dataframe will be used to
                extract the analyses required for the meta-analysis.

            meta_analysis_type : str
                Name of the current meta-analysis type.

            Returns
            ----------
            modified_meta_analysis_function : function
                Decorated meta-analysis function to apply to calculate the current meta-analysis.
            """
            # Decondensed information on each analysis contained in the dictionaries of common meta-analysis groups.
            common_meta_analysis_group_dictionaries = MacularAnalysisDataframes.multiple_common_meta_analysis_group_parser(
                macular_analysis_dataframes.multiple_dicts_analysis["MetaAnalysis"][meta_analysis_type])

            # Loop through each dictionary of common meta-analysis groups to execute them one by one.
            for dictionary in common_meta_analysis_group_dictionaries:
                macular_analysis_dataframes.make_common_group_meta_analysis(meta_analysis_function, dictionary,
                                                                            meta_analysis_type, dict_index)

        return modified_meta_analysis_function

    def setup_index_dictionary(self, multi_macular_dict_array):
        """Function creating a dictionary containing all indexes describing a macular analysis dataframe and its
        associated multiple macular dict array.

        The dictionary is divided into sub-dictionaries for each condition. The keys of these sub-dictionaries are
        the names of the indexes in the multiple macular dict array (‘temporal’, “spatial_x”, etc.). The index
        dictionary also contains a special key called ‘overall’, which is composed of keys corresponding to each of
        the conditions present in the macular analysis dataframes. These keys are used to contain the indexes for each
        of these conditions.

        Example :
        {"barSpeed3dps": {"temporal": np.array(), "spatial_x": np.array(), "spatial_y": np.array()},
        "barSpeed6dps": {"temporal": np.array(), "spatial_x": np.array(), "spatial_y": np.array()},
        "barSpeed30dps": {"temporal": np.array(), "spatial_x": np.array(), "spatial_y": np.array()},
        "overall": {"barSpeed": np.array(), "wAmaGang": np.array(), "hBip": np.array()}        }

        Parameters
        ----------
        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with different MacularDictArray.

        Returns
        ----------
        dict_index : dict of dict
            Dictionary containing all the indexes from a multiple macular dict array.
        """
        # Generation of dictionary keys from the multiple macular dict array.
        dict_index = {condition: multi_macular_dict_array[condition].index for condition in multi_macular_dict_array}

        # Generation of index keys for each condition in the MacularAnalysisDataframes.
        dict_index["overall"] = {}
        for condition in self.dict_analysis_dataframes["Conditions"].index.values:
            dict_index["overall"][condition.split(" ")[0]] = self.dict_analysis_dataframes["Conditions"].loc[
                condition].values.astype(float)

        return dict_index

    @staticmethod
    def multiple_common_meta_analysis_group_parser(meta_analysis_dictionaries):
        """Function that transforms a list of condensed common meta-analysis dictionaries to detail all the
        meta-analyses that comprise it.

        The process works by decondensing one by one all the condensed common analysis dictionaries associated with
        each meta-analysis argument.

        Parameters
        ----------
        meta_analysis_dictionaries : list of dict
            List of multiple different dictionaries of common meta-analysis group where keys are meta-analysis function
            arguments and values are condensed common analysis group dictionaries.

        Returns
        ----------
        parsed_meta_analysis_dictionaries : list of dict
            List of a common meta-analysis group dictionaries with detailed meta-analyses for each meta-analysis
            arguments.
        """
        parsed_meta_analysis_dictionaries = []
        for common_group_meta_analysis_dictionary in meta_analysis_dictionaries:
            parsed_meta_analysis_dictionaries += [MacularAnalysisDataframes.common_meta_analysis_group_parser(
                common_group_meta_analysis_dictionary)]

        return parsed_meta_analysis_dictionaries

    @staticmethod
    def common_meta_analysis_group_parser(common_meta_analysis_group_dictionary):
        """Function that decondenses all meta-analyses contained in a dictionary of common meta-analysis groups. This is
        characterised by the presence of a list of the levels defining each analyses containing in meta-analyses
        arguments: its dimension, condition, measure, type of analysis and any associated flag.

        To perform this process, dictionaries of common analysis groups must not contain duplicate names among their
        levels. Parsing will replace each of meta-analysis arguments dictionaries with the list of tuples of level names
        that define the different analyses to be extracted from the MacularAnalysisDataframes. The dictionary also
        contains a ‘params’ key associated with external parameters to be used for meta-analysis. This dictionary is not
        modified during parsing.

        In the case where the lists of analysis level for each argument are of different sizes, en error will be raised
        except a list of size 1, which will adjust to the maximum size observed.

        Example :
        common_meta_analysis_group_parser(
        {"numerator": {"dimensions": "X:Y", "conditions": "barSpeed27ps:barSpeed30ps", "measurements": "VSDI",
                        "analyses": "peak_amplitude", "flag": "internal_flag"},
        "denominator": {"dimensions": "Conditions", "conditions": "barSpeed27ps:barSpeed30ps", "measurements": "VSDI",
                        "analyses": "peak_amplitude", "flag": "internal_flag"},
        "output": {"dimensions": "X:Y", "conditions": "barSpeed27ps:barSpeed30ps", "measurements": "VSDI",
                        "analyses": "peak_amplitude"},
        "params": {"factor": 8})

        > {"numerator": [("X", "barSpeed27ps", "VSDI", "peak_amplitude", "internal_flag"),
                        ("X", "barSpeed27ps", "VSDI", "peak_amplitude", "internal_flag")
                        ("Y", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag"),
                        ("Y", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag")],
        "denominator": [("Conditions", "barSpeed27ps", "VSDI", "peak_amplitude", "internal_flag"),
                        ("Conditions", "barSpeed27ps", "VSDI", "peak_amplitude", "internal_flag"),
                        ("Conditions", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag"),
                        ("Conditions", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag")],
        "output": [("X", "barSpeed27ps", "VSDI", "peak_amplitudes_division"),
                   ("X", "barSpeed27ps", "VSDI", "peak_amplitudes_division"),
                   ("Y", "barSpeed30dps", "VSDI", "peak_amplitudes_division"),
                   ("Y", "barSpeed30dps", "VSDI", "peak_amplitudes_division")],
        "params": {"factor": 8}}

        Parameters
        ----------
        common_meta_analysis_group_dictionary : dict of dict
            Dictionary of a common meta-analysis group where keys are meta-analysis function arguments and values are
            common analysis group dictionaries.

            The dictionary of the common meta-analysis group consists of keys corresponding to the arguments to be
            passed to the meta-analysis function. Each key is associated with the dictionary of the common analysis
            group whose argument will take the values.

        Returns
        ----------
        parsed_dictionary : dict of list
            Dictionary of a common meta-analysis group with lists of tuples of level names characterising each analysis
            associated with each of the meta-analysis arguments.
        """
        # Verification that there are no duplicates in the dictionaries of common analysis groups.
        MacularAnalysisDataframes.check_common_analysis_group_repeats(common_meta_analysis_group_dictionary)

        # Initialisation of the dictionary containing the lists of levels tuples of the common meta-analysis group.
        parsed_dictionary = {}

        # Initialisation of the maximum length of the level lists for common analysis groups.
        common_meta_analysis_group_max_length = 1
        # Loop through all argument names for the meta-analysis function in its dictionary.
        for meta_analysis_argument in common_meta_analysis_group_dictionary:
            # Copy the parameter dictionary to the parsed dictionary of common meta-analysis group.
            if meta_analysis_argument == "params":
                parsed_dictionary["params"] = common_meta_analysis_group_dictionary[meta_analysis_argument].copy()
            # Initialisation of the list of levels of one common analysis group for one meta-analysis argument.
            else:
                argument_common_group_analysis = common_meta_analysis_group_dictionary[meta_analysis_argument]
                if "output" in meta_analysis_argument:
                    # Creation of levels generator for the common analysis group of the current output argument.
                    argument_common_analysis_group_generator = MacularAnalysisDataframes.common_analysis_group_parser(
                        [argument_common_group_analysis["dimensions"], argument_common_group_analysis["conditions"],
                         argument_common_group_analysis["measurements"], argument_common_group_analysis["analyses"]])
                else:
                    # Creation of levels generator for the common analysis group of the current not output argument.
                    argument_common_analysis_group_generator = MacularAnalysisDataframes.common_analysis_group_parser(
                        [argument_common_group_analysis["dimensions"], argument_common_group_analysis["conditions"],
                         argument_common_group_analysis["measurements"], argument_common_group_analysis["analyses"],
                         argument_common_group_analysis["flag"]])
                # Transformation of levels generator into a list of levels of the common analysis group.
                parsed_dictionary[meta_analysis_argument] = [analysis_levels for analysis_levels in
                                                             argument_common_analysis_group_generator]
                # Calculate the maximum length of the level lists for each argument.
                if len(parsed_dictionary[meta_analysis_argument]) > common_meta_analysis_group_max_length:
                    common_meta_analysis_group_max_length = len(parsed_dictionary[meta_analysis_argument])

        # Adjusting the length of levels lists of common analysis group that were too small.
        for meta_analysis_argument in parsed_dictionary:
            if meta_analysis_argument != "params":
                parsed_dictionary[meta_analysis_argument] = (MacularAnalysisDataframes.
                check_common_analysis_group_levels_size(
                    parsed_dictionary[meta_analysis_argument],
                    common_meta_analysis_group_max_length))

        return parsed_dictionary

    @staticmethod
    def check_common_analysis_group_repeats(common_meta_analysis_group_dictionary):
        """Function that checks that no level is repeated multiple times within the keys of a common analysis group
        dictionary (dimension, conditions, measures, analyses).

        Parameters
        ----------
        common_meta_analysis_group_dictionary : dict of dict
            Dictionary of a common meta-analysis group where keys are meta-analysis function arguments and values are
            common analysis group dictionaries.

        Raises
        ----------
        KeyError
            A level of the common analysis group is repeated at least once.
        """
        # Loop through the arguments of a common analysis group dictionary except ‘params’.
        for argument in common_meta_analysis_group_dictionary:
            if argument != "params":
                # Loop through the levels of a common analysis group dictionary except ‘flag’.
                for element in common_meta_analysis_group_dictionary[argument]:
                    if element != "flag":
                        # Verify that no duplicate level names are present.
                        if (len(common_meta_analysis_group_dictionary[argument][element].split(":")) !=
                                len(set(common_meta_analysis_group_dictionary[argument][element].split(":")))):
                            raise KeyError(f"A {element} is repeated several times in the argument {argument} of the "
                                           f"common analysis group dictionary : "
                                           f"{common_meta_analysis_group_dictionary[argument][element]}")

    @staticmethod
    def check_common_analysis_group_levels_size(common_analysis_group_levels_list, expected_length):
        """Function for checking the size of the list of levels for a group of common analyses so that it corresponds
        to an expected length.

        If the length does not match, there are two possibilities. If the list is of size 1, it will be repeated as many
        times as the expected length. However, if the size is greater than 1, an error will be raised.

        Example : check_common_analysis_group_levels_size([(element1, element2, element3)], 2)
        > [(element1, element2, element3), (element1, element2, element3)]

        Parameters
        ----------
        common_analysis_group_levels_list : list of tuples
            List containing tuples of level names from a common analysis group.

        expected_length : int
            Expected length that you want to reach with the list of levels in the common analysis group.

        Returns
        ----------
        checked_levels_list : list of tuples
            List level tuples of a common analysis group checked in size.

        Raises
        ----------
        ValueError
            The length of the level list is smaller than expected and is also greater than 1.
        """
        levels_list_length = len(common_analysis_group_levels_list)
        # Cases where the length of the level list is smaller than expected.
        if levels_list_length < expected_length and levels_list_length == 1:
            checked_levels_list = []
            # Correct the length by repeating each element in the list in an equivalent way.
            for levels in common_analysis_group_levels_list:
                checked_levels_list += [levels] * (expected_length // levels_list_length)
        # Case where the length of the level list is equal to the expected size.
        elif levels_list_length == expected_length:
            checked_levels_list = common_analysis_group_levels_list
        else:
            raise ValueError(f"The length of the common analysis group level list does not match the maximum length"
                             f" {expected_length}")

        return checked_levels_list

    def make_common_group_meta_analysis(self, meta_analysis_function, common_meta_analysis_group_dictionary,
                                        meta_analysis, dict_index):
        """Function performing all meta-analyses present in a group of decondensed common meta-analyses.

        The decondensed group of common meta-analyses is structured as a dictionary associating the names of the
        meta-analysis arguments with lists of tuples of all analyses levels for which the argument will take the value.

        A meta-analysis groups together all the analyses located at the same index in all the lists of arguments of
        meta-analyses. Therefore, all the arguments in a dictionary of common meta-analyses have a list of the same
        length, each element of which represents the sequence of values that the argument will take during the common
        meta-analysis.

        The function iterates over the arguments (except ‘params’) of all meta-analyses defined in the common
        meta-analysis group. The level tuples associated with all these arguments are stored as is in a first dictionary
        that is used after the loop to construct the names of all output arguments of the meta-analysis. The level
        tuples of the non-output arguments are also used to extract the arrays of analyses they describe into a second
        dictionary. This dictionary also contains the analysis levels defined in the output arguments. This dictionary
        is finally used in the execution of the current meta-analysis function.

        Parameters
        ----------
        meta_analysis_function : function
            Analysis function to apply to calculate the current meta-analysis.

            This meta-analysis function changes between each meta-analysis, so they must all have the same three input
            arguments: the data, the index and the analysis parameters. However, the data may be in different forms
            depending on the function of the meta-analysis.

        common_meta_analysis_group_dictionary : dict of dict
            Dictionary of a decondensed common meta-analysis group where keys are meta-analysis function arguments and
            values are decondensed common analysis group dictionaries.

            The dictionary of the common meta-analysis group consists of keys corresponding to the arguments to be
            passed to the meta-analysis function. Each key is associated with the list of each analysis levels whose
            argument will take the values.

        meta_analysis : str
            Name of the current meta-analysis type.

        dict_index : dict of dict
            Dictionary of all indexes present in the multiple macular dict array used in the current
            MacularAnalysisDataframes.
        """
        # Initiate the dictionary containing all the information of one meta-analysis.
        current_meta_analysis_dictionary = {}

        # Define a list of all argument names needed for the current meta-analysis.
        meta_analysis_arguments_list = list(common_meta_analysis_group_dictionary.keys())
        meta_analysis_arguments_list.remove("params")

        # Measurement of the number of meta-analyses to be performed in the current common meta-analysis group.
        meta_analysis_arguments_length = len(common_meta_analysis_group_dictionary[meta_analysis_arguments_list[0]])

        # Loop on the indexes of all meta-analyses contained in the common meta-analysis group.
        for analysis_levels_index in range(meta_analysis_arguments_length):
            # Loop over the arguments of the current meta-analysis function.
            for meta_analysis_argument in meta_analysis_arguments_list:
                # Store the levels defining the current meta-analysis in the meta-analysis dictionary.
                current_meta_analysis_dictionary[meta_analysis_argument] = (
                    common_meta_analysis_group_dictionary)[meta_analysis_argument][analysis_levels_index]

            # Creation of a dictionary of names for each output argument of the current meta-analysis.
            current_meta_analysis_dictionary.update(MacularAnalysisDataframes.make_meta_analysis_outputs(
                meta_analysis, current_meta_analysis_dictionary, common_meta_analysis_group_dictionary["params"]))

            # Execution of the current meta-analysis function.
            meta_analysis_function(self, current_meta_analysis_dictionary, dict_index,
                                   common_meta_analysis_group_dictionary["params"].copy())

    @staticmethod
    def extract_all_analysis_array_from_dataframes(macular_analysis_dataframes, meta_analysis_dictionary):
        """Function used to extract the value(s) associated with all analysis involved in the meta-analysis and
        contained in a MacularAnalysisDataframes.

        Parameters
        ----------
        macular_analysis_dataframes : MacularAnalysisDataframes
            Macular Analyses Dataframes that the user wishes to use to extract a row from a given dataframe.

        meta_analysis_dictionary : dict of tuple
            Meta-analysis dictionary linking the names of arguments in a meta-analysis with the names of the levels
            defining a given analysis (dimension, condition, measurements, analysis type, flag).
        """
        # Loop on meta-analysis arguments except output ones.
        for meta_analysis_argument in meta_analysis_dictionary.keys():
            if "output" not in meta_analysis_argument:
                meta_analysis_dictionary[meta_analysis_argument] = (
                    MacularAnalysisDataframes.extract_one_analysis_array_from_dataframes(
                        macular_analysis_dataframes, meta_analysis_dictionary[meta_analysis_argument]))

    @staticmethod
    def extract_one_analysis_array_from_dataframes(macular_analysis_dataframes, analysis_levels):
        """Function used to extract the value(s) associated with a given analysis and contained in a
        MacularAnalysisDataframes.

        Each analysis of a MacularAnalysisDataframes can be defined by the four hierarchical levels of a
        MacularAnalysisDataframes. There is the dimension (‘X’, ‘Y’, “Conditions”), the condition (‘barSpeed30dps’,
        ‘ampGang30Hz’), the measurement (‘VSDI’, ‘FiringRate_GanglionGainControl’) and the type of analysis (“latency”,
        ‘peak_amplitude’). These four levels must therefore be used to locate and extract an analysis. In addition,
        there is also the name of the flag, if there is one.

        In some cases, an analysis to be extracted may be defined only by the dimension, condition and type of the
        analysis, but not by the measurement. This can happen for analyses from meta-analyses. In this case, the
        measurement value is set to an empty character string ‘’.

        In the case of the ‘Conditions’ dimension, there is a single dataframe containing the data, whereas in the
        other spatio-temporal dimensions there is one dataframe per condition. Therefore, two methods must be used to
        extract values from these two types of dataframes.

        Parameters
        ----------
        macular_analysis_dataframes : MacularAnalysisDataframes
            Macular Analyses Dataframes that the user wishes to use to extract a row from a given dataframe.

        analysis_levels : tuple
            Names of the levels defining a given analysis (dimension, condition, measure, analysis type).

        Returns
        ----------
        analysis_array : int, float or np.ndarray
            Array of values or single value of the analysis to be extracted.
        """
        # Construction of the name of the analysis line to be extracted.
        if analysis_levels[2] == "":
            # Cases that only include the analysis type in the name of the analysis to be extracted.
            dataframe_row = f"{analysis_levels[3]}_{analysis_levels[4]}".strip("_")
        else:
            # Cases that include the analysis type and measurement in the name of the analysis to be extracted.
            dataframe_row = f"{analysis_levels[3]}_{analysis_levels[2]}_{analysis_levels[4]}".strip("_")

        # Cases of conditions dataframe.
        if analysis_levels[0] == "Conditions":
            analysis_array = macular_analysis_dataframes.dict_analysis_dataframes[analysis_levels[0]].loc[
                dataframe_row, analysis_levels[1]]
        # Case of multiple spatio-temporal dataframes.
        else:
            analysis_array = macular_analysis_dataframes.dict_analysis_dataframes[analysis_levels[0]][
                                 analysis_levels[1]].loc[
                             dataframe_row, :]

        return analysis_array

    @staticmethod
    def make_meta_analysis_outputs(meta_analysis_name, meta_analysis_dictionary, parameters_meta_analysis_dict):
        """Function for formatting meta-analysis outputs names.

        The name of the meta-analysis is primarily retrieved from the name defined among the arguments of the
        meta-analysis function. All arguments containing the term ‘output’ will be used to retrieve as many names as
        will be defined in a dictionary. If this is not the case, the name defined in the meta-analysis parameter
        dictionary will be used. All parameters containing the term ‘output’ are retrieved again. Finally, the default
        behaviour if no output exists is to format using the measurements and analysis types of each argument of the
        meta-analysis. It is possible to slightly adapt this default case by adding a ‘flag’ parameter in the
        meta-analysis parameter dictionary. This ‘flag’ character string will be added as last suffix.

        Parameters
        ----------
        meta_analysis_name : str
            Name of the meta-analysis for which a format is needed.

        meta_analysis_dictionary : dict of tuples
            Meta-analysis dictionary linking the names of arguments in a meta-analysis with the names of the levels
            defining a given analysis (dimension, condition, measurements, analysis type, flag).

            Among the arguments of the meta-analysis function, there may be one or more arguments used as output within
            a MacularAnalysisDataframes. These arguments can be recognised by the presence of the term ‘output’ in
            their key.

        parameters_meta_analysis_dict : dict
            Dictionary containing all the parameters of the meta-analysis to be formatted.

            This dictionary can contain a ‘flag’ key associated with the final suffix to be added at the end of the
            name of the dataframe row to be created. The dictionary may also contain ‘output’ keys corresponding to the
            names of the rows to be used as output if no “output” key exists in the names of the arguments of the
            meta-analysis function. These keys are characterised by the presence of the term ‘output’ in their name.

        Returns
        ----------
        meta_analysis_outputs_dict : dict
            Dictionary associating each output with the output name(s) of the current meta-analysis to use in a
            dataframe.
        """
        meta_analysis_outputs_dict = {}

        output = False
        # Loop over all arguments of the meta-analysis function.
        for meta_analysis_arguments in meta_analysis_dictionary:
            # Cases where one or more outputs have been defined in the meta-analysis arguments.
            if "output" in meta_analysis_arguments:
                meta_analysis_outputs_dict[meta_analysis_arguments] = {
                    "dimension": meta_analysis_dictionary[meta_analysis_arguments][0],
                    "condition": meta_analysis_dictionary[meta_analysis_arguments][1],
                    "name": f'{meta_analysis_dictionary[meta_analysis_arguments][3]}'
                }
                output = True

        # Cases where no output has been defined in the meta-analysis arguments.
        if not output:
            # Cases where one or more outputs have been defined in the meta-analysis settings.
            for params in parameters_meta_analysis_dict:
                if "output" in params:
                    meta_analysis_outputs_dict[params] = {"name": f'{parameters_meta_analysis_dict[params]}'}
                    output = True

        # Default case performing formatting based on the analysis information.
        if not output:
            dataframe_row_list = []

            for meta_analysis_argument in sorted(list(meta_analysis_dictionary.keys())):
                dataframe_row_list += [meta_analysis_dictionary[meta_analysis_argument][3],
                                       meta_analysis_dictionary[meta_analysis_argument][2]]
            dataframe_row_list += [meta_analysis_name, parameters_meta_analysis_dict["flag"]]
            meta_analysis_outputs_dict["output"] = {"name": "_".join(dataframe_row_list).strip("_")}

        return meta_analysis_outputs_dict

    @staticmethod
    def add_array_line_to_dataframes(macular_analysis_dataframes, dimension, condition, output, array_output):
        """Function to add a new line within a dataframe of a MacularDictDataframes.

        Creating a new line requires all the names of the levels in the MacularDictDataframes to identify the position
        of the new line and its name.

        Parameters
        ----------
        macular_analysis_dataframes : MacularAnalysisDataframes
            Macular Analyses Dataframes that the user wishes to use to add a new row in a given dataframe.

        dimension : str
            Dimension of the dataframe in which to add the new row.

        condition : str
            Condition under which to add the new line.

        output : str
            Name of the new line to be created.

        array_output : int, float or np.ndarray
            Value of the new line to be created.
        """
        # Case of the conditions dataframe.
        if dimension == "Conditions":
            # Case of adding an array of values for all conditions in the conditions Dataframe.
            if isinstance(array_output, np.ndarray):
                macular_analysis_dataframes.dict_analysis_dataframes[dimension].loc[output, :] = array_output
            # Case of adding a single value of a given condition to the condition dataframe.
            else:
                macular_analysis_dataframes.dict_analysis_dataframes[dimension].loc[output, condition] = array_output
        elif dimension == "MetaConditions":
            macular_analysis_dataframes.dict_analysis_dataframes[dimension].loc[output, condition] = array_output
        # Case of spatio-temporal dataframes.
        else:
            macular_analysis_dataframes.dict_analysis_dataframes[dimension][condition].loc[output] = array_output

    @staticmethod
    @meta_analysis
    def normalization_analyzing(macular_analysis_dataframes, meta_analysis_dictionary, index,
                                parameters_meta_analysis_dict):
        """Function that calculates a normalization between two given analyses and multiplies the result by a
        multiplication factor.

        To work, this meta-analysis requires two arguments: the numerator and the denominator, which must defined in
        the meta-analysis dictionaries. A third key, ‘output’, must also be defined, which corresponds to the position
        where you want to save the result of the operation.

        Finally, the last key, “params”, must contain the key ‘factor’ with the value of the multiplication factor and
        the flag to be used as a suffix for the name of the meta-analysis.

        Parameters
        ----------
        macular_analysis_dataframes : MacularAnalysisDataframes
            Macular Analyses Dataframes whose analyses the user wishes to use for meta-analysis.

        meta_analysis_dictionary : dict of tuple and dict of dict
            Meta-analysis dictionary linking the names of arguments in a meta-analysis with the associated array of
            values. In the case of arguments containing the term ‘output’, the key is associated with the name of the
            outputs created for the dataframe.

        index : dict of dict
            Dictionary of all indexes present in the multiple macular dict array used in the current
            MacularAnalysisDataframes.

        parameters_meta_analysis_dict : dict
            Dictionary containing all the parameters of the meta-analysis to be formatted.

            This dictionary must contain the ‘factor’ key associated with the value you want to use as the
            multiplication factor.
        """
        # Convert all non-outputs meta-analysis arguments levels into the corresponding analysis array.
        MacularAnalysisDataframes.extract_all_analysis_array_from_dataframes(macular_analysis_dataframes,
                                                                             meta_analysis_dictionary)

        # Calculation of the division of the two analysis values and multiplication by the factor.
        normalized_values = MetaAnalyser.normalization_computing(meta_analysis_dictionary["numerator"],
                                                                 meta_analysis_dictionary["denominator"],
                                                                 parameters_meta_analysis_dict["factor"])

        # Adds the output value(s) to a new row in the output dataframe.
        MacularAnalysisDataframes.add_array_line_to_dataframes(macular_analysis_dataframes,
                                                               meta_analysis_dictionary["output"]["dimension"],
                                                               meta_analysis_dictionary["output"]["condition"],
                                                               meta_analysis_dictionary["output"]["name"],
                                                               normalized_values)

    @staticmethod
    @meta_analysis
    def peak_speed_analyzing(macular_analysis_dataframes, meta_analysis_dictionary, index,
                             parameters_meta_analysis_dict):
        """Function that calculates the speed of movement of an object based on the movement of its peak in one of the
        spatial dimensions.

        To work, this meta-analysis requires 1 argument: the time to peak that needs to be fitted. It must be defined in
        the meta-analysis dictionaries. This dictionary does not require an ‘output’ key to define the output to which
        the peak speed should be sent. Instead, the output is automatically set to the conditions dataframe and directly
        uses the conditions defined in the ‘time to peak’ analysis.

        The dictionary also contains the ‘params’ parameters, whose dictionary must contain the ‘index’ parameter,
        which corresponds to the name of the spatial index to be used (X or Y). The spatial index depends on the axis
        of movement. Another key must be added to define the name of the output to be created in the condition
        dataframe. The first key, ‘output’, allows you to define a specific name, while the second alternative key,
        ‘flag’, allows you to use the default output name by simply adding a suffix.

        Parameters
        ----------
        macular_analysis_dataframes : MacularAnalysisDataframes
            Macular Analyses Dataframes whose analyses the user wishes to use for meta-analysis.

        meta_analysis_dictionary : dict of tuple and dict of dict
            Meta-analysis dictionary linking the names of arguments in a meta-analysis with the associated array of
            values. In the case of arguments containing the term ‘output’, the key is associated with the name of the
            outputs created for the dataframe.

        index : dict of dict
            Dictionary of all indexes present in the multiple macular dict array used in the current
            MacularAnalysisDataframes.

        parameters_meta_analysis_dict : dict
            Dictionary containing all the parameters of the meta-analysis to be formatted.

            This dictionary must contain the spatial index name to be used for fitting.
        """
        # Store dimensions and conditions of output.
        meta_analysis_dictionary["output"]["dimension"] = "Conditions"
        meta_analysis_dictionary["output"]["condition"] = meta_analysis_dictionary["time_to_peak"][1]

        # Convert all non-outputs meta-analysis arguments levels into the corresponding analysis array.
        MacularAnalysisDataframes.extract_all_analysis_array_from_dataframes(macular_analysis_dataframes,
                                                                             meta_analysis_dictionary)

        # Calculation of the peak speed of the time to peak data array.
        peak_speed_fit = MetaAnalyser.linear_fit_computing(
            meta_analysis_dictionary["time_to_peak"], index[meta_analysis_dictionary["output"]["condition"]][
                parameters_meta_analysis_dict["index"]], 1)

        # Adds the output value(s) to a new row in the output dataframe.
        MacularAnalysisDataframes.add_array_line_to_dataframes(macular_analysis_dataframes,
                                                               meta_analysis_dictionary["output"]["dimension"],
                                                               meta_analysis_dictionary["output"]["condition"],
                                                               meta_analysis_dictionary["output"]["name"],
                                                               peak_speed_fit["slopes"][0])

    @staticmethod
    @meta_analysis
    def stationary_peak_delay_analyzing(macular_analysis_dataframes, meta_analysis_dictionary, index,
                                        parameters_meta_analysis_dict):
        """Function for calculating a stationary value at which the peak delay remains despite variation in the
        spatial coordinate.

        To work, this meta-analysis requires 1 argument: the peak delay that needs to be averaged. It must be defined in
        the meta-analysis dictionaries. This dictionary does not require an ‘output’ key to define the output to which
        the peak speed should be sent. Instead, the output is automatically set to the conditions dataframe and directly
        uses the conditions defined in the ‘time to peak’ analysis.

        The dictionary also contains the ‘params’ parameters, whose dictionary must contain a key to  define the name
        of the output to be created in the condition dataframe. The first key, ‘output’, allows you to define a specific
        name, while the second alternative key, ‘flag’, allows you to use the default output name by simply adding a
        suffix.

        Parameters
        ----------
        macular_analysis_dataframes : MacularAnalysisDataframes
            Macular Analyses Dataframes whose analyses the user wishes to use for meta-analysis.

        meta_analysis_dictionary : dict of tuple and dict of dict
            Meta-analysis dictionary linking the names of arguments in a meta-analysis with the associated array of
            values. In the case of arguments containing the term ‘output’, the key is associated with the name of the
            outputs created for the dataframe.

        index : dict of dict
            Dictionary of all indexes present in the multiple macular dict array used in the current
            MacularAnalysisDataframes.

        parameters_meta_analysis_dict : dict
            Dictionary containing all the parameters of the meta-analysis to be formatted.

            This dictionary don't contain any parameters other than outputs ones.
        """
        # Store dimensions and conditions of output.
        meta_analysis_dictionary["output"]["dimension"] = "Conditions"
        meta_analysis_dictionary["output"]["condition"] = meta_analysis_dictionary["peak_delay"][1]

        # Convert all non-outputs meta-analysis arguments levels into the corresponding analysis array.
        MacularAnalysisDataframes.extract_all_analysis_array_from_dataframes(macular_analysis_dataframes,
                                                                             meta_analysis_dictionary)

        # Calculation of the stationary peak delay
        stationary_peak_delay_value = MetaAnalyser.mean_computing(meta_analysis_dictionary["peak_delay"])

        # Adds the output value(s) to a new row in the output dataframe.
        MacularAnalysisDataframes.add_array_line_to_dataframes(macular_analysis_dataframes,
                                                               meta_analysis_dictionary["output"]["dimension"],
                                                               meta_analysis_dictionary["output"]["condition"],
                                                               meta_analysis_dictionary["output"]["name"],
                                                               stationary_peak_delay_value)

    @staticmethod
    @meta_analysis
    def linear_fit_analyzing(macular_analysis_dataframes, meta_analysis_dictionary, index,
                             parameters_meta_analysis_dict):
        """Function that calculates all properties of a linear fit along one dimension of the dataframe.

        In the case of the ‘conditions’ dimension of the dataframe, the calculated global values can be stored in the
        ‘MetaAnalysis’ dataframe, which summarises all the general meta-analyses calculated on the conditions.

        To work, this meta-analysis requires 1 argument: the "data_to_fit" that needs to be fitted. This dictionary
        requires an ‘output’ key to define an output to which each property of the fit is sent. The ‘output_slopes’ key
        is created if you want to retrieve the slopes of the fits. The “output_inflection_points_data” and
        ‘output_inflection_points_index’ keys are used if you want to retrieve the abscissa or ordinate of the
        inflection points of the fit. The keys ‘data_intercepts’ and ‘index_intercepts’ contain the ordinate and
        abscissa at the origin of the lines corresponding to each of the linear segments of the fit. Finally, the keys
        ‘output_data_prediction’ and ‘output_index_prediction’ are applied to obtain the values of the index and the
        data predicted by the fit.

        The dictionary also contains the ‘params’ parameters, whose dictionary must contain the ‘index’ parameter,
        which corresponds to the name of the index to be used. The second key in the dictionary is ‘n_segments’, which
        indicates the number of linear segments that the fit must analyse. This number will influence the number of
        slopes and inflection points obtained. The last parameter, ‘n_points’, is used to select the resolution of the
        fit. It is important to note that regardless of this resolution, the data predictions and fit indexes within a
        given dataframe will be binning to ensure the correct size.

        Parameters
        ----------
        macular_analysis_dataframes : MacularAnalysisDataframes
            Macular Analyses Dataframes whose analyses the user wishes to use for meta-analysis.

        meta_analysis_dictionary : dict of tuple and dict of dict
            Meta-analysis dictionary linking the names of arguments in a meta-analysis with the associated array of
            values. In the case of arguments containing the term ‘output’, the key is associated with the name of the
            outputs created for the dataframe.

        index : dict of dict
            Dictionary of all indexes present in the multiple macular dict array used in the current
            MacularAnalysisDataframes.

        parameters_meta_analysis_dict : dict
            Dictionary containing all the parameters of the meta-analysis to be formatted.

            This dictionary must contain the index name, the number of segments and the resolution to be used for
            fitting.
        """
        # Store dimensions and conditions of output.
        meta_analysis_dictionary["index"] = {"condition": meta_analysis_dictionary["data_to_fit"][1]}

        # Convert all non-outputs meta-analysis arguments levels into the corresponding analysis array.
        MacularAnalysisDataframes.extract_all_analysis_array_from_dataframes(macular_analysis_dataframes,
                                                                             meta_analysis_dictionary)

        # Getting the index to use for the fitting.
        current_index = index[meta_analysis_dictionary["index"]["condition"]][parameters_meta_analysis_dict["index"]]

        # Fit of the variable to be fitted, respecting the number of segments given in the parameters.
        linear_fit = MetaAnalyser.linear_fit_computing(current_index, meta_analysis_dictionary["data_to_fit"],
                                                       parameters_meta_analysis_dict["n_segments"],
                                                       n_points=parameters_meta_analysis_dict["n_points"])

        # Binning of prediction arrays from data and index arrays to obtain the size of the fitted arrays.
        linear_fit["data_prediction"], linear_fit["index_prediction"] = MetaAnalyser.statistic_binning(
            linear_fit["data_prediction"], linear_fit["index_prediction"], current_index.shape[0])

        # Adds the output slopes value(s) to a new row in the output dataframe.
        if "output_slopes" in meta_analysis_dictionary.keys():
            for slope, output_name in zip(linear_fit["slopes"], meta_analysis_dictionary["output_slopes"]["name"]):
                print("slope", slope)
                MacularAnalysisDataframes.add_array_line_to_dataframes(
                    macular_analysis_dataframes, meta_analysis_dictionary["output_slopes"]["dimension"],
                    meta_analysis_dictionary["output_slopes"]["condition"], output_name, slope)

        # Adds the output inflection points data value(s) to a new row in the output dataframe.
        if "output_inflection_points_data" in meta_analysis_dictionary.keys():
            for inflection_points_data, output_name in zip(linear_fit["inflection_points_data"],
                                                           meta_analysis_dictionary["output_inflection_points_data"][
                                                               "name"]):
                MacularAnalysisDataframes.add_array_line_to_dataframes(
                    macular_analysis_dataframes, meta_analysis_dictionary["output_inflection_points_data"]["dimension"],
                    meta_analysis_dictionary["output_inflection_points_data"]["condition"], output_name,
                    inflection_points_data)

        # Adds the output inflection point index value(s) to a new row in the output dataframe.
        if "output_inflection_points_index" in meta_analysis_dictionary.keys():
            for inflection_points_index, output_name in zip(linear_fit["inflection_points_index"],
                                                            meta_analysis_dictionary["output_inflection_points_index"][
                                                                "name"]):
                MacularAnalysisDataframes.add_array_line_to_dataframes(
                    macular_analysis_dataframes,
                    meta_analysis_dictionary["output_inflection_points_index"]["dimension"],
                    meta_analysis_dictionary["output_inflection_points_index"]["condition"], output_name,
                    inflection_points_index)

        # Adds the output index prediction value(s) to a new row in the output dataframe.
        if "output_index_prediction" in meta_analysis_dictionary.keys():
            print("fit", linear_fit["index_prediction"])
            MacularAnalysisDataframes.add_array_line_to_dataframes(
                macular_analysis_dataframes, meta_analysis_dictionary["output_index_prediction"]["dimension"],
                meta_analysis_dictionary["output_index_prediction"]["condition"],
                meta_analysis_dictionary["output_index_prediction"]["name"], linear_fit["index_prediction"])

        # Adds the output data prediction value(s) to a new row in the output dataframe.
        if "output_data_prediction" in meta_analysis_dictionary.keys():
            MacularAnalysisDataframes.add_array_line_to_dataframes(
                macular_analysis_dataframes, meta_analysis_dictionary["output_data_prediction"]["dimension"],
                meta_analysis_dictionary["output_data_prediction"]["condition"],
                meta_analysis_dictionary["output_data_prediction"]["name"], linear_fit["data_prediction"])

        # Adds the output data intercept to a new row in the output dataframe.
        if "output_data_intercepts" in meta_analysis_dictionary.keys():
            for intercept, output_name in zip(linear_fit["data_intercepts"], meta_analysis_dictionary[
                "output_data_intercepts"]["name"]):
                MacularAnalysisDataframes.add_array_line_to_dataframes(
                    macular_analysis_dataframes, meta_analysis_dictionary["output_data_intercepts"]["dimension"],
                    meta_analysis_dictionary["output_data_intercepts"]["condition"], output_name, intercept)

        # Adds the output index intercept to a new row in the output dataframe.
        if "output_index_intercepts" in meta_analysis_dictionary.keys():
            for intercept, output_name in zip(linear_fit["index_intercepts"], meta_analysis_dictionary[
                "output_index_intercepts"]["name"]):
                MacularAnalysisDataframes.add_array_line_to_dataframes(
                    macular_analysis_dataframes, meta_analysis_dictionary["output_index_intercepts"]["dimension"],
                    meta_analysis_dictionary["output_index_intercepts"]["condition"], output_name, intercept)

    @staticmethod
    @meta_analysis
    def anticipation_fit_analyzing(macular_analysis_dataframes, meta_analysis_dictionary, index,
                                   parameters_meta_analysis_dict):
        """Function that calculates all the anticipation fit indicators generated by a moving object.

        The anticipation is calculated based on the activation time of the cortical columns on the trajectory of the
        movement. Therefore, only the dimension corresponding to the axis of the object's movement is taken into
        account here. The fit achieved is that of the graph of the distance of the cell from the origin of the movement
        as a function of the activation time.

        The process is based on a two-segment fit, which calculates and extracts the inflection point and the two slopes
        for each simulation condition and adds them to the condition dataframe. For both slopes, the speed of the object
        is subtracted to retain only the speed caused by anticipation. At the same time, it also adds the predictions
        of the data and the index by the fit within the dataframe of the dimension of the axis of movement.

        To work, this meta-analysis requires 1 argument: the "activation_time" that needs to be fitted. This dictionary
        does not require an ‘output’ key to define the output to which the peak speed should be sent. Instead, the
        output is automatically set to the conditions dataframe and directly uses the conditions defined in the
        ‘activation_time’ analysis.

        The dictionary also contains the ‘params’ parameters, whose dictionary must contain the ‘index’ parameter,
        which corresponds to the name of the spatial index to be used (X or Y). The spatial index depends on the axis
        of movement. The second key in the dictionary is ‘n_segments’, which indicates the number of linear segments
        that the fit must analyse. This number will influence the number of slopes and inflection points obtained. The
        last parameter, ‘n_points’, is used to select the resolution of the fit. It is important to note that regardless
        of this resolution, the data predictions and fit indexes within a given dataframe will be binning to ensure the
        correct size.

        Another key must be added to define the name of the output to be created in the condition dataframe. The first
        key, ‘output’, allows you to define a specific name, while the second alternative key, ‘flag’, allows you to use
        the default output name by simply adding a suffix.

        Parameters
        ----------
        macular_analysis_dataframes : MacularAnalysisDataframes
            Macular Analyses Dataframes whose analyses the user wishes to use for meta-analysis.

        meta_analysis_dictionary : dict of tuple and dict of dict
            Meta-analysis dictionary linking the names of arguments in a meta-analysis with the associated array of
            values. In the case of arguments containing the term ‘output’, the key is associated with the name of the
            outputs created for the dataframe.

        index : dict of dict
            Dictionary of all indexes present in the multiple macular dict array used in the current
            MacularAnalysisDataframes.

        parameters_meta_analysis_dict : dict
            Dictionary containing all the parameters of the meta-analysis to be formatted.

            This dictionary must contain the spatial index name, the number of segments and the resolution to be used
            for fitting.
        """
        # Store dimensions and conditions of output.
        meta_analysis_dictionary["output"] = {"dimension": "Conditions",
                                              "condition": meta_analysis_dictionary["activation_time"][1]}
        meta_analysis_dictionary["output_prediction"] = {"dimension": meta_analysis_dictionary["activation_time"][0],
                                                         "condition": meta_analysis_dictionary["activation_time"][1]}

        # Convert all non-outputs meta-analysis arguments levels into the corresponding analysis array.
        MacularAnalysisDataframes.extract_all_analysis_array_from_dataframes(macular_analysis_dataframes,
                                                                             meta_analysis_dictionary)

        # Determining the speed of the moving object causing anticipation.
        if "speed" in parameters_meta_analysis_dict:
            speed = parameters_meta_analysis_dict["speed"]
        else:
            speed = float(macular_analysis_dataframes.condition_reg.findall(
                meta_analysis_dictionary["output"]["condition"])[0][1].replace(",", ".")) / 1000

        # Getting the index to use for the fitting.
        current_index = index[meta_analysis_dictionary["output"]["condition"]][parameters_meta_analysis_dict["index"]]

        # Fit of the activation time, respecting the number of segments given in the parameters.
        linear_fit = MetaAnalyser.linear_fit_computing(meta_analysis_dictionary["activation_time"], current_index,
                                                       parameters_meta_analysis_dict["n_segments"],
                                                       n_points=parameters_meta_analysis_dict["n_points"])

        # Binning of prediction arrays from data and index arrays to obtain the size of the fitted arrays.
        linear_fit["data_prediction"], linear_fit["index_prediction"] = MetaAnalyser.statistic_binning(
            linear_fit["index_prediction"], linear_fit["data_prediction"], current_index.shape[0])

        # Adds the slope output value(s) to a new row in the output dataframe.
        for slope, output_name in zip(linear_fit["slopes"], meta_analysis_dictionary["output_slopes"]["name"]):
            MacularAnalysisDataframes.add_array_line_to_dataframes(
                macular_analysis_dataframes, meta_analysis_dictionary["output"]["dimension"],
                meta_analysis_dictionary["output"]["condition"], output_name, slope - speed)

        # Adds the anticipation range output value(s) to a new row in the output dataframe.
        MacularAnalysisDataframes.add_array_line_to_dataframes(
            macular_analysis_dataframes, meta_analysis_dictionary["output"]["dimension"],
            meta_analysis_dictionary["output"]["condition"],
            meta_analysis_dictionary["output_anticipation_range"]["name"], linear_fit["inflection_points_data"][0])

        # Adds the output index prediction value(s) to a new row in the output dataframe.
        MacularAnalysisDataframes.add_array_line_to_dataframes(
            macular_analysis_dataframes, meta_analysis_dictionary["output_prediction"]["dimension"],
            meta_analysis_dictionary["output_prediction"]["condition"],
            meta_analysis_dictionary["output_index_prediction"]["name"], linear_fit["index_prediction"])

        # Adds the output data prediction value(s) to a new row in the output dataframe.
        MacularAnalysisDataframes.add_array_line_to_dataframes(
            macular_analysis_dataframes, meta_analysis_dictionary["output_prediction"]["dimension"],
            meta_analysis_dictionary["output_prediction"]["condition"],
            meta_analysis_dictionary["output_data_prediction"]["name"], linear_fit["data_prediction"])

    @staticmethod
    @meta_analysis
    def maximal_latency_analyzing(macular_analysis_dataframes, meta_analysis_dictionary, index,
                                  parameters_meta_analysis_dict):
        """Function to calculate the maximal latency at which latency begins to saturate for a distance between cell and
        object motion origin that exceeds the anticipation range.

        The calculation is performed by obtaining the latency values and then extracting only the stationary portion.
        This stationary portion is located beyond the anticipation range value. To use this meta-analysis, you must
        first perform an ‘anticipation_fit_analyzing’ meta-analysis.

        To work, this meta-analysis requires 2 arguments: the "latency" that needs to be averaged and
        ‘anticipation_range’ which should be used to isolate the stationary portion. This dictionary does not require an
        ‘output’ key to define the output to which the maximal latency should be sent. Instead, the output is
        automatically set to the conditions dataframe and directly uses the conditions defined in the ‘latency’
        analysis.

        The dictionary also contains the ‘params’ parameters, whose dictionary must contain a key to  define the name
        of the output to be created in the condition dataframe. The first key, ‘output’, allows you to define a specific
        name, while the second alternative key, ‘flag’, allows you to use the default output name by simply adding a
        suffix.

        Parameters
        ----------
        macular_analysis_dataframes : MacularAnalysisDataframes
            Macular Analyses Dataframes whose analyses the user wishes to use for meta-analysis.

        meta_analysis_dictionary : dict of tuple and dict of dict
            Meta-analysis dictionary linking the names of arguments in a meta-analysis with the associated array of
            values. In the case of arguments containing the term ‘output’, the key is associated with the name of the
            outputs created for the dataframe.

        index : dict of dict
            Dictionary of all indexes present in the multiple macular dict array used in the current
            MacularAnalysisDataframes.

        parameters_meta_analysis_dict : dict
            Dictionary containing all the parameters of the meta-analysis to be formatted.

            This dictionary don't contain any parameters other than output one.
        """
        # Store dimensions and conditions of output.
        meta_analysis_dictionary["output"]["dimension"] = "Conditions"
        meta_analysis_dictionary["output"]["condition"] = meta_analysis_dictionary["latency"][1]

        # Convert all non-outputs meta-analysis arguments levels into the corresponding analysis array.
        MacularAnalysisDataframes.extract_all_analysis_array_from_dataframes(macular_analysis_dataframes,
                                                                             meta_analysis_dictionary)

        stationary_latency = meta_analysis_dictionary["latency"].values[np.where(
            meta_analysis_dictionary["latency"].index.values > meta_analysis_dictionary["anticipation_range"])[0][0]:]

        # Calculation of the maximal latency
        maximal_latency_value = MetaAnalyser.mean_computing(stationary_latency)

        # Adds the output value(s) to a new row in the output dataframe.
        MacularAnalysisDataframes.add_array_line_to_dataframes(macular_analysis_dataframes,
                                                               meta_analysis_dictionary["output"]["dimension"],
                                                               meta_analysis_dictionary["output"]["condition"],
                                                               meta_analysis_dictionary["output"]["name"],
                                                               maximal_latency_value)

    @staticmethod
    @meta_analysis
    def subtraction_analyzing(macular_analysis_dataframes, meta_analysis_dictionary, index,
                              parameters_meta_analysis_dict):
        """Function that calculates a subtraction of one value by one or multiple values.

        To work, this meta-analysis requires two arguments: the "numerator" and the "denominator", which must defined in
        the meta-analysis dictionaries. A third key, ‘output’, must also be defined, which corresponds to the position
        where you want to save the result of the operation.

        Finally, the last key, “params”, must contain the key ‘factor’ with the value of the multiplication factor and
        the flag to be used as a suffix for the name of the meta-analysis.

        Parameters
        ----------
        macular_analysis_dataframes : MacularAnalysisDataframes
            Macular Analyses Dataframes whose analyses the user wishes to use for meta-analysis.

        meta_analysis_dictionary : dict of tuple and dict of dict
            Meta-analysis dictionary linking the names of arguments in a meta-analysis with the associated array of
            values. In the case of arguments containing the term ‘output’, the key is associated with the name of the
            outputs created for the dataframe.

        index : dict of dict
            Dictionary of all indexes present in the multiple macular dict array used in the current
            MacularAnalysisDataframes.

        parameters_meta_analysis_dict : dict
            Dictionary containing all the parameters of the meta-analysis to be formatted.

            This dictionary don't contain any parameters.
        """
        # Convert all non-outputs meta-analysis arguments levels into the corresponding analysis array.
        MacularAnalysisDataframes.extract_all_analysis_array_from_dataframes(macular_analysis_dataframes,
                                                                             meta_analysis_dictionary)

        values_subtracted = [meta_analysis_dictionary[argument] for argument in meta_analysis_dictionary
                             if "values_subtracted" in argument]

        # Calculation of the subctraction of the multiple analysis values.
        subtraction = MetaAnalyser.subtraction_computing(meta_analysis_dictionary["initial_value"],
                                                         values_subtracted)

        # Adds the output value(s) to a new row in the output dataframe.
        MacularAnalysisDataframes.add_array_line_to_dataframes(macular_analysis_dataframes,
                                                               meta_analysis_dictionary["output"]["dimension"],
                                                               meta_analysis_dictionary["output"]["condition"],
                                                               meta_analysis_dictionary["output"]["name"],
                                                               subtraction.astype(float)
                                                               )
