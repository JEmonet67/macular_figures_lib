import re

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

    name_value_unit_reg : re.Pattern
        Summary attr1

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

        # Create dict_paths_pyb attributes to store each path_pyb associated to its condition.
        self.dict_paths_pyb = {}
        for condition in multi_macular_dict_array:
            self.dict_paths_pyb[condition] = multi_macular_dict_array[condition].path_pyb

        # Create and clean the multiple_dicts_analysis attributes.
        self.multiple_dicts_analysis = multiple_dicts_analysis
        self.multiple_dicts_analysis = self.cleaning_multiple_dicts_features(multiple_dicts_analysis)

        # Create and clean the multiple_dicts_simulations attributes.
        self._multiple_dicts_simulations = multiple_dicts_simulations
        self._multiple_dicts_simulations = self.cleaning_multiple_dicts_features(multiple_dicts_simulations)

        # Create and clean the multiple_dicts_preprocessings attributes.
        self._multiple_dicts_preprocessings = multiple_dicts_preprocessings
        self._multiple_dicts_preprocessings = self.cleaning_multiple_dicts_features(multiple_dicts_preprocessings)


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
        multiple_dicts_features_cleaned = {feature: multiple_dicts_features[feature].copy() for feature in
                                           multiple_dicts_features}

        for dataframe in multiple_dicts_features:
            # Removal of false features.
            for feature in multiple_dicts_features[dataframe]:
                if not multiple_dicts_features[dataframe][feature]:
                    del multiple_dicts_features_cleaned[dataframe][feature]

            # Removed empty features dictionaries.
            if not multiple_dicts_features_cleaned[dataframe]:
                del multiple_dicts_features_cleaned[dataframe]

        return multiple_dicts_features_cleaned

