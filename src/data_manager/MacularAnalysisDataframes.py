import re

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
        self.multiple_dicts_analysis = self.cleaning_multiple_dicts_analysis()

        # Add multiple_dicts_simulations and multiple_dicts_preprocessings attributes.
        self.multiple_dicts_simulations = multiple_dicts_simulations
        self.multiple_dicts_preprocessings = multiple_dicts_preprocessings


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
        self._multiple_dicts_analysis = multiple_dicts_analysis

    @property
    def multiple_dicts_preprocessings(self):
        """Getter for the multiple_dicts_preprocessings attribute.
        """
        return self.multiple_dicts_preprocessings

    @multiple_dicts_preprocessings.setter
    def multiple_dicts_preprocessings(self, multiple_dicts_preprocessings):
        """Setter for the multiple_dicts_preprocessings attribute.
        """
        print("The attribute multiple_dicts_preprocessings can't be modified.")

    @property
    def multiple_dicts_simulations(self):
        """Getter for the multiple_dicts_simulations attribute.
        """
        return self.multiple_dicts_simulations

    @multiple_dicts_simulations.setter
    def multiple_dicts_simulations(self, multiple_dicts_simulations):
        """Setter for the multiple_dicts_simulations attribute.
        """
        print("The attribute multiple_dicts_simulations can't be modified.")

    def cleaning_multiple_dicts_analysis(self):
        """Cleans the analysis dictionary by removing all keys associated with a value of False.

        The purpose of this cleanup is to take into account that analysis missing from the analysis dictionary are
        equivalent to analysis that are present but with a value set to False.

        Returns
        ----------
        multiple_dicts_analysis_cleaned : dict of dict
            Multiple analysis dictionary with no keys associated with False values.
        """
        multiple_dicts_analysis_cleaned = self.multiple_dicts_analysis.copy()

        for dataframe in self.multiple_dicts_analysis:
            # Removal of false analyses.
            for analysis in self.multiple_dicts_analysis[dataframe]:
                if not self.multiple_dicts_analysis[dataframe][analysis]:
                    del multiple_dicts_analysis_cleaned[dataframe][analysis]

            # Removed empty dataframe analysis dictionaries.
            if not self.multiple_dicts_analysis[dataframe]:
                del multiple_dicts_analysis_cleaned[dataframe]

        return multiple_dicts_analysis_cleaned
