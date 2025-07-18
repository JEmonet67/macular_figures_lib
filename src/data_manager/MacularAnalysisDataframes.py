import pickle
import re
import copy
from functools import wraps

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.data_manager.MacularDictArray import MacularDictArray
from src.data_manager.MetaAnalyser import MetaAnalyser
from src.data_manager.SpatialAnalyser import SpatialAnalyser
from src.data_manager.ConditionsAnalyser import ConditionsAnalyser


class MacularAnalysisDataframes:
    """The MacularAnalysisDataframe is a data structure containing a set of analyses performed on different measurements
    and based on a particular dimension of several MacularDictArrays.

    The core of MacularAnalysisDataframes is a set of dataframes containing all the analyses performed. These dataframes
    are organised within a dictionary, grouped by dimension and, when necessary, by simulation conditions. Curently,
    MacularAnalysisDataframes consist of five different dimensions: horizontal (X) and vertical (Y) spatial dimensions,
    the time dimension, the simulation conditions dimension, and finally the MetaConditions dimension, which contains
    the analysis results summarising all conditions.

    The positions of the analyses or meta-analyses contained in these dataframes follow an analysis coordinate system.
    These coordinates are composed of five hierarchical levels present in the MacunarAnalysisDataframes, which allow
    each analysis to be distinguished. The first level is that of dimensions ("X", "Y", "Conditions"), which allows the
    dataframe to be determined. The second level is that of conditions ("barSpeed30dps", "ampGang30Hz"), which are used
    to determine either the dataframe or which column of the dataframe. The third is the simulation measurement used
    ("VSDI", "FiringRate_GanglionGainControl"), the fourth is the type of analysis used ("latency", "peak_amplitude"),
    and the last is a ‘flag’ label used to characterise each group of common analyses. These last three are used
    together to determine the name of the row where the analysis is located. Each analysis is therefore associated with
    a unique analysis coordinate tuple: (dimension, condition, measurement, analysis, flag).

    Each of these dimensional dataframes can be filled by performing various analyses. All analyses extract specific
    information from a series of MacularDictArrays, each associated with a simulation condition. Each MacularDictArray
    is characterised by data, indexes and two configuration dictionaries for simulation and for the pre-processing that
    needed to be carried out. These two dictionaries will be stored in the MacularAnalysisDataframes to keep track of
    these configuration files. However, the data and indexes are not stored as they are potentially too large.

    Each analysis is specific to one or more dimensions of the MacularAnalysisDataframes. The type of information
    extracted and how it is extracted varies depending on how the analysis was implemented. A number of default analyses
    have been created, but it is possible for each developer to contribute their own analyses. To do this, simply create
    the new analysis function with the three arguments ‘data’, “index” and ‘parameters_analysis_dict’ and then add it to
    the list of analyses available within the make_dimension_dataframes_analysis function of the corresponding dimension
    dataframe (e.g. make_spatial_dataframes_analysis).

    In addition to analyses, there are also meta-analyses, which define analyses based on other pre-existing analyses.
    Here again, there are default meta-analyses, but it is possible to add more within the make_meta_analysis_dataframes
    _analysis. Each meta-analysis function has four arguments: ‘macular_analysis_dataframes’,‘meta_analysis_dictionary’,
    “index”, ‘parameters_meta_analysis_dict’. Meta-analyses are a little more complex to create than analyses. This is
    because each one requires a number of arguments to be defined, which will contain the analyses used when calculating
    the meta-analysis, as well as outputs representing the output analyses to which the various results should be saved.
    A single meta-analysis can lead to several results that can be saved in completely different dataframes. The
    developer must determine which arguments and outputs are necessary for the meta-analysis. In practice, each argument
    and output can be associated with analyses from various dimensions and conditions. However, some outputs may also be
    dedicated to a particular dimension and/or condition. In this case these particular dimensions and conditions are
    defined in the implementation of the meta-analysis function and only the name of the output needs to be provided by
    the user. Some meta-analyses may also require an undefined and potentially unlimited number of outputs for a single
    meta-analysis. To do this, simply store them all in a single output in the form of a character string, all separated
    by ‘;’.

    In order to perform all these analyses, MacularAnalysisDataframes uses a master plan, which is the multiple analysis
    dictionary provided as input to the class constructor. This plan is structured as a succession of dictionaries
    nested within each other. The first level of the dictionary contains the path to the pyb file where
    MacularAnalysisDataframes is to be saved. This path will be removed from the dictionary and added to the dictionary
    attribute containing all file paths. The other keys in this first level are the four dimensions that can be used for
    analyses (X, Y, Time, Conditions) and the ‘MetaAnalysis’ key. The second level of the dictionary consists of all the
    types of analyses. From there, in the case of simple analyses, each type of analysis is associated with a list
    containing dictionaries of groups of common analyses. These are all the identical analyses performed on an entire
    batch of dimensions, conditions and measurements. The order of this list is important in defining the order in which
    each different analysis will be performed. This organisation allows for a more condensed writing of the lists of
    analyses to be performed. The dictionary of common analysis groups is thus composed here of the two keys
    ‘measurements’ and ‘conditions’. For each of these, we associate a character string containing all the names of
    measurements and conditions that are included in the group, all separated by ‘:’. In addition to this, we also have
    another key, ‘params’, associated with the dictionary of constant analysis parameters within the common analysis
    group. Once uncompressed, these dictionaries are transformed into a list containing all the coordinates of the
    analyses to be performed. Each coordinate corresponds to a combination of the different possible values for the
    measurements and conditions. When creating a new analysis (or meta-analysis), it is important not to forget to also
    add it to the multiple analysis dictionary.

    For meta-analyses, there is an intermediate dictionary level that is added to the list of common analysis groups.
    This level corresponds to the arguments and outputs of the meta-analysis. It also contains the ‘params’ key with the
    meta-analysis parameter dictionary. It is called a common meta-analysis group because it groups together all
    meta-analyses performed on different dimensions, conditions and measures. In the case of meta-analyses, we have a
    dictionary of common analysis groups composed of 5 keys corresponding to a hierarchical level different from the
    MacularAnalysisDataframes (dimensions, conditions, measurements, types of analyses and flag). Once again, it is in
    condensed form with the names that each hierarchical level will take, separated by ‘:’. Once decondensed, it will
    also provide all the coordinates of the analyses to be used for each argument and output.

    Attributes
    ----------
    dict_paths_pyb : dict of dict and dict of str
        Dictionary containing all important path pyb of MacularAnalysisDataframes.

        The dictionary is constructed with a first key ‘self’ associated with the path of the pyb file of the current
        MacularAnalysisDataframes. The second key ‘MacularDictArrays’ is linked to a dictionary containing all the paths
        of the conditions of the multiple MacularDictArray used.

    dict_analysis_dataframes : dict of dict or pd.DataFrame
        Dictionary containing all dataframes in which analyses or meta-analyses are placed. Each dataframe represents a
        dimension of the MacularAnalysisDataframes, whose columns are the different values of the dimension and the
        rows are the names of the analyses.

        The dictionary contains 5 dimensions associated with dataframes:
        - ‘X’ contains the analyses represented according to the spatial dimension X. Each column corresponds to a
        spatial index on the horizontal axis.
        - ‘Y’ contains the analyses represented according to the spatial dimension Y. Each column corresponds to a
        spatial index on the vertical axis.
        - ‘Time’ groups together the analyses expressed in terms of time. Each column corresponds to a time in the
        simulation.
        - ‘Conditions’ consists of analyses depending on the simulation condition. Each column is one simulation
        condition. The dataframe contains the first lines summarising the values of each of the parameters
        distinguishing each condition. To do this, the condition must follow the format ‘NameValueUnit1_NameValueUnit2’.
        - ‘MetaConditions’ is a dataframe that brings together all the analyses summarising all the conditions. It is
        composed of a single “overall” column.

        The spatial and temporal dimensions (X, Y, Time) of the dataframe dictionary are each associated with a second
        level of dictionary corresponding to the simulation conditions. This is because each spatial or temporal
        analysis performed depends on this condition.

        The positions of the analyses or meta-analyses contained in these dataframes follow an analysis coordinate
        system. These coordinates are composed of five hierarchical levels present in the MacunarAnalysisDataframes,
        which allow each analysis to be distinguished. The first level is that of dimensions, which allows the dataframe
        to be determined. The second level is that of conditions, which are used to determine either the dataframe or
        which column of the dataframe. The third is the simulation measure used, the fourth is the type of analysis
        used, and the last is a ‘flag’ label used to characterise each group of common analyses. These last three are
        used together to determine the name of the row where the analysis is located. Each analysis is therefore
        associated with a unique analysis coordinate tuple: (dimension, condition, measurement, analysis, flag).

    multiple_dicts_analysis : dict of dict
        Dictionaries containing all analyses or meta-analyses to be performed for each dimension of the
        MacularAnalysisDataframes, in a condensed format.

        The dictionary initially contains a key ‘path_pyb’ which corresponds to the path of the pyb file in which the
        MacularAnalysisDataframes will be saved. The rest of the dictionary consists of a series of dictionaries
        included in the previous one, each representing a hierarchical level of the analysis to be performed. The keys
        of the first dictionary are those of the dimensions of the MacularAnalysisDataframes (‘Conditions’, ‘X’, ‘Y’,
        ‘Time’, 'MetaAnalysis'). The keys of the second dictionary are those of the analyses to be performed on the
        given dimension. The rest of the data structure differs depending on whether you are considering analyses or
        meta-analyses. Analyses concern all dimensions of the first dictionary except for the ‘MetaAnalysis’ key, which
        corresponds to meta-analyses. Meta-analyses are analyses performed solely on the basis of other analyses from
        the MacularAnalysisDataframes.

        For each analyses, there is a list of dictionaries, each representing a group of common analyses. This is a set
        of measurements for certain conditions and dimensions that undergo the same analysis process (identical
        parameters). The order in which these dictionaries are entered is important, especially if two dictionaries
        cause analyses at the same position in the MacularAnalysisDataframes. The common analysis group dictionary
        contains a key for each hierarchical level of the MacularAnalysisDataframes. In the case of analyses, simply add
        the two levels ‘conditions’ and ‘measurements’ because the other two, “dimensions” and ‘analysis type’, are
        present in the keys of the dictionaries in which they are found. For each hierarchical level, enter a character
        string with all the names of the hierarchical levels for which the same analysis must be performed and separated
        by ‘:’ (‘barSpeed15dps:barSpeed30dps’ or ‘FiringRate_GanglionGainControl:VSDI’). Finally there is also a
        ‘params’ key that contains the dictionary of constant parameters for the analysis or meta-analysis. It is in
        this dictionary that the last hierarchical level ‘flag’ must be entered, which serves as a suffix indicating the
        group of common analyses. For example, to differentiate two ‘activation_time’ with two different thresholds, two
        flags can be used:‘flag’:‘threshold0,1’ and ‘flag’:‘threshold0,05’ which gives two column names in the
        dataframe: ‘activation_time_threshold0,1’ and 'activation_time_threshold0,05’. Note that there is an exception
        for the ‘sorting’ analysis located in the ‘Conditions’ dictionary. This is a global analysis that contains only
        a character string corresponding to the sorting type of the condition names in the conditions dataframe.

        In the case of meta-analyses, the meta-analysis key is associated with a list of dictionaries containing groups
        of common meta-analyses. The order of the list is important. Each group of common meta-analyses is a batch of
        analyses, measurements under certain conditions, dimensions sharing the same meta-analysis processing. Each of
        these dictionaries has a key for each of the arguments needed to calculate the meta-analysis, as well as outputs
        to which the various results should be sent. There is also a ‘params’ key with the meta-analysis parameter
        dictionary, which may also contain output names, but all of the coordinates for this output must be defined by
        the meta-analysis function. The number and name of the arguments and outputs (in params or not) are specified
        when the meta-analysis function is created. Each of the argument or output keys is associated with a dictionary
        of common analysis groups.

        These dictionaries of common analyses used for meta-analyses differ in a few ways from those used for analyses.
        Firstly, they contain the five hierarchical levels used as coordinates to find the position in the
        MacularAnalysisDataframes (dimensions, conditions, measurements, analyses, flag). In the case of meta-analysis
        outputs, the ‘measurements’ key in the dictionary may be empty as it is optional. However, it may be filled in
        for traceability purposes. It is also preferable to remove it when it comes to arguments retrieving results from
        previous meta-analyses. This absence of measurements is still considered by the system as a measure in itself
        and will suffice to differentiate between two analyses. Another exception concerns the possibility of placing
        the alias ‘overall’ within the “conditions” key when the dimension is that of ‘conditions’. This alias allows
        the meta-analysis to be performed directly on the array of analyses according to the conditions instead of
        performing a repeated meta-analysis for each condition. This alias is particularly useful in the context of
        meta-analyses with outputs directed to the 5th dimension ‘MetaConditions’, which is only accessible for
        meta-analyses. This dataframe groups together analyses summarising all the conditions. An example would be the
        calculation of slopes or inflection points of a fit of a given analysis expressed as a function of conditions.
        The dictionary of common analysis groups in the output must have a ‘dimensions’ key equal to ‘MetaConditions’
        and a “conditions” key equal to ‘overall’ which is the only column in the dataframe. In the case of
        meta-analyses, the meta-analysis key is associated with a list of dictionaries containing groups of common
        meta-analyses. The order of the list is important.  Each group of common meta-analyses is a batch of analyses,
        measurements under certain conditions, dimensions sharing the same meta-analysis processing. Each of these
        dictionaries has a key for each of the arguments needed to calculate the meta-analysis, as well as outputs to
        which the various results should be sent. There is also a ‘params’ key with the meta-analysis parameter
        dictionary, which may also contain output names, but all of the coordinates for this output must be defined by
        the meta-analysis function. The number and name of the arguments and outputs (in params or not) are specified
        when the meta-analysis function is created. Each of the argument or output keys is associated with a dictionary
        of common analysis groups.
        Example :"output": {"dimensions": "MetaConditions", "conditions": "overall", "measurements": "VSDI", "analyses":
        "horizontal_slope_anticipation_range"}

        The common analysis groups detailed above are used to reduce and condense the writing of all the analyses to be
        performed and their respective coordinates. The structure can be summarised as a dictionary with a key for each
        of the hierarchical levels of the MacularAnalysisDataframes (dimensions, conditions, measurements, analyses,
        flag). Each key is associated with a character string containing all the names of the hierarchical levels
        included in the common analysis group. These names are separated by ‘:’. In the case of the “analyses” and
        ‘flag’ levels, only one hierarchical level name is allowed. All equivalent analyses of common analysis groups
        will be performed indifferently for all combinations of dimensions, conditions and measurements present in the
        group. Thus, after decondensation, a list of analysis coordinates corresponding to all combinations between the
        different names of hierarchical levels is obtained. In the case of a meta-analysis, a list of analysis
        coordinates is available for each argument and output. In this context, it is crucial that all lists are the
        same size. To achieve this, it is important to use common analysis groups with the same number of dimension
        names, conditions and measurements. An exception is allowed for lists of size 1, which will be repeated as many
        times as necessary for each analysis.

        Global aliases can be used for each element used in common analysis group dictionaries. These aliases are used
        when you want to use all the hierarchical level names available in the MacularAnalysisDataframes. This avoids
        having to enter all the names one by one. In the case of “measurements” and “analyses”, the possible
        hierarchical level names vary depending on the dimensions and conditions present in the multiple analysis group.
        These aliases are constructed from the name of the hierarchical level for which all available names are desired,
        preceded by the prefix ‘all_’. These aliases are substituted within the MacularAnalysisDataframes by the getter
        of the multiple analysis dictionary. This allows both the original dictionary and the substituted dictionary to
        be retained.

        Example :
        "path_pyb": "path/to/file.pyb,
        "Conditions": {
            "sorting": "NameValueUnit",
            "peak_amplitude": [{"conditions": "all_conditions",
                            "measurements": "FiringRate_GanglionGainControl:VSDI:"
                                            "muVn_CorticalExcitatory:muVn_CorticalInhibitory",
                            "params": {"x": 36, "y": 7, "flag": ""}}]
        },
        "X": {
            "activation_time": [{"conditions": "all_conditions", "measurements": "VSDI",
                             "params": {"threshold": 0.001, "threshold_type": "static", "index": "temporal_ms", "y": 7,
                                        "flag": "ms"}}],
            "latency": [{"conditions": "all_conditions", "measurements": "VSDI",
                     "params": {"threshold": 0.001, "threshold_type": "static", "index": "temporal_centered_ms",
                                "y": 7, "axis": "horizontal", "flag": "ms"}}],
            "time_to_peak": [{"conditions": "all_conditions", "measurements": "VSDI:FiringRate_GanglionGainControl",
                          "params": {"index": "temporal_ms", "y": 7, "flag": "ms"}}],
            "spatial_mean": [{"conditions": "all_conditions", "measurements": "vertical_mean_section_max_ratio_threshold",
                          "params": {"axis": 0, "flag": ""}}]
        },
        "Y": {
            "activation_time": [{"conditions": "all_conditions", "measurements": "VSDI",
                                 "params": {"threshold": 0.001, "threshold_type": "static", "index": "temporal_ms",
                                            "x": 36, "flag": "ms"}}],
            "time_to_peak": [{"conditions": "all_conditions", "measurements": "VSDI:FiringRate_GanglionGainControl",
                              "params": {"index": "temporal_ms", "x": 36, "flag": "ms"}}]
        },
        "MetaAnalysis": {
            "peak_speed": [
                {"time_to_peak": {"dimensions": "X", "conditions": "all_conditions", "measurements": "VSDI",
                                  "analyses": "time_to_peak", "flag": "ms"},
                 "params": {"output": "horizontal_peak_speed", "index": "spatial_x", "n_points": 100, "breaks": "auto"}}
            ],
            "stationary_peak_delay": [
                {"peak_delay": {"dimensions": "X", "conditions": "all_conditions", "measurements": "VSDI",
                                "analyses": "peak_delay", "flag": "ms"},
                 "params": {"output": "VSDI_horizontal_stationary_peak_delay_ms"}},
            ],
            "linear_fit": [
                {"data_to_fit": {"dimensions": "X", "conditions": "all_conditions", "measurements": "VSDI",
                                 "analyses": "activation_time", "flag": "ms"},
                 "output_slopes": {"dimensions": "Conditions", "conditions": "all_conditions", "measurements": "VSDI",
                                   "analyses": "horizontal_first_slope_speed;horizontal_second_slope_speed;"
                                               "horizontal_third_slope_speed;horizontal_fourth_slope_speed"},
                 "output_inflection_points_data": {"dimensions": "Conditions", "conditions": "all_conditions",
                                                   "measurements": "VSDI",
                                                   "analyses": "horizontal_first_inflection_point;"
                                                               "horizontal_second_inflection_point;"
                                                               "horizontal_third_inflection_point"},
                 "output_inflection_points_index": {"dimensions": "Conditions", "conditions": "all_conditions",
                                                    "measurements": "VSDI",
                                                    "analyses": "horizontal_first_inflection_point_time;"
                                                                "horizontal_second_inflection_point_time;"
                                                                "horizontal_third_inflection_point_time"},
                 "output_index_prediction": {"dimensions": "X", "conditions": "all_conditions",
                                             "measurements": "VSDI",
                                             "analyses": "horizontal_data_to_fit_index_prediction"},
                 "output_data_prediction": {"dimensions": "X", "conditions": "all_conditions",
                                            "measurements": "VSDI",
                                            "analyses": "horizontal_data_to_fit_data_prediction"},
                 "output_data_intercepts": {"dimensions": "Conditions", "conditions": "all_conditions",
                                            "measurements": "VSDI",
                                            "analyses": "first_horizontal_data_intercept_VSDI;"
                                                        "second_horizontal_data_intercept_VSDI"},
                 "output_index_intercepts": {"dimensions": "Conditions", "conditions": "all_conditions",
                                             "measurements": "VSDI",
                                             "analyses": "first_horizontal_data_intercept_VSDI,5dps_VSDI;"
                                                         "second_horizontal_data_intercept_VSDI"},
                 "params": {"n_segments": 4, "index": "spatial_x", "n_points": 100, "breaks": "auto"}},

                {"data_to_fit": {"dimensions": "Conditions", "conditions": "overall", "measurements": "",
                                 "analyses": "horizontal_anticipation_range", "flag": ""},
                 "output_slopes": {"dimensions": "MetaConditions", "conditions": "overall", "measurements": "VSDI",
                                   "analyses": "horizontal_slope_anticipation_range"},
                 "params": {"n_segments": 1, "index": "barSpeed", "n_points": 100, "breaks": "auto"}},
            ],
            "anticipation_fit": [
                {"activation_time": {"dimensions": "X", "conditions": "all_conditions", "measurements": "VSDI",
                                     "analyses": "activation_time", "flag": "ms"},
                 "params": {"output_slopes": "horizontal_short_range_anticipation_speed_dpms;"
                                             "horizontal_long_range_anticipation_speed_dpms",
                            "output_anticipation_range": "horizontal_anticipation_range",
                            "output_index_prediction": "horizontal_anticipation_index_prediction",
                            "output_data_prediction": "horizontal_anticipation_data_prediction",
                            "n_segments": 2, "index": "spatial_x", "n_points": 100, "breaks": "auto"}}
            ],
            "minimal_latency": [
                {"latency": {"dimensions": "X", "conditions": "all_conditions", "measurements": "VSDI",
                             "analyses": "latency", "flag": "ms"},
                 "anticipation_range": {"dimensions": "Conditions", "conditions": "all_conditions",
                                        "measurements": "",
                                        "analyses": "horizontal_anticipation_range", "flag": ""},
                 "params": {"output": "horizontal_minimal_latency_ms", "index": "spatial_x"}}
            ],
            "subtraction": [
                {"initial_value": {"dimensions": "Conditions", "conditions": "all_conditions",
                                   "measurements": "muVn_CorticalExcitatory", "analyses": "peak_amplitude",
                                   "flag": ""},
                 "values_subtracted": {"dimensions": "Conditions", "conditions": "all_conditions",
                                       "measurements": "muVn_CorticalExcitatory",
                                       "analyses": "initial_amplitude", "flag": ""},
                 "output": {"dimensions": "Conditions", "conditions": "all_conditions",
                            "measurements": "muVn_CorticalExcitatory", "analyses": "subtraction_excitatory_mean_voltage"},
                 "params": {}}
            ],
            "normalization": [
                {"value_to_normalize": {"dimensions": "Conditions", "conditions": "all_conditions",
                                        "measurements": "muVn_CorticalExcitatory", "analyses": "peak_amplitude",
                                        "flag": ""},
                 "baseline": {"dimensions": "Conditions", "conditions": "all_conditions",
                              "measurements": "muVn_CorticalExcitatory", "analyses": "initial_amplitude", "flag": ""},
                 "output": {"dimensions": "Conditions", "conditions": "all_conditions",
                            "measurements": "muVn_CorticalExcitatory", "analyses": "normalized_excitatory_mean_voltage"},
                 "params": {"factor": 1}}
            ]
        }

    multiple_dicts_simulations : dict of dict
        Dictionary grouping together dictionaries of simulations of each condition present in the multiple macular
        dict array analysed in this MacularAnalysisDataframes.

        Each dictionary is retrieved from each MacularDictArray. They are therefore already cleaned of any empty fields
        and do not contain any pyb file paths or ‘global’ keys.

    multiple_dicts_preprocessings : dict of dict
        Dictionary grouping together dictionaries of preprocessing of each condition present in the multiple macular
        dict array analysed in this MacularAnalysisDataframes.

        Each dictionary is retrieved from each MacularDictArray. They are therefore already cleaned of any empty fields
        and do not contain any ‘global’ keys.

    condition_reg : re.Pattern
        Regular expression to extract the name of the condition, its value and its unit from the keys of the
        MacularDictArray multiple dictionary.

        This pattern is primarily used to sort conditions in the ‘Conditions’ dataframe of MacularAnalysisDataframes.
        By default, the regular expression entered allows you to read conditions that follow a ‘NameValueUnit’ format.
        For example : "barSpeed6dps" or "ampGang30Hz".

    analysis_dataframes_levels : dict of str or dict of dict
        Container grouping all the names of conditions, measurements, dimensions and analyses found in the
        MacularAnalysisDataframes or associated multiple Macular Dict Array, separated by ‘:’.

        The first key "conditions" is associated to a character string with all conditions names. The second key,
        ‘measurements’, is associated with another dictionary level linking each MacularDictArray condition with all the
        measurements they contain. The third key, “dimensions”, is associated with a character string containing all the
        dimension dataframes contained in the current MacularAnalysisDataframes. The last key, ‘analyses’, contains two
        other dictionary levels. The first dictionary corresponds to the dimensions of the dataframes, while the second
        contains the conditions for each MacularDictArray. This allows all analyses for each dimension dataframe and for
        each condition to be stored.

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
    """

    def __init__(self, multi_macular_dict_array, multiple_dicts_analysis):
        """Function for constructing a MacularAnalysisDataframes.

        The function begins by extracting the path to the pyb file and removing it from the multiple analysis
        dictionary. It continues by cleaning up the same dictionary to remove any empty fields. Once finished, it can
        begin creating or loading the MacularAnalysisDataframes.

        Before creating a new MacularAnalysisDataframes, the function will search for the existence of a pyb file at the
        path specified in the multiple analysis dictionary. If the file does not exist,  a new MacularAnalysisDataframes
        will be constructed and saved at that location. If a file exists, it is loaded to save time. The multiple
        analysis dictionary of the MacularAnalysisDataframes that has been loaded is then compared with the one given as
        input to the __init__ function. If there is a difference,between the two dictionaries, it will be up to the user
        to decide whether to keep the pyb loaded from the file or creating a new MacularAnalysisDataframes based on the
        multiple analysis dictionary provided as input. In this case, the MacularAnalysisDataframes will be saved and
        will therefore replace the existing pyb file.

        In the event that the multiple analysis dictionary provided as input contains only a single key for the path to
        the pyb file, then the MacularAnalysisDataframes will simply be loaded without comparisons. In this context, the
        multiple analysis dictionary would be empty once the pyb path is extracted.

        Parameters
        ----------
        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with different MacularDictArray.

            Each MacularDictArray is defined by a set of data, indexes, a dictionary for configuring the simulation,
            and another for the pre-processing it has undergone.

        multiple_dicts_analysis : dict of dict
            Dictionaries containing all analyses or meta-analyses to be performed for each dimension of the
            MacularAnalysisDataframes, in a condensed format.

            The dictionary initially contains a key ‘path_pyb’ which corresponds to the path of the pyb file in which
            the MacularAnalysisDataframes will be saved. The rest of the dictionary consists of a series of dictionaries
            included in the previous one, each representing a hierarchical level of the analysis to be performed. The
            keys of the first dictionary are those of the dimensions of the MacularAnalysisDataframes (‘Conditions’,
            ‘X’, ‘Y’, ‘Time’, 'MetaAnalysis'). The keys of the second dictionary are those of the analyses to be
            performed on the given dimension. The rest of the data structure differs depending on whether you are
            considering analyses or meta-analyses. Analyses concern all dimensions of the first dictionary except for
            the ‘MetaAnalysis’ key, which corresponds to meta-analyses. Meta-analyses are analyses performed solely on
            the basis of other analyses from the MacularAnalysisDataframes.

            For each analyses, there is a list of dictionaries, each representing a group of common analyses. This is a
            set of measurements for certain conditions and dimensions that undergo the same analysis process (identical
            parameters). The order in which these dictionaries are entered is important, especially if two dictionaries
            cause analyses at the same position in the MacularAnalysisDataframes. The common analysis group dictionary
            contains a key for each hierarchical level of the MacularAnalysisDataframes. In the case of analyses, simply
            add the two levels ‘conditions’ and ‘measurements’ because the other two, “dimensions” and ‘analysis type’,
            are present in the keys of the dictionaries in which they are found. For each hierarchical level, enter a
            character string with all the names of the hierarchical levels for which the same analysis must be performed
            and separated by ‘:’ (‘barSpeed15dps:barSpeed30dps’ or ‘FiringRate_GanglionGainControl:VSDI’). Finally,
            there is also a ‘params’ key that contains the dictionary of constant parameters for the analysis or
            meta-analysis. This dictionary is where the last hierarchical level ‘flag’ must be entered, which serves as
            a suffix indicating the group of common analyses. For example, to differentiate two ‘activation_time’ with
            two different thresholds, two flags can be used:‘flag’:‘threshold0,1’ and ‘flag’:‘threshold0,05’ which gives
            two column names in the dataframe: ‘activation_time_threshold0,1’ and 'activation_time_threshold0,05’. Note
            that there is an exception for the ‘sorting’ analysis located in the ‘Conditions’ dictionary. This is a
            global analysis that contains only a character string corresponding to the sorting type of the condition
            names in the conditions dataframe.

            In the case of meta-analyses, the meta-analysis key is associated with a list of dictionaries containing
            groups of common meta-analyses. The order of the list is important. Each group of common meta-analyses is a
            batch of analyses, measurements under certain conditions, dimensions sharing the same meta-analysis
            treatment. Each of these dictionaries has a key for each of the arguments needed to calculate the
            meta-analysis, as well as outputs to which the various results should be sent. There is also a ‘params’ key
            with the meta-analysis parameter dictionary, which may also contain output names, but all the coordinates
            for this output must be defined by the meta-analysis function. The number and name of the arguments and
            outputs (in params or not) are specified when the meta-analysis function is created. Each of the argument or
            output keys are associated with a dictionary of common analysis groups. This dictionary contains the five
            hierarchical levels used as coordinates to find the position in the MacularAnalysisDataframes (dimensions,
            conditions, measurements, analyses, flag).

            Global aliases can be used for each element used in common analysis group dictionaries. These aliases are
            used when you want to use all the hierarchical level names available in the MacularAnalysisDataframes. This
            avoids having to enter all the names one by one. In the case of ‘measurements’ and 'analyses', the possible
            hierarchical levels names vary depending on the dimensions and conditions present in the multiple analysis
            group. These aliases are constructed from the name of the hierarchical level for which all available names
            are desired, preceded by the prefix ‘all_’. These aliases are substituted within the
            MacularAnalysisDataframes by the getter of the multiple analysis dictionary. Be careful to only group
            together analyses that share the same configurations.
        """
        # Storing of the pyb path.
        path_pyb = multiple_dicts_analysis["path_pyb"]

        # Deletion of the pyb path of the current MacularDictArray from a copy of the multiple analysis dictionary.
        multiple_dicts_analysis_copy = multiple_dicts_analysis.copy()
        del multiple_dicts_analysis_copy["path_pyb"]

        # Clean the multiple_dicts_analysis attributes.
        multiple_dicts_analysis_copy = self.cleaning_multiple_dicts_features(multiple_dicts_analysis_copy)

        # Import a pyb file with the same name or create a new MacularAnalysisDataframes from the dictionary.
        self.managing_pre_existing_file(path_pyb, multi_macular_dict_array, multiple_dicts_analysis_copy)

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

    def __repr__(self):
        """Function to display a MacularAnalysisDataframes.

        Example of one dataframe :
         ######## barSpeed28,5dps Y dataframe ########
        ╒═════════════════════════════════════════════════╤══════════╤═════════╤═══════════╤═══════════╤═══════════╤══════════╤═══════════╤═══════════╤═══════════╤═══════════╤═══════════╤═══════════╤═══════════╤════════════╤══════════╕
        │                                                 │      0.0 │   0.225 │      0.45 │     0.675 │       0.9 │    1.125 │      1.35 │     1.575 │       1.8 │     2.025 │      2.25 │     2.475 │       2.7 │      2.925 │     3.15 │
        ╞═════════════════════════════════════════════════╪══════════╪═════════╪═══════════╪═══════════╪═══════════╪══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪════════════╪══════════╡
        │ activation_time_VSDI_ms                         │ 290.2    │ 295     │ 291.8     │ 290.2     │ 288.6     │ 287      │ 287       │ 285.4     │ 287       │ 287       │ 288.6     │ 290.2     │ 291.8     │ 295        │ 290.2    │
        ├─────────────────────────────────────────────────┼──────────┼─────────┼───────────┼───────────┼───────────┼──────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┤
        │ time_to_peak_VSDI_ms                            │ 523.8    │ 517.4   │ 514.2     │ 517.4     │ 511       │ 485.4    │ 477.4     │ 475.8     │ 475.8     │ 483.8     │ 509.4     │ 515.8     │ 514.2     │ 517.4      │ 523.8    │
        ├─────────────────────────────────────────────────┼──────────┼─────────┼───────────┼───────────┼───────────┼──────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┤
        │ time_to_peak_FiringRate_GanglionGainControl_ms  │ 471      │ 471     │ 471       │ 471       │ 471       │ 471      │ 471       │ 471       │ 471       │ 471       │ 471       │ 471       │ 471       │ 471        │ 471      │
        ├─────────────────────────────────────────────────┼──────────┼─────────┼───────────┼───────────┼───────────┼──────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┤
        │ peak_amplitude_VSDI                             │   0.014  │   0.012 │   0.013   │   0.014   │   0.016   │   0.027  │   0.038   │   0.039   │   0.038   │   0.028   │   0.016   │   0.014   │   0.013   │   0.012    │   0.015  │
        ├─────────────────────────────────────────────────┼──────────┼─────────┼───────────┼───────────┼───────────┼──────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┤
        │ peak_amplitude_FiringRate_GanglionGainControl   │   0.002  │   0.01  │   0.053   │   0.265   │   1.128   │   3.57   │   5.985   │   6.672   │   6.028   │   3.664   │   1.173   │   0.277   │   0.056   │   0.011    │   0.002  │
        ├─────────────────────────────────────────────────┼──────────┼─────────┼───────────┼───────────┼───────────┼──────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┤
        │ spatial_mean_horizontal_mean_section_fixed_edge │   0.0024 │   0.002 │   0.0022  │   0.0023  │   0.0026  │   0.0044 │   0.0061  │   0.0063  │   0.0061  │   0.0045  │   0.0026  │   0.0023  │   0.0022  │   0.002    │   0.0024 │
        ├─────────────────────────────────────────────────┼──────────┼─────────┼───────────┼───────────┼───────────┼──────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┤
        │ spatial_peak_amplitudes_normalization           │  48      │   1.6   │  -6.03774 │  -7.57736 │  -7.88652 │  -7.9395 │  -7.94921 │  -7.95324 │  -7.94957 │  -7.93886 │  -7.89088 │  -7.59567 │  -6.14286 │   0.727273 │  52      │
        ╘═════════════════════════════════════════════════╧══════════╧═════════╧═══════════╧═══════════╧═══════════╧══════════╧═══════════╧═══════════╧═══════════╧═══════════╧═══════════╧═══════════╧═══════════╧════════════╧══════════╛
        """
        # Initialisation of text to be displayed.
        str_to_display = ""

        # Loop through the dimensions to display.
        for dimension in self.analysis_dataframes_levels["dimensions"].split(":"):
            # Single dataframe cases.
            if dimension in ["Conditions", "MetaConditions"]:
                # Increment the display by adding the one of the current dataframe.
                str_to_display += f"\n\n ######## {dimension} dataframe ######## \n"
                str_to_display += tabulate(self.dict_analysis_dataframes[dimension], headers='keys',
                                           tablefmt='fancy_grid')

            # Case of dataframes divided into several conditions.
            else:
                # Loop through the dimensions to display.
                for condition in self.analysis_dataframes_levels["conditions"].split(":"):
                    # Increment the display by adding the one of the current dataframe.
                    str_to_display += f"\n\n ######## {condition} {dimension} dataframe ######## \n"
                    str_to_display += tabulate(self.dict_analysis_dataframes[dimension][condition], headers='keys',
                                               tablefmt='fancy_grid')

        return str_to_display

    def print_specific_dataframes(self, dimensions="all", conditions="all"):
        """Function to display specific dataframes of MacularAnalysisDataframes.

        It is possible to specify which conditions or dimensions are displayed. To do this, the two corresponding
        parameters are given a character string containing the names of the conditions and dimensions separated by ‘:’.

        Example :
         ######## barSpeed28,5dps Y dataframe ########
        ╒═════════════════════════════════════════════════╤══════════╤═════════╤═══════════╤═══════════╤═══════════╤══════════╤═══════════╤═══════════╤═══════════╤═══════════╤═══════════╤═══════════╤═══════════╤════════════╤══════════╕
        │                                                 │      0.0 │   0.225 │      0.45 │     0.675 │       0.9 │    1.125 │      1.35 │     1.575 │       1.8 │     2.025 │      2.25 │     2.475 │       2.7 │      2.925 │     3.15 │
        ╞═════════════════════════════════════════════════╪══════════╪═════════╪═══════════╪═══════════╪═══════════╪══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪════════════╪══════════╡
        │ activation_time_VSDI_ms                         │ 290.2    │ 295     │ 291.8     │ 290.2     │ 288.6     │ 287      │ 287       │ 285.4     │ 287       │ 287       │ 288.6     │ 290.2     │ 291.8     │ 295        │ 290.2    │
        ├─────────────────────────────────────────────────┼──────────┼─────────┼───────────┼───────────┼───────────┼──────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┤
        │ time_to_peak_VSDI_ms                            │ 523.8    │ 517.4   │ 514.2     │ 517.4     │ 511       │ 485.4    │ 477.4     │ 475.8     │ 475.8     │ 483.8     │ 509.4     │ 515.8     │ 514.2     │ 517.4      │ 523.8    │
        ├─────────────────────────────────────────────────┼──────────┼─────────┼───────────┼───────────┼───────────┼──────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┤
        │ time_to_peak_FiringRate_GanglionGainControl_ms  │ 471      │ 471     │ 471       │ 471       │ 471       │ 471      │ 471       │ 471       │ 471       │ 471       │ 471       │ 471       │ 471       │ 471        │ 471      │
        ├─────────────────────────────────────────────────┼──────────┼─────────┼───────────┼───────────┼───────────┼──────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┤
        │ peak_amplitude_VSDI                             │   0.014  │   0.012 │   0.013   │   0.014   │   0.016   │   0.027  │   0.038   │   0.039   │   0.038   │   0.028   │   0.016   │   0.014   │   0.013   │   0.012    │   0.015  │
        ├─────────────────────────────────────────────────┼──────────┼─────────┼───────────┼───────────┼───────────┼──────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┤
        │ peak_amplitude_FiringRate_GanglionGainControl   │   0.002  │   0.01  │   0.053   │   0.265   │   1.128   │   3.57   │   5.985   │   6.672   │   6.028   │   3.664   │   1.173   │   0.277   │   0.056   │   0.011    │   0.002  │
        ├─────────────────────────────────────────────────┼──────────┼─────────┼───────────┼───────────┼───────────┼──────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┤
        │ spatial_mean_horizontal_mean_section_fixed_edge │   0.0024 │   0.002 │   0.0022  │   0.0023  │   0.0026  │   0.0044 │   0.0061  │   0.0063  │   0.0061  │   0.0045  │   0.0026  │   0.0023  │   0.0022  │   0.002    │   0.0024 │
        ├─────────────────────────────────────────────────┼──────────┼─────────┼───────────┼───────────┼───────────┼──────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼────────────┼──────────┤
        │ spatial_peak_amplitudes_normalization           │  48      │   1.6   │  -6.03774 │  -7.57736 │  -7.88652 │  -7.9395 │  -7.94921 │  -7.95324 │  -7.94957 │  -7.93886 │  -7.89088 │  -7.59567 │  -6.14286 │   0.727273 │  52      │
        ╘═════════════════════════════════════════════════╧══════════╧═════════╧═══════════╧═══════════╧═══════════╧══════════╧═══════════╧═══════════╧═══════════╧═══════════╧═══════════╧═══════════╧═══════════╧════════════╧══════════╛
        """
        # Gets all dimensions present in the MacularAnalysisDataframes.
        if dimensions == "all":
            dimensions = self.analysis_dataframes_levels["dimensions"]

        # Gets all conditions present in the MacularAnalysisDataframes.
        if conditions == "all":
            conditions = self.analysis_dataframes_levels["conditions"]

        # Initialisation of text to be displayed.
        str_to_display = ""

        # Loop through the dimensions to display.
        for dimension in dimensions.split(":"):
            # Single dataframe cases.
            if dimension in ["Conditions", "MetaConditions"]:
                # Increment the display by adding the one of the current dataframe.
                str_to_display += f"\n\n ######## {dimension} dataframe ######## \n"
                str_to_display += tabulate(self.dict_analysis_dataframes[dimension], headers='keys',
                                           tablefmt='fancy_grid')


            # Case of dataframes divided into several conditions.
            else:
                # Loop through the dimensions to display.
                for condition in conditions.split(":"):
                    # Increment the display by adding the one of the current dataframe.
                    str_to_display += f"\n\n ######## {condition} {dimension} dataframe ######## \n"
                    str_to_display += tabulate(self.dict_analysis_dataframes[dimension][condition], headers='keys',
                                               tablefmt='fancy_grid')

        print(str_to_display)
        return str_to_display

    def make_from_dictionary(self, path_pyb, multi_macular_dict_array, multiple_dicts_analysis):
        """Creation of a new MacularAnalysisDataframes based on the multiple analysis dictionary and the multiple
        macular dict array provided as input by the user.

        The MacularAnalysisDataframes is first initialised to create all its attributes. All analysis dataframes are
        also constructed with their respective indexes. Once this initialisation is complete, all analyses in the
        multiple analysis dictionary are performed. The analyses of each dataframe are performed one at a time. When
        all analyses are complete, the resulting MacularAnalysisDataframes is saved using the pyb file path provided
        by the user.

        Parameters
        ----------
        path_pyb : str
            Path to the pyb file to be created.

        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with different MacularDictArray.

        multiple_dicts_analysis : dict of dict
            Dictionaries containing all analyses or meta-analyses to be performed for each dimension of the
            MacularAnalysisDataframes, in a condensed format.
        """
        # Initialisation of the pyb path dictionary with that of MacularAnalysisDataframes.
        self._dict_paths_pyb = {"self": path_pyb, "MacularDictArrays": {}}

        # Initialisation of the analysis dataframes dictionary and the dictionary of all the indexes it contains.
        dict_index = self.initialize_macular_analysis_dataframes(multi_macular_dict_array, multiple_dicts_analysis)

        # Make analysis
        self.make_spatial_dataframes_analysis("X", multi_macular_dict_array)
        self.make_spatial_dataframes_analysis("Y", multi_macular_dict_array)
        # self.make_temporal_dataframes_analysis(multi_macular_dict_array)
        self.make_conditions_dataframes_analysis(multi_macular_dict_array)

        # Extract the dimensions/analyses levels from the MacularAnalysisDataframes.
        self._analysis_dataframes_levels.update(self.get_levels_of_macular_analysis_dataframes())

        # #Make meta-analysis with a dictionary of all indexes present in the multiple macular dict array.
        self.make_meta_analysis_dataframes_analysis(dict_index)

        # Saving the MacularAnalysisDataframes.
        self.save()

    def update_from_file(self, path_pyb):
        """Method for updating a MacularAnalysisDataframes object by replacing it with another MacularAnalysisDataframes
        object contained in a binary pyb file.

        Parameters
        ----------
        path_pyb : str
            Path to file with .pyb extension where a MacularAnalysisDataframes object is saved in binary.

            The path can be absolute or relative.

        """
        print("FILE UPDATING...", end="")
        with open(path_pyb, "rb") as pyb_file:
            tmp_dict = pickle.load(pyb_file).__dict__
        self.__dict__.clear()
        self.__dict__.update(tmp_dict)
        print("UPDATED!")

    def managing_pre_existing_file(self, path_pyb, multi_macular_dict_array, multiple_dicts_analysis):
        """Manages whether a pyb file corresponding to the path of the pyb file provided by the user exists.

        If the pyb file exists, it is imported into the MacularAnalysisDataframes as a priority to save time and avoid
        additional processing. If it does not exist, the multiple macular dictionaries are processed to create a new
        MacularAnalysisDataframes which will be saved. If the pyb file exists, a comparison will be made between its
        multiple analysis dictionary after import and the one provided by the user to check whether a file/json
        inconsistency needs to be handled.

        Parameters
        ----------
        path_pyb : str
            Path to the pyb file to be loaded or created.

        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with different MacularDictArray.

        multiple_dicts_analysis : dict of dict
            Dictionaries containing all analyses or meta-analyses to be performed for each dimension of the
            MacularAnalysisDataframes, in a condensed format.
        """
        try:
            # Update MacularAnalysisDataframes from an existing file if possible.
            self.update_from_file(path_pyb)

            # The comparison with json occurs if the multiple analysis dictionary does not contain only the pyb path.
            if len(multiple_dicts_analysis.keys()) != 0:
                self.checking_difference_file_json(path_pyb, multi_macular_dict_array, multiple_dicts_analysis)

        except (FileNotFoundError, EOFError):
            # Construction of a MacularAnalysisDataframes from the dictionaries if no file exists.
            print("NO FILE FOR THE UPDATE. Using the dictionaries.")
            self.make_from_dictionary(path_pyb, multi_macular_dict_array, multiple_dicts_analysis)

    def checking_difference_file_json(self, path_pyb, multi_macular_dict_array, multiple_dicts_analysis):
        """Comparison between the multiple analysis dictionary contained in the imported pyb and that specified in the
        init function of MacularDictArray.

        The verification is carried out on the elements of the multiple analysis dictionaries. In case of a difference
        between the two, it is up to the user to choose between the dictionaries contained in the pyb or the one entered
        as input for the init function.

        Parameters
        ----------
        path_pyb : str
            Path to the pyb file containing a MacularAnalysisDataframes to be compared with the multiple analysis
            dictionary entered by the user.

        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with different MacularDictArray.

        multiple_dicts_analysis : dict of dict
            Dictionaries containing all analyses or meta-analyses to be performed for each dimension of the
            MacularAnalysisDataframes, in a condensed format.

        Raises
        ----------
        ValueError
            The value error is raised in the event of an incorrect response from the user.
        """
        # Checking difference between both multiple analysis dictionary.
        if self._multiple_dicts_analysis != multiple_dicts_analysis:
            print("Multiple analysis dictionary differ...")
            user_choice = input("Which configuration should be kept ? json or pyb : ").lower()
            # Conservation of the json file.
            if user_choice == "json":
                self.make_from_dictionary(path_pyb, multi_macular_dict_array, multiple_dicts_analysis)
            # Conservation of the pyb file.
            elif user_choice == "pyb":
                pass
            # Incorrect user response.
            else:
                raise ValueError("Incorrect configuration")

            return 1
        return 0

    @classmethod
    def equal(cls, macular_analysis_dataframe1, macular_analysis_dataframe2):
        """Checking the equality between two MacularAnalysisDataframes.

        MacularAnalysisDataframes are equal if they have the same attributes with the same values. Two MacularDictArray
        are equal if they have the same attributes and values associated with each of these attributes. Only the
        path_pyb and conditions_reg attribute can differ between the two MacularAnalysisDataframes.

        Parameters
        ----------
        macular_analysis_dataframe1 : MacularAnalysisDataframes
            First MacularAnalysisDataframes to compare.

        macular_analysis_dataframe1 : MacularAnalysisDataframes
            Second MacularAnalysisDataframes to compare.

        Returns
        ----------
        equality : Bool
            Returns True if both MacularAnalysisDataframes are equal and False otherwise.
        """
        equality = True

        # Equality between the attributes of the two MacularDictArray.
        if macular_analysis_dataframe1.__dict__.keys() == macular_analysis_dataframe2.__dict__.keys():
            # Dictionary attributes search.
            for attributes in macular_analysis_dataframe1.__dict__:
                # Case of the dict analysis dataframes attributes.
                if attributes == "_dict_analysis_dataframes":
                    # Equality between the dataframes contained in dict analysis dataframes attribute.
                    equality = equality & (cls.equal_dict_analysis_dataframes(macular_analysis_dataframe1.__dict__[attributes],
                                                                macular_analysis_dataframe2.__dict__[attributes]))
                # Case of the dict path pyb attribute, which is ignored.
                elif attributes in ("_dict_paths_pyb", "_condition_reg"):
                    continue

                # Case of other attributes.
                else:
                    equality = equality & (
                            macular_analysis_dataframe1.__dict__[attributes] == macular_analysis_dataframe2.__dict__[attributes])
        else:
            equality = False

        return equality

    @classmethod
    def equal_dict_analysis_dataframes(cls, dict_analysis_dataframes1, dict_analysis_dataframes2):
        """Function to compare equality between dictionaries of analysis dataframes of MacularAnalysisDataframes.

        Parameters
        ----------
        dict_analysis_dataframes1 : dict of numpy.array or dict of dict
            First dict of analysis dataframes to compare.

        dict_analysis_dataframes2 : dict of numpy.array
            Second dict of analysis dataframes to compare.

        Returns
        ----------
        equality : Bool
            Returns True if both dict of analysis dataframes are equal and False otherwise.
        """
        equality = True

        if dict_analysis_dataframes1.keys() == dict_analysis_dataframes2.keys():
            # Equality between all the arrays of both dictionaries.
            for dimension in dict_analysis_dataframes1:
                if dimension in ["Conditions", "MetaConditions"]:
                    equality = equality & dict_analysis_dataframes1[dimension].equals(
                        dict_analysis_dataframes2[dimension])
                else:
                    for condition in dict_analysis_dataframes1[dimension]:
                        equality = equality & dict_analysis_dataframes1[dimension][condition].equals(
                            dict_analysis_dataframes2[dimension][condition])
        else:
            equality = False

        return equality

    def copy(self, path_pyb=""):
        """Function used to copy a MacularAnalysisDataframes.

        The copy is performed deeply by also copying all the objects included in the MacularAnalysisDataframes. It is
        also possible to specify a new .pyb file path to be used in the copy of the MacularAnalysisDataframes.

        Parameters
        ----------
        path_pyb : str
            Path to file with .pyb extension where to save the MacularAnalysisDataframes object in a binary file.

        Returns
        ----------
        macular_analysis_dataframe_copy : MacularAnalysisDataframes
            Returns the copy of the current MacularAnalysisDataframes.
        """
        macular_analysis_dataframe_copy = copy.deepcopy(self)

        if path_pyb:
            macular_analysis_dataframe_copy._dict_paths_pyb["self"] = path_pyb

        return macular_analysis_dataframe_copy

    @classmethod
    def load(cls, path_pyb):
        """Class method that allows importing a MacularAnalysisDataframes object from a pyb file containing it in
        binary format.

        Parameters
        ----------
        path_pyb : str
            Path to file with .pyb extension where a MacularAnalysisDataframes object is saved in binary.

            The path can be absolute or relative.
        """
        print("FILE LOADING...", end="")
        with open(path_pyb, "rb") as pyb_file:
            macular_analysis_dataframe = pickle.load(pyb_file)
        print("LOADED!")

        return macular_analysis_dataframe

    def save(self):
        """Saving the MacularAnalysisDataframes in a pyb (python binary) file whose path and name correspond to that
        present in the attribute of the analysis dictionary.
        """
        with open(f"{self.dict_paths_pyb['self']}", "wb") as pyb_file:
            pickle.dump(self, pyb_file)

    def initialize_macular_analysis_dataframes(self, multi_macular_dict_array, multiple_dicts_analysis):
        """Function to initialise a MacularAnalysisDataframes.

        First, the three parameter dictionaries (simulation, processing and analysis) are cleaned to remove empty fields
        before being stored as attributes. Next, the dictionaries from the multiple Macular dict array are constructed.
        The first dictionary stores the paths to the pyb files for each MacularDictArray included, while the second
        gathers and hierarchises all possible values for the first two hierarchical levels of the
        MacularAnalysisDataframes. The process continues by creating the attribute containing the regular expression
        compiled to process the condition names. The next step is to initialise each of the analysis dataframes using
        their respective indexes. In the case of the condition dataframe, the index can be ordered as desired by the
        user. Finally, a dictionary is created that brings together all the indexes present in the dataset of multiple
        MacularDictArrays.

        Parameters
        ----------
        multi_macular_dict_array : dict of MacularDictArray
            Dictionary associating specific conditions with different MacularDictArray.

        multiple_dicts_analysis : dict of dict
            Dictionaries containing all analyses or meta-analyses to be performed for each dimension of the
            MacularAnalysisDataframes, in a condensed format.

        Returns
        ----------
        dict_index : dict of dict
            Dictionary of all indexes present in the multiple macular dict array used in the current
            MacularAnalysisDataframes.
        """
        # Create the multiple_dicts_analysis attributes.
        self._multiple_dicts_analysis = multiple_dicts_analysis

        # Create and clean the multiple_dicts_simulations attributes.
        self._multiple_dicts_simulations = self.cleaning_multiple_dicts_features(
            {condition: multi_macular_dict_array[condition].dict_simulation for condition in multi_macular_dict_array})

        # Create and clean the multiple_dicts_preprocessings attributes.
        self._multiple_dicts_preprocessings = self.cleaning_multiple_dicts_features(
            {condition: multi_macular_dict_array[condition].dict_preprocessing for condition in
             multi_macular_dict_array})

        # Added the pyb paths for each condition of the multiple MacularDictArray to the dict_paths_pyb attribute.
        for condition in multi_macular_dict_array:
            self._dict_paths_pyb["MacularDictArrays"][condition] = multi_macular_dict_array[condition].path_pyb

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

        return dict_index


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
                    x_index, name_dataframe) for condition in self.dict_paths_pyb["MacularDictArrays"].keys()}
            # Create x-axis dataframe.
            elif name_dataframe == "Y":
                self.dict_analysis_dataframes[name_dataframe] = {condition: self.initialize_analysis_dataframe(
                    y_index, name_dataframe) for condition in self.dict_paths_pyb["MacularDictArrays"].keys()}
            # Create t-axis dataframe.
            elif name_dataframe == "Time":
                self.dict_analysis_dataframes[name_dataframe] = {condition: self.initialize_analysis_dataframe(
                    t_index, name_dataframe) for condition in self.dict_paths_pyb["MacularDictArrays"].keys()}
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
        sorted_conditions = list(self.dict_paths_pyb["MacularDictArrays"].keys())
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
        sorted_conditions = list(self.dict_paths_pyb["MacularDictArrays"].keys())

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
        all_conditions = ":".join(sorted([condition for condition in self.dict_paths_pyb["MacularDictArrays"]]))

        # Create dictionary associating each multiple MacularDictArray conditions with their "all_conditions".
        all_measurements = {
            condition: ":".join(sorted([measure for measure in multi_macular_dict_array[condition].data]))
            for condition in self.dict_paths_pyb["MacularDictArrays"]}

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
        multiple_dicts_analysis : dict of dict
            Dictionaries containing all analyses or meta-analyses to be performed for each dimension of the
            MacularAnalysisDataframes, in a condensed format.

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
            "peak_amplitude": self.peak_amplitude_analyzing,
            "initial_amplitude": self.initial_amplitude_analyzing,
            "spatial_mean": self.spatial_mean_analyzing
        }

        # Performs all spatial analyses listed in the current analysis dictionary.
        for analysis in self.multiple_dicts_analysis[dimension]:
            if analysis in available_spatial_analyses_dict:
                available_spatial_analyses_dict[analysis](self, multi_macular_dict_array, dimension, analysis)

    # def make_temporal_dataframes_analysis(self, multi_macular_dict_array):
    #     for analysis in self.multiple_dicts_analysis["Time"]:
    #         pass

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
            "peak_amplitude": self.peak_amplitude_analyzing,
            "initial_amplitude": self.initial_amplitude_analyzing
        }

        # Performs all conditions analyses listed in the current analysis dictionary.
        for analysis in self.multiple_dicts_analysis[dimension]:
            if analysis in available_spatial_analyses_dict:
                available_spatial_analyses_dict[analysis](self, multi_macular_dict_array, dimension, analysis)

    def make_meta_analysis_dataframes_analysis(self, dict_index):
        """Function used to perform all MacularAnalysisDataframe meta-analyses.

        The names of all meta-analyses type in the multiple analysis dictionaries are scanned and identified in a
        specific order. For each of them, a conditional block allows the corresponding analysis function to be executed.
        All these functions take as inputs the current MacularAnalysisDataframes, the dimension, and the current
        analysis.
        """
        # Dictionary containing all meta-analyses type currently implemented.
        available_spatial_analyses_dict = {
            "peak_speed": self.peak_speed_analyzing,
            "stationary_peak_delay": self.stationary_peak_delay_analyzing,
            "anticipation_fit": self.anticipation_fit_analyzing,
            "minimal_latency": self.minimal_latency_analyzing,
            "linear_fit": self.linear_fit_analyzing,
            "normalization": self.normalization_analyzing,
            "subtraction": self.subtraction_analyzing
        }

        # Order in which meta-analyses are performed.
        available_spatial_analyses_dict_order = ("peak_speed", "stationary_peak_delay", "anticipation_fit",
                                                 "minimal_latency", "linear_fit", "normalization", "subtraction")

        # Performs all meta-analyses type listed in the current analysis dictionary.
        for meta_analysis_type in available_spatial_analyses_dict_order:
            if meta_analysis_type in self.multiple_dicts_analysis["MetaAnalysis"]:
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
    def common_analysis_group_parser(list_grouped_levels, list_analysis_coordinates=None, n=0):
        """Recursive function for creating a unique association generator for analysis coordinates. hierarchical levels.

        The goal is to generate all possible combinations of analysis coordinates from all the different names at each
        of the five hierarchical levels. These names are contained in character strings and separated by ‘:’ symbols.
        The analysis coordinates are tuples formed from the 5 hierarchical levels classified in their hierarchical
        order: (dimension, condition, measurement, analysis, flag).

        Parameters
        ----------
        list_grouped_levels : list of str
            List containing different hierarchical level names separated by ‘:’. For example: ‘X:Y’ or
            ‘barSpeed30dps:barSpeed27dps’. The number of hierarchical levels contained in this list will define the
            depth of recursion.

        list_analysis_coordinates : list of str
            List containing all the hierarchical level to compose one analysis coordinates. This list is incremented at
            each hierarchical level by adding the new hierarchical level. It's then reset between each analysis
            coordinates. The list is transformed into a tuple once it is complete.

        n : int
            Hierarchical level counter in the analysis dictionary that increases with recursion. It is always equal to
            the length of the grouped levels list.
        """
        # Loop on the current hierarchical level of the current analysis.
        for level in list_grouped_levels[n].split(":"):
            # Initialisation of the list of analysis coordinates when at level 0.
            if not n:
                list_analysis_coordinates = []
            # Increment the list of analysis coordinates with the current level.
            new_list_current_analysis_levels = list_analysis_coordinates + [level]

            # Call the recursive function to go down one level if you are not at the last level.
            if n < len(list_grouped_levels) - 1:
                yield from MacularAnalysisDataframes.common_analysis_group_parser(list_grouped_levels,
                                                                                  new_list_current_analysis_levels,
                                                                                  n + 1)
            else:
                # Returns the analysis coordinates tuple to the generator once the last level has been reached.
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
            Dictionary of parameters to be used for peak amplitude analysis. It must contain only the x and/or y
            position to be analysed.

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

    @staticmethod
    @analysis
    def initial_amplitude_analyzing(data, index, parameters_analysis_dict):
        """Function that analyses initial amplitude based on a single spatial or conditions dimension.

        The initial amplitude is calculated in the 3D array and the index of a measurement of a condition. It
        corresponds to the time section at position 0 on the time axis. It is obtained in the form of a 2D array from
        which only the desired X or Y positions can be taken. This can be a 1D array with a single spatial dimension or
        a single value at a specific position.

        Parameters
        ----------
        data : np.ndarray
            3D array containing the values of a measurement for a given condition.

        index : dict of np.ndarray
            Dictionary containing all the indexes of a MacularDictArray in the form of a 1D array.

        parameters_analysis_dict : dict
            Dictionary of parameters to be used for initial amplitude analysis. It must contain only the x and/or y
            position to be analysed.

        Returns
        ----------
        initial_amplitude_computing : np.ndarray or float
            1D array of amplitude along a single spatial axis or value of peak amplitude at a specific spatial position.
        """
        # Calculation of the 2D array of amplitude.
        amplitude_2d_array = SpatialAnalyser.initial_amplitude_computing(data)

        # Extracting a single spatial dimension from the amplitude array.
        if "x" in parameters_analysis_dict and "y" not in parameters_analysis_dict:
            initial_amplitude = amplitude_2d_array[:, parameters_analysis_dict["x"]]
        elif "x" not in parameters_analysis_dict and "y" in parameters_analysis_dict:
            initial_amplitude = amplitude_2d_array[parameters_analysis_dict["y"], :]

        # Extracting a single spatial position from the amplitude array.
        elif "x" in parameters_analysis_dict and "y" in parameters_analysis_dict:
            initial_amplitude = amplitude_2d_array[parameters_analysis_dict["y"], parameters_analysis_dict["x"]]

        return initial_amplitude

    @staticmethod
    @analysis
    def spatial_mean_analyzing(data, index, parameters_analysis_dict):
        """Function that calculates and represents the average of a measurement along one spatial axis.

        The input data can be two-dimensional or three-dimensional. Both cases are handled to produce either a single
        average from the other axis or two averages if there are two axes.

        Parameters
        ----------
        data : np.ndarray
            3D or 2D array containing the values of a measurement for a given condition.

        index : dict of np.ndarray
            Dictionary containing all the indexes of a MacularDictArray in the form of a 1D array.

        parameters_analysis_dict : dict
            Dictionary of parameters to be used for initial amplitude analysis. It only must contain the index of the
            axis along which the average is to be represented.

        Returns
        ----------
        spatial_mean_array : np.ndarray
            1D array of mean along a single spatial axis.
        """
        # Calculation of the 1D array of the spatial mean.
        spatial_mean_array = SpatialAnalyser.spatial_mean_computing(data, parameters_analysis_dict["axis"])

        return spatial_mean_array

    @staticmethod
    def meta_analysis(meta_analysis_function):
        """Decorator for functions used to perform a specific meta-analysis of a multiple analysis dictionary.

        Parameters
        ----------
        meta_analysis_function : function
            Analysis function to apply to calculate the current meta-analysis.
            Analysis function to apply to calculate the current meta-analysis.

            This meta-analysis function changes between each meta-analysis, so they must all have the same three input
            arguments: the data, the index and the analysis parameters. However, the data may be in different forms
            depending on the function of the meta-analysis.
        """

        @wraps(meta_analysis_function)
        def modified_meta_analysis_function(macular_analysis_dataframes, meta_analysis_type, dict_index):
            """Function applied within the decorator, prior to the meta-analysis function, pour parser chaque groupes de
            méta-analyses communes puis en extraire chaque méta-analyses pour les effectuer.

            The meta-analysis of a MacularAnalysisDataframes is performed by conducting an ordered list of specific
            meta-analyses. Depending on the type of meta-analysis being performed, the name of the meta-analysis and the
            associated function are modified. Each specific meta-analysis can in turn be subdivided into a list of
            groups of common meta-analyses.

            Common meta-analysis groups are sets of analyses from different conditions, dimensions, and measurements
            that share the same meta-analysis treatment. A meta-analysis group therefore contains a succession of
            iterations of equivalent meta-analyses. These groups are presented in the form of dictionaries of
            dictionaries. The first key in this dictionary contains the dictionary of all constant parameters of the
            meta-analysis in question. Each meta-analysis is associated with different mandatory or optional parameters.
            All other keys represent the arguments and outputs of the meta-analysis.

            The arguments of a meta-analysis are necessary for calculating the meta-analysis, while its outputs
            represent the different output paths of the meta-analyses. Each type of meta-analysis requires a specific
            number of arguments and outputs with specific names. In the case of outputs, some of them may be optional
            depending on the implementation of the meta-analysis. Some meta-analyses also define outputs whose
            dimension, conditions and measurement coordinates are already partially fixed. In this case, the user only
            needs to enter the name of the output. This name must be added with its corresponding key in the parameter
            dictionary. Finally, some meta-analyses use an undefined and potentially unlimited number of outputs for a
            single meta-analysis. All of these outputs are placed in one or more ‘unlimited’ outputs, which are
            characterised by a succession of output names separated by ‘;’ symbols.

            The argument and output keys are each associated with a dictionary representing a group of common analyses.
            A group of common analyses is a set of different measurements and conditions sharing the same analysis
            process within different dimensions. This dictionary consists of a key representing each of the hierarchical
            levels of the MacularAnalysisDataframes and allowing the identification of a position within the latter.
            This is the position of the MacularAnalaysisDataframes from which to extract the analysis to be used as an
            argument or the meta-analysis intended to receive an output.

            Each position in the MacularAnalysisDataframes is defined by a set of coordinates corresponding to the
            different hierarchical levels associated with an analysis. The first is the dimension of the dataframe (X,
            Y, Time, Conditions, MetaConditions). The second is the condition of the Macular simulation performed. The
            next is the measurement of the Macular simulation data retrieved. Finally, there is the type of analysis to
            be performed. There is also a flag that allows you to differentiate between two identical analyses by adding
            a suffix to the analysis name. This suffix only works for arguments, not outputs.

            Initially, the common analysis group dictionary is presented in a condensed form. Each hierarchical level
            key is associated with a character string containing all the names of the hierarchical levels represented in
            the common analysis group. Each name is separated by the symbol ‘:’. It is also possible to use the aliases
            ‘all_conditions’, “all_measurements” and ‘all_dimensions’ when you want all the existing names of a
            hierarchical level without having to write everything. Once expanded, the common analysis groups will be
            defined by the combination of all the names of the hierarchical levels they contain. Any duplicate
            hierarchical level names will be removed.

            Dictionaries of common meta-analysis groups must contain dictionaries of common analysis groups that are
            equivalent in terms of arguments for this to work. This means that the dictionaries must have the same
            number of dimensions, conditions, measures and analyses. The only exception is the use of a common analysis
            group dictionary representing only a single analysis, which is in this case repeated as many times as the
            number of analyses in the argument containing the most. Dictionaries of common analysis groups must also not
            contain duplicates among the names at these levels.

            This function is called for each type of meta-analysis present in the multiple analysis dictionary. It
            begins by parsing the list of dictionaries of common meta-analysis groups to decompress it. Each of the
            dictionaries of common analysis groups associated with the arguments and outputs are decompressed. This
            transformation allows the list of coordinates (hierarchical levels ) of the analysis positions of the
            MacularAnalysisDataframes whose argument or output will take the successive values to be extracted. These
            coordinates are the hierarchical levels of the analyses or meta-analyses.

            A meta-analysis is defined here as the set of analysis coordinates located at the same index of the lists
            of arguments and outputs. For this reason, it is important that the lists of analysis coordinates for each
            argument and output are equal. For this to happen, all dictionaries of common analysis groups must be
            equivalent. This means that the dictionaries must have the same number of dimensions, conditions, measures
            and unique analyses. The only exception is the use of a common analysis group dictionary representing only
            a single meta-analysis, which in this case is repeated as many times as the number of meta-analyses of the
            argument containing the most.

            Once the dictionaries of common meta-analysis groups have been parsed, the function retrieves each
            meta-analysis one by one by simultaneously browsing all the lists of analysis coordinates of the arguments
            and outputs. A dictionary is created for each meta-analysis. The analysis coordinate tuples of each output
            are modified to create a dictionary with each hierarchical level. In the case of unlimited outputs that must
            be associated with several names, each name is retrieved by splitting at the ‘;’ level The meta-analysis
            dictionary thus obtained is provided as input to the meta-analysis function in progress.

            Note : In the context of meta-analysis outputs and some of their arguments derived from meta-analyses, the
            hierarchical level ‘measurement’ of dictionaries of common analysis groups is optional. This choice is
            motivated by the fact that a meta-analysis may depend on several measurements at the same time. However, it
            is still possible to provide this information for the sake of traceability of the measurements used to
            calculate each meta-analysis. Furthermore, any absence of measurements will be considered as a measurement
            in itself and will differentiate between two analyses.

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
        characterised by the presence of a list of analysis coordinates for each analyses containing in meta-analyses
        arguments: its dimension, condition, measure, type of analysis and any associated flag.

        To perform this process, dictionaries of common analysis groups must not contain duplicate names among their
        levels. Parsing will replace each of meta-analysis arguments dictionaries with the list of analysis coordinates
        that define the different analyses to be extracted from the MacularAnalysisDataframes. The dictionary also
        contains a ‘params’ key associated with external parameters to be used for meta-analysis. This dictionary is not
        modified during parsing.

        In the case where the lists of analysis coordinates for each argument are of different sizes, en error will be
        raised except a list of size 1, which will adjust to the maximum size observed.

        Example :
        common_meta_analysis_group_parser(
        {"value_to_normalize": {"dimensions": "X:Y", "conditions": "barSpeed27ps:barSpeed30ps", "measurements": "VSDI",
                        "analyses": "peak_amplitude", "flag": "internal_flag"},
        "baseline": {"dimensions": "Conditions", "conditions": "barSpeed27ps:barSpeed30ps", "measurements": "VSDI",
                        "analyses": "peak_amplitude", "flag": "internal_flag"},
        "output": {"dimensions": "X:Y", "conditions": "barSpeed27ps:barSpeed30ps", "measurements": "VSDI",
                        "analyses": "peak_amplitude"},
        "params": {"factor": 8})

        > {"value_to_normalize": [("X", "barSpeed27ps", "VSDI", "peak_amplitude", "internal_flag"),
                        ("X", "barSpeed27ps", "VSDI", "peak_amplitude", "internal_flag")
                        ("Y", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag"),
                        ("Y", "barSpeed30dps", "VSDI", "peak_amplitude", "internal_flag")],
        "baseline": [("Conditions", "barSpeed27ps", "VSDI", "peak_amplitude", "internal_flag"),
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
            Dictionary of a common meta-analysis group with lists of tuples of coordinates characterising each analysis
            associated with each of the meta-analysis arguments.
        """
        # Verification that there are no duplicates in the dictionaries of common analysis groups.
        MacularAnalysisDataframes.check_common_analysis_group_repeats(common_meta_analysis_group_dictionary)

        # Initialisation of the dictionary containing the lists of coordinates of the common meta-analysis group.
        parsed_dictionary = {}

        # Initialisation of the maximum length of the analysis coordinates lists for common analysis groups.
        common_meta_analysis_group_max_length = 1
        # Loop through all argument names for the meta-analysis function in its dictionary.
        for meta_analysis_argument in common_meta_analysis_group_dictionary:
            # Copy the parameter dictionary to the parsed dictionary of common meta-analysis group.
            if meta_analysis_argument == "params":
                parsed_dictionary["params"] = common_meta_analysis_group_dictionary[meta_analysis_argument].copy()
            # Initialisation of the list of coordinates of one common analysis group for one meta-analysis argument.
            else:
                argument_common_group_analysis = common_meta_analysis_group_dictionary[meta_analysis_argument]
                if "output" in meta_analysis_argument:
                    # Creation of analysis coordinates generator for the common analysis group of the current output.
                    argument_common_analysis_group_generator = MacularAnalysisDataframes.common_analysis_group_parser(
                        [argument_common_group_analysis["dimensions"], argument_common_group_analysis["conditions"],
                         argument_common_group_analysis["measurements"], argument_common_group_analysis["analyses"]])
                else:
                    # Creation of analysis coordinates generator for the common analysis group of the current argument.
                    argument_common_analysis_group_generator = MacularAnalysisDataframes.common_analysis_group_parser(
                        [argument_common_group_analysis["dimensions"], argument_common_group_analysis["conditions"],
                         argument_common_group_analysis["measurements"], argument_common_group_analysis["analyses"],
                         argument_common_group_analysis["flag"]])
                # Transformation of coordinates generator into a list of coordinates of the common analysis group.
                parsed_dictionary[meta_analysis_argument] = [analysis_coordinates for analysis_coordinates in
                                                             argument_common_analysis_group_generator]
                # Calculate the maximum length of the analysis coordinates lists for each argument.
                if len(parsed_dictionary[meta_analysis_argument]) > common_meta_analysis_group_max_length:
                    common_meta_analysis_group_max_length = len(parsed_dictionary[meta_analysis_argument])

        # Adjusting the length of analysis coordinates lists of common analysis group that were too small.
        for meta_analysis_argument in parsed_dictionary:
            if meta_analysis_argument != "params":
                parsed_dictionary[meta_analysis_argument] = (MacularAnalysisDataframes.
                check_common_analysis_group_coordinates_size(
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
    def check_common_analysis_group_coordinates_size(common_analysis_group_coordinates_list, expected_length):
        """Function for checking the size of the list of analysis coordinates for a group of common analyses so that it
        corresponds to an expected length.

        If the length does not match, there are two possibilities. If the list is of size 1, it will be repeated as many
        times as the expected length. However, if the size is greater than 1, an error will be raised.

        Example : check_common_analysis_group_coordinates_size([(element1, element2, element3)], 2)
        > [(element1, element2, element3), (element1, element2, element3)]

        Parameters
        ----------
        common_analysis_group_coordinates_list : list of tuples
            List containing analysis coordinates from a common analysis group.

        expected_length : int
            Expected length that you want to reach with the list of analysis coordinates in the common analysis group.

        Returns
        ----------
        checked_coordinates_list : list of tuples
            List analysis coordinates tuples of a common analysis group checked in size.

        Raises
        ----------
        ValueError
            The length of the analysis coordinates list is smaller than expected and is also greater than 1.
        """
        coordinates_list_length = len(common_analysis_group_coordinates_list)
        # Cases where the length of the analysis coordinates list is smaller than expected.
        if coordinates_list_length < expected_length and coordinates_list_length == 1:
            checked_coordinates_list = []
            # Correct the length by repeating each element in the list in an equivalent way.
            for coordinates in common_analysis_group_coordinates_list:
                checked_coordinates_list += [coordinates] * (expected_length // coordinates_list_length)
        # Case where the length of the analysis coordinates list is equal to the expected size.
        elif coordinates_list_length == expected_length:
            checked_coordinates_list = common_analysis_group_coordinates_list
        else:
            raise ValueError(f"The length of the common analysis group coordinates list does not match the maximum "
                             f"length {expected_length}")

        return checked_coordinates_list

    def make_common_group_meta_analysis(self, meta_analysis_function, common_meta_analysis_group_dictionary,
                                        meta_analysis, dict_index):
        """Function performing all meta-analyses present in a group of decondensed common meta-analyses.

        The decondensed group of common meta-analyses is structured as a dictionary associating the names of the
        meta-analysis arguments with lists of tuples of all analyses coordinates for which the argument will take the
        value.

        A meta-analysis groups together all the analyses located at the same index in all the lists of arguments of
        meta-analyses. Therefore, all the arguments in a dictionary of common meta-analyses have a list of the same
        length, each element of which represents the sequence of values that the argument will take during the common
        meta-analysis.

        The function iterates over the arguments (except ‘params’) of all meta-analyses defined in the common
        meta-analysis group. The analysis coordinates associated with all these arguments are stored as is in a first
        dictionary that is used after the loop to construct the names of all output of the meta-analysis. The analysis
        coordinates of the arguments are also used to extract the arrays of analyses they describe into a second
        dictionary. This dictionary also contains the analysis coordinates defined in the output. This dictionary is
        finally used in the execution of the current meta-analysis function.

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
            passed to the meta-analysis function. Each key is associated with the list of each analysis coordinates whose
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
        for analysis_coordinates_index in range(meta_analysis_arguments_length):
            # Loop over the arguments of the current meta-analysis function.
            for meta_analysis_argument in meta_analysis_arguments_list:
                # Store the analyses coordinates defining the current meta-analysis in the meta-analysis dictionary.
                current_meta_analysis_dictionary[meta_analysis_argument] = (
                    common_meta_analysis_group_dictionary)[meta_analysis_argument][analysis_coordinates_index]

            # Make a copy of the current meta analysis dictionary to avoid modification of it during the process.
            current_meta_analysis_dictionary_copy = current_meta_analysis_dictionary.copy()

            # Creation of a dictionary of names for each output argument of the current meta-analysis.
            current_meta_analysis_dictionary_copy.update(MacularAnalysisDataframes.make_meta_analysis_outputs(
                meta_analysis, current_meta_analysis_dictionary_copy, common_meta_analysis_group_dictionary["params"]))

            # Execution of the current meta-analysis function.
            meta_analysis_function(self, current_meta_analysis_dictionary_copy, dict_index,
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
            Meta-analysis dictionary linking the names of arguments in a meta-analysis with the names of the coordinates
            defining a given analysis (dimension, condition, measurements, analysis type, flag).
        """
        # Loop on meta-analysis arguments except output ones.
        for meta_analysis_argument in meta_analysis_dictionary.keys():
            if "output" not in meta_analysis_argument and "index" not in meta_analysis_argument:
                meta_analysis_dictionary[meta_analysis_argument] = (
                    MacularAnalysisDataframes.extract_one_analysis_array_from_dataframes(
                        macular_analysis_dataframes, meta_analysis_dictionary[meta_analysis_argument]))

    @staticmethod
    def extract_one_analysis_array_from_dataframes(macular_analysis_dataframes, analysis_coordinates):
        """Function to extract the value(s) located at a given analysis coordinate in the MacularAnalysisDataframes.

        Each analysis of a MacularAnalysisDataframes can be defined by an analysis coordinates with the five
        hierarchical levels of a MacularAnalysisDataframes. There is the dimension (‘X’, ‘Y’, “Conditions”), the
        condition (‘barSpeed30dps’, ‘ampGang30Hz’), the measurement (‘VSDI’, ‘FiringRate_GanglionGainControl’), the
        type of analysis (“latency”, ‘peak_amplitude’) and the name of the flag, if there is one. These levels must
        therefore be used to locate and extract an analysis.

        In some cases, an analysis to be extracted may be defined only by the dimension, condition and type of the
        analysis, but not by the measurement. This can happen for analyses from meta-analyses. In this case, the
        measurement value is set to an empty character string ‘’.

        In the case of the ‘Conditions’ dimension, there is a single dataframe containing the data, whereas in the
        other spatio-temporal dimensions there is one dataframe per condition. Therefore, two methods must be used to
        extract values from these two types of dataframes. In the case of conditions, it is possible to either extract
        a single value associated with a condition or get everything if the hierarchical level of the dimensions has
        been set with the term ‘overall’.

        Parameters
        ----------
        macular_analysis_dataframes : MacularAnalysisDataframes
            Macular Analyses Dataframes that the user wishes to use to extract a row from a given dataframe.

        analysis_coordinates : tuple
            Coordinates defining a given analysis (dimension, condition, measure, analysis type).

        Returns
        ----------
        analysis_array : int, float or np.ndarray
            Array of values or single value of the analysis to be extracted.
        """
        # Construction of the name of the analysis line to be extracted.
        if analysis_coordinates[2] == "":
            # Cases that only include the analysis type in the name of the analysis to be extracted.
            dataframe_row = f"{analysis_coordinates[3]}_{analysis_coordinates[4]}".strip("_")
        else:
            # Cases that include the analysis type and measurement in the name of the analysis to be extracted.
            dataframe_row = f"{analysis_coordinates[3]}_{analysis_coordinates[2]}_{analysis_coordinates[4]}".strip("_")

        # Cases of conditions dataframe.
        if analysis_coordinates[0] == "Conditions":
            # Case of every condition in condition dataframe.
            if analysis_coordinates[1] == "overall":
                analysis_array = macular_analysis_dataframes.dict_analysis_dataframes[analysis_coordinates[0]].loc[
                                 dataframe_row, :].values.astype(float)
            # Case of the single conditions in condition dataframe.
            else:
                analysis_array = macular_analysis_dataframes.dict_analysis_dataframes[analysis_coordinates[0]].loc[
                    dataframe_row, analysis_coordinates[1]]
        # Case of multiple spatio-temporal dataframes.
        else:
            analysis_array = macular_analysis_dataframes.dict_analysis_dataframes[analysis_coordinates[0]][
                                 analysis_coordinates[1]].loc[dataframe_row, :].values

        return analysis_array

    @staticmethod
    def make_meta_analysis_outputs(meta_analysis_name, meta_analysis_dictionary, parameters_meta_analysis_dict):
        """Function for formatting meta-analysis outputs names.

        The name of the meta-analysis is primarily retrieved from the name defined among the arguments of the
        meta-analysis function. All arguments containing the term ‘output’ will be used to retrieve as many names as
        will be defined in a dictionary. If this is not the case, the name defined in the meta-analysis parameter
        dictionary will be used. All parameters containing the term ‘output’ are retrieved again. Each output can be
        associated with multiple output names to be used simultaneously in one meta-analysis. Each of these names is
        separated by the symbol ‘;’ and are therefore divided in this function into a list for later use.

        If no output is present in the arguments or parameter dictionary, a default behaviour is performed. Outputs are
        created by formatting from measurements and analyses types of each argument of the meta-analysis. It is possible
        to slightly adapt this default case by adding a ‘flag’ parameter in the meta-analysis parameter dictionary.
        This ‘flag’ character string will be added as last suffix.

        Parameters
        ----------
        meta_analysis_name : str
            Name of the meta-analysis for which a format is needed.

        meta_analysis_dictionary : dict of tuples
            Meta-analysis dictionary linking the names of arguments in a meta-analysis with the coordinates defining a
            given analysis (dimension, condition, measurements, analysis type, flag).

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
                    "name": meta_analysis_dictionary[meta_analysis_arguments][3]
                }
                # Separation into a list of multiple outputs of a single argument from a meta-analysis.
                if ";" in meta_analysis_outputs_dict[meta_analysis_arguments]["name"]:
                    meta_analysis_outputs_dict[meta_analysis_arguments]["name"] = meta_analysis_outputs_dict[
                        meta_analysis_arguments]["name"].split(";")
                output = True

        # Cases where no output has been defined in the meta-analysis arguments.
        if not output:
            # Cases where one or more outputs have been defined in the meta-analysis settings.
            for params in parameters_meta_analysis_dict:
                if "output" in params:
                    meta_analysis_outputs_dict[params] = {"name": parameters_meta_analysis_dict[params]}
                    # Separation into a list of multiple outputs of a single argument from a meta-analysis.
                    if ";" in meta_analysis_outputs_dict[params]["name"]:
                        meta_analysis_outputs_dict[params]["name"] = meta_analysis_outputs_dict[params]["name"
                        ].split(";")
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

        Creating a new line requires the analysis coordinates corresponding to the names of each hierarchical level in
        the MacularDictDataframes. This allows to identify the position of the new line and its name.

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

        To work, this meta-analysis requires two arguments: the "value_to_normalize" and the "baseline", which must defined in
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
        # Convert all non-outputs meta-analysis arguments coordinates into the corresponding analysis array.
        MacularAnalysisDataframes.extract_all_analysis_array_from_dataframes(macular_analysis_dataframes,
                                                                             meta_analysis_dictionary)

        # Calculation of the division of the two analysis values and multiplication by the factor.
        normalized_values = MetaAnalyser.normalization_computing(meta_analysis_dictionary["value_to_normalize"],
                                                                 meta_analysis_dictionary["baseline"],
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

        To work, this meta-analysis requires 1 argument: the "time_to_peak" that needs to be fitted. This dictionary
        does not require an ‘output’ key to define the output to which the peak speed should be sent. Instead, the
        output is automatically set to the conditions dataframe and directly uses the conditions defined in the
        ‘time_to_peak’ analysis.

        The dictionary also contains the ‘params’ parameters, whose dictionary must contain the ‘index’ parameter,
        which corresponds to the name of the spatial index to be used (X or Y). The spatial index depends on the axis
        of movement. Two important parameters must also be specified for the fit, namely the number of points ‘n_points’
        for predicting the fit and the list “breaks” of the edges of the segments to be fitted. If the user do not want
        to define this list, the user can instead associate the term “auto” with the key ‘breaks’. Another key must be
        added to define the name of the output to be created in the condition dataframe. The first key, ‘output’,
        allows you to define a specific name, while the second alternative key, ‘flag’, allows you to use the default
        output name by simply adding a suffix.

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

            This dictionary must contain the spatial index name, the breaks and the number of points to be used for
            fitting.
        """
        # Store dimensions and conditions of output.
        meta_analysis_dictionary["output"]["dimension"] = "Conditions"
        meta_analysis_dictionary["output"]["condition"] = meta_analysis_dictionary["time_to_peak"][1]

        # Convert all non-outputs meta-analysis arguments coordinates into the corresponding analysis array.
        MacularAnalysisDataframes.extract_all_analysis_array_from_dataframes(macular_analysis_dataframes,
                                                                             meta_analysis_dictionary)

        # Calculation of the peak speed of the time to peak data array.
        peak_speed_fit = MetaAnalyser.linear_fit_computing(
            meta_analysis_dictionary["time_to_peak"], index[meta_analysis_dictionary["output"]["condition"]][
                parameters_meta_analysis_dict["index"]], 1, parameters_meta_analysis_dict["breaks"],
            parameters_meta_analysis_dict["n_points"])

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

        To work, this meta-analysis requires 1 argument: the "peak_delay" that needs to be averaged. This dictionary
        does not require an ‘output’ key to define the output to which the peak speed should be sent. Instead, the
        output is automatically set to the conditions dataframe and directly uses the conditions defined in the
        ‘peak_delay’ analysis.

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

        # Convert all non-outputs meta-analysis arguments coordinates into the corresponding analysis array.
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
        slopes and inflection points obtained. The third key is "breaks" that contains the list of the edges of the
        segments to be fitted. If you do not want to define this list, you can instead associate the term “auto” with
        the key ‘breaks’. The last parameter, ‘n_points’, is used to select the resolution of the fit. It is important
        to note that regardless of this resolution, the data predictions and fit indexes within a given dataframe will
        be binning to ensure the correct size.

        Please note that breaks made within the parameter dictionary must correspond in size to the dataframe in which
        they will be stored. It is therefore not possible at this time to perform fits on sub-parts of the dataframe.

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

            This dictionary must contain the index name, the number of segments, the breaks and the resolution to be
            used for fitting.
        """
        # Store dimensions and conditions of output.
        meta_analysis_dictionary["index"] = {"condition": meta_analysis_dictionary["data_to_fit"][1]}

        # Convert all non-outputs meta-analysis arguments coordinates into the corresponding analysis array.
        MacularAnalysisDataframes.extract_all_analysis_array_from_dataframes(macular_analysis_dataframes,
                                                                             meta_analysis_dictionary)

        # Getting the index to use for the fitting.
        current_index = index[meta_analysis_dictionary["index"]["condition"]][parameters_meta_analysis_dict["index"]]

        # Fit of the variable to be fitted, respecting the number of segments given in the parameters.
        linear_fit = MetaAnalyser.linear_fit_computing(current_index, meta_analysis_dictionary["data_to_fit"],
                                                       parameters_meta_analysis_dict["n_segments"],
                                                       parameters_meta_analysis_dict["breaks"],
                                                       parameters_meta_analysis_dict["n_points"])

        # Binning of prediction arrays from data and index arrays to obtain the size of the fitted arrays.
        linear_fit["index_prediction"], linear_fit["data_prediction"] = MetaAnalyser.statistic_binning(
            linear_fit["index_prediction"], linear_fit["data_prediction"], current_index.shape[0])

        # Manage unique outputs by transforming them into a dictionary of size 1.
        for arguments in meta_analysis_dictionary:
            if arguments not in ["output_index_prediction", "output_data_prediction", "index", "data_to_fit"]:
                if not isinstance(meta_analysis_dictionary[arguments]["name"], list):
                    meta_analysis_dictionary[arguments]["name"] = [meta_analysis_dictionary[arguments]["name"]]

        # Adds the output slopes value(s) to a new row in the output dataframe.
        if "output_slopes" in meta_analysis_dictionary.keys():
            for slope, output_name in zip(linear_fit["slopes"], meta_analysis_dictionary["output_slopes"]["name"]):
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
        as a function of the activation time. Therefore, in the event of a manual fit, it is important to provide the
        latency values.

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
        third key is "breaks" that contains the list of the edges of the segments to be fitted. If you do not want to
        define this list, you can instead associate the term “auto” with the key ‘breaks’. The last parameter,
        ‘n_points’, is used to select the resolution of the fit. It is important to note that regardless of this
        resolution, the data predictions and fit indexes within a given dataframe will be binning to ensure the correct
        size. Another key must be added to define the name of the output to be created in the condition dataframe. The
        first key, ‘output’, allows you to define a specific name, while the second alternative key, ‘flag’, allows you
        to use the default output name by simply adding a suffix.

        Please note that breaks made within the parameter dictionary must correspond in size to the dataframe in which
        they will be stored. It is therefore not possible at this time to perform fits on sub-parts of the dataframe.

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

            This dictionary must contain the spatial index name, the number of segments, the breaks and the resolution
            to be used for fitting.
        """
        # Store dimensions and conditions of output.
        meta_analysis_dictionary["output"] = {"dimension": "Conditions",
                                              "condition": meta_analysis_dictionary["activation_time"][1]}
        meta_analysis_dictionary["output_prediction"] = {"dimension": meta_analysis_dictionary["activation_time"][0],
                                                         "condition": meta_analysis_dictionary["activation_time"][1]}

        # Convert all non-outputs meta-analysis arguments coordinates into the corresponding analysis array.
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
                                                       parameters_meta_analysis_dict["breaks"],
                                                       parameters_meta_analysis_dict["n_points"])

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
    def minimal_latency_analyzing(macular_analysis_dataframes, meta_analysis_dictionary, index,
                                  parameters_meta_analysis_dict):
        """Function to calculate the minimal latency at which latency begins to saturate for a distance between cell and
        object motion origin that exceeds the anticipation range.

        The calculation is performed by obtaining the latency values and then extracting only the stationary portion.
        This stationary portion is located beyond the anticipation range value. To use this meta-analysis, you must
        first perform an ‘anticipation_fit_analyzing’ meta-analysis.

        To work, this meta-analysis requires 2 arguments: the "latency" that needs to be averaged and
        ‘anticipation_range’ which should be used to isolate the stationary portion. This dictionary does not require an
        ‘output’ key to define the output to which the minimal latency should be sent. Instead, the output is
        automatically set to the conditions dataframe and directly uses the conditions defined in the ‘latency’
        analysis.

        The dictionary also contains the ‘params’ parameters, whose dictionary must contain a key "index" to define
        which index to use. In addition, the dictionary needs also  the name of the output to be created in the
        condition dataframe. The first key, ‘output’, allows you to define a specific name, while the second alternative
        key, ‘flag’, allows you to use the default output name by simply adding a suffix.

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

            This dictionary must contain the spatial index name.
        """
        # Store dimensions and conditions of output.
        meta_analysis_dictionary["output"]["dimension"] = "Conditions"
        meta_analysis_dictionary["output"]["condition"] = meta_analysis_dictionary["latency"][1]

        # Convert all non-outputs meta-analysis arguments coordinates into the corresponding analysis array.
        MacularAnalysisDataframes.extract_all_analysis_array_from_dataframes(macular_analysis_dataframes,
                                                                             meta_analysis_dictionary)

        current_index = index[meta_analysis_dictionary["output"]["condition"]][parameters_meta_analysis_dict["index"]]
        stationary_latency = meta_analysis_dictionary["latency"][np.where(
            current_index > meta_analysis_dictionary["anticipation_range"])[0][0]:]

        # Calculation of the minimal latency
        minimal_latency_value = MetaAnalyser.mean_computing(stationary_latency)

        # Adds the output value(s) to a new row in the output dataframe.
        MacularAnalysisDataframes.add_array_line_to_dataframes(macular_analysis_dataframes,
                                                               meta_analysis_dictionary["output"]["dimension"],
                                                               meta_analysis_dictionary["output"]["condition"],
                                                               meta_analysis_dictionary["output"]["name"],
                                                               minimal_latency_value)

    @staticmethod
    @meta_analysis
    def subtraction_analyzing(macular_analysis_dataframes, meta_analysis_dictionary, index,
                              parameters_meta_analysis_dict):
        """Function that calculates a subtraction of one value by one or multiple values.

        To work, this meta-analysis requires two arguments: the "value_to_normalize" and the "baseline", which must defined in
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
        # Convert all non-outputs meta-analysis arguments coordinates into the corresponding analysis array.
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
