import numpy as np
import pwlf
from scipy.stats import binned_statistic


class MetaAnalyser:
    """Class grouping all functions intended to meta-analyse certain properties of a 3D spatio-temporal data array.

    Each of these functions is initially created to be used for meta-analysis within a MacularAnalysisDataframes.
    """

    @staticmethod
    def normalization_computing(value_to_normalize, baseline, factor):
        """Calculation of a normalization between two arrays, int or float with the possibility to multiply by a given
        factor.

        Parameters
        ----------
        value_to_normalize : np.ndarray, int or float
            Value to be normalized from a baseline.

        baseline : np.ndarray, int or float
            Baseline value to use for normalization.

        factor : float or int
            Multiplication factor to be used for normalization.

        Returns
        ----------
        normalization_values : np.ndarray, int or float
            Value(s) obtained from the normalization.
        """
        normalization_values = ((value_to_normalize - baseline) / baseline) * factor

        return normalization_values

    @staticmethod
    def linear_fit_computing(index_array, data_array, n_segments, breaks, n_points):
        """Calculation of the linear fit of a linear segment or several consecutive linear segments.

        It is possible to perform the fit manually by specifying the limits of the segments to be fitted. At the same
        time, an automatic fit can also be performed without specifying these limits. In the case of manual fitting,
        the interval defined by the breaks must have two ends that are identical to those of the index so as not to
        change the size of the latter. It is currently not possible to perform a fit on a sub-section of the index.

        Parameters
        ----------
        index_array : np.ndarray
            Index of data for the linear segment(s).

        data_array : np.ndarray
            Data for a quantity consisting of one or more linear segments.

        n_segments : int
            Number of different linear segments to search for in the data.

        breaks : np.ndarray or list or str
            Values of all indices that serve as index edges between each linear segment of the fit. It is also possible
            to enter the term ‘auto’ if you do not want to define the breaks manually.

        n_points : int
            Number of points used to predict the fit.

        Returns
        ----------
        fitting_dict : np.ndarray, int or float
            Dictionary containing all the properties of the fit.
        """
        # Performing an automatic piecewise linear fit with automatically defined breaks.
        if breaks == "auto":
            fitting_dict, breaks = MetaAnalyser.automatically_defined_linear_fit_computing(
                index_array, data_array, n_segments, n_points)
        # Performing an automatic piecewise linear fit with manually defined breaks.
        else:
            fitting_dict = MetaAnalyser.manually_defined_linear_fit_computing(index_array, data_array, n_segments,
                                                                              breaks, n_points)
            breaks = np.array(list(breaks))

        # Index intercepts computing.
        fitting_dict["index_intercepts"] = (fitting_dict["data_intercepts"] / fitting_dict["slopes"]).round(4)

        if n_segments > 1:
            # Compute inflection points.
            inflection_points = [float(fitting_dict["data_prediction"][np.where(
                fitting_dict["index_prediction"] >= breaks[i + 1])[0][0]].round(3)) for i in range(n_segments - 1)]

            # Updated the fit dictionary with the remaining properties.
            fitting_dict["inflection_points_index"] = breaks[1:-1].round(3).tolist()
            fitting_dict["inflection_points_data"] = inflection_points

        return fitting_dict

    @staticmethod
    def automatically_defined_linear_fit_computing(index_array, data_array, n_segments, n_points):
        """Computing a single or multiple consecutive linear fit using breaks defined automatically.

        Parameters
        ----------
        index_array : np.ndarray
            Index of data for the linear segment(s).

        data_array : np.ndarray
            Data for a quantity consisting of one or more linear segments.

        n_segments : int
            Number of different linear segments to search for in the data.

        n_points : int
            Number of points used to predict the fit.

        Returns
        ----------
        fitting_dict : np.ndarray, int or float
            Dictionary containing all the properties of the fit.

        breaks : np.ndarray
            Values of all indices that serve as edges between each linear segment of the fit.
        """
        # Piecewise linear fit of the data.
        linear_fit = pwlf.PiecewiseLinFit(index_array, data_array)

        # Fitting properties computing.
        breaks = linear_fit.fit(n_segments).round(3)
        slopes = linear_fit.calc_slopes().round(4).tolist()
        data_intercepts = linear_fit.intercepts.round(4)

        # Data prediction.
        index_prediction = np.linspace(index_array.min(), index_array.max(), n_points).round(3)
        data_prediction = linear_fit.predict(index_prediction).round(3)

        # Fitting dictionary initialisation.
        fitting_dict = {"slopes": slopes, "index_prediction": index_prediction, "data_prediction": data_prediction,
                        "data_intercepts": data_intercepts}

        return fitting_dict, breaks

    @staticmethod
    def manually_defined_linear_fit_computing(index_array, data_array, n_segments, breaks, n_points):
        """Computing a single or multiple consecutive linear fit using breaks defined manually by the user.

        This function performs an individual linear fit for the number of segments and interval limits specified by the
        user. The intervals for each segment are also defined by the user. It is therefore possible that the fit will
        only be performed on a subset of the index and data. If the interval limits do not correspond exactly to a
        specific position in the index, the closest position is used. For each fit, the sub-parts of the data and index
        used are adjusted. The same applies to the prediction index, which is calculated in its entirety at the
        beginning of the function. All fit properties are finally stored in lists or arrays. All of this is then added
        to a dictionary.

        Please note that the interval defined by the breaks must have two ends that are identical to those of the index
        so as not to change the size of the latter. It is currently not possible to perform a fit on a sub-section of
        the index.

        Note :
        We did not use the fit_by_breaks function of pwlf because it gave us incorrect results in the simple test cases.

        Parameters
        ----------
        index_array : np.ndarray
            Index of data for the linear segment(s).

        data_array : np.ndarray
            Data for a quantity consisting of one or more linear segments.

        n_segments : int
            Number of different consecutive linear segments to search for in the data.

        breaks : np.ndarray or list
            Values of all indices that serve as index edges between each linear segment of the fit.

        n_points : int
            Total number of points used to predict the fit.

        Returns
        ----------
        fitting_dict : np.ndarray, int or float
            Dictionary containing all the properties of the fit.
        """
        # Properties initialization.
        breaks = np.array(list(breaks))
        slopes = []
        index_prediction = np.linspace(min(breaks), max(breaks), n_points).round(3)
        data_prediction = np.array([])
        data_intercepts = np.array([])

        # Calculate the list of positions of the index values closest to the breaks.
        index_breaks = [np.argmin(np.abs(index_array - break_value)) for break_value in breaks]

        # Loop on segments.
        for i_segment in range(n_segments):
            # Gets the index that best matches the breaks in the current segment.
            current_segment_index_array = index_array[index_breaks[i_segment]:index_breaks[i_segment+1]+1]
            # Gets the data that best matches the breaks in the current segment.
            current_segment_data_array = data_array[index_breaks[i_segment]:index_breaks[i_segment+1]+1]

            # Piecewise linear fit of the data.
            linear_fit = pwlf.PiecewiseLinFit(current_segment_index_array, current_segment_data_array)
            linear_fit.fit(1)

            # Fitting properties computing and incrementing the corresponding variables.
            slopes += linear_fit.calc_slopes().round(4).tolist()
            data_intercepts = np.concatenate((data_intercepts, linear_fit.intercepts.round(4)), axis=-1)

            # Extract the prediction index interval delimited by the current segment breaks.
            if i_segment < n_segments - 1:
                # General case that does not include the highest-value break in the current segment index.
                current_index_prediction = index_prediction[np.where((index_prediction < breaks[i_segment+1]) &
                                                                     (index_prediction >= breaks[i_segment]))]
            else:
                # Case of the last segment including the break with the highest value in its index.
                current_index_prediction = index_prediction[np.where((index_prediction <= breaks[i_segment+1]) &
                                                                     (index_prediction >= breaks[i_segment]))]

            # Data prediction for the current segment and incrementation in the data prediction array.
            data_prediction = np.concatenate((data_prediction, linear_fit.predict(
                current_index_prediction).round(3)), axis=-1)

        # Filling fitting dictionary.
        fitting_dict = {"slopes": slopes, "index_prediction": index_prediction, "data_prediction": data_prediction,
                        "data_intercepts": data_intercepts}

        return fitting_dict

    @staticmethod
    def mean_computing(data_array):
        """Calculating the average value of a set of data.

        Parameters
        ----------
        data_array : np.ndarray
            Data to average.

        Returns
        ----------
        mean_value : int or float
            Mean value of the data.
        """
        mean_value = data_array.mean().round(3)

        return mean_value

    @staticmethod
    def statistic_binning(index_array, data_array, n_bin):
        """Function to bin two arrays index and data to reduce them from their initial size to a new desired size.

        Here we use the binned_statistic function from the scipy stats module because it allows us to bin a data set
        while preserving the dynamics of the data. It also allows us to perform binning with a coefficient that is a
        float.

        Parameters
        ----------
        data_array : np.ndarray
            Data to bin.

        index_array : np.ndarray
            Index to bin.

        n_bin : np.ndarray
            Desired size for both data and index arrays.

        Returns
        ----------
        binned_data_array : np.ndarray
            Binned data of size n_bin.

        binned_index_array : np.ndarray
            Binned index of size n_bin.
        """
        # Define the edges of the bins to divide n_bin times.
        bins = np.linspace(index_array[0], index_array[-1], n_bin + 1)

        # Binning of the data array.
        binned_data_array = binned_statistic(index_array, data_array, statistic='mean', bins=bins)[0].round(3)

        # Binning the index array using the centre of each bin interval.
        binned_index_array = np.linspace((bins[1] + bins[0]) / 2, (bins[-1] + bins[-2]) / 2, n_bin).round(3)

        return binned_index_array, binned_data_array

    @staticmethod
    def subtraction_computing(initial_value, values_subtracted):
        """Calculation of a normalization between two arrays, int or float with the possibility to multiply by a given
        factor.

        Parameters
        ----------
        initial_value : np.ndarray, int or float
            Initial value from which to subtract.

        values_subtracted : list of np.ndarray or list of float or list of int
            Values to be subtracted from the initial value.

        Returns
        ----------
        subtraction_values : np.ndarray, int or float
            Value(s) obtained from the normalization.
        """
        # Initialisation of the initial value.
        subtraction_values = initial_value

        # List of values to be subtracted from the initial value.
        for value_subtracted in values_subtracted:
            subtraction_values -= value_subtracted

        subtraction_values = np.round(subtraction_values, 3)

        return subtraction_values
