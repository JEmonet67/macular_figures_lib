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
    def linear_fit_computing(index_array, data_array, n_segments, n_points=100):
        """Calculation of the linear fit of a linear segment or several successive linear segments.

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
            Slope value obtained from the fitting.
        """
        # Piecewise linear fit of latency time.
        linear_fit = pwlf.PiecewiseLinFit(index_array, data_array)
        breaks = linear_fit.fit(n_segments)
        slopes = linear_fit.calc_slopes().round(4).tolist()
        data_intercepts = linear_fit.intercepts.round(4)
        index_intercepts = (data_intercepts/slopes).round(4)

        # Data prediction.
        index_prediction = np.linspace(index_array.min(), index_array.max(), n_points).round(3)
        data_prediction = linear_fit.predict(index_prediction).round(3)

        fitting_dict = {"slopes": slopes, "index_prediction": index_prediction, "data_prediction": data_prediction,
                        "data_intercepts": data_intercepts, "index_intercepts": index_intercepts}

        if n_segments > 1:
            # Compute inflection points.
            inflection_points = [float(data_prediction[np.where(index_prediction > breaks[i + 1])[0][0]].round(3)) for i
                                in range(n_segments - 1)]
            fitting_dict["inflection_points_index"] = breaks[1:-1].round(3).tolist()
            fitting_dict["inflection_points_data"] = inflection_points

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
