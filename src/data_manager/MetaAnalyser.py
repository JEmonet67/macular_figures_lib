import numpy as np
import pwlf


class MetaAnalyser:
    """Class grouping all functions intended to meta-analyse certain properties of a 3D spatio-temporal data array.

    Each of these functions is initially created to be used for meta-analysis within a MacularAnalysisDataframes.
    """

    @staticmethod
    def normalization_computing(numerator, denominator, factor):
        """Calculation of a normalization between two arrays, int or float with the possibility to multiply by a given
        factor.

        Parameters
        ----------
        numerator : np.ndarray, int or float
            Numerator of the division.

        denominator : np.ndarray, int or float
            Denominator of the division.

        factor : float or int
            Multiplication factor to be used for normalization.

        Returns
        ----------
        normalization_values : np.ndarray, int or float
            Value(s) obtained from the normalization.
        """
        normalization_values = numerator / denominator * factor

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

        # Data prediction.
        index_prediction = np.linspace(index_array.min(), index_array.max(), n_points).round(3)
        data_prediction = linear_fit.predict(index_prediction).round(3)

        fitting_dict = {"slopes": slopes, "index_prediction": index_prediction, "data_prediction": data_prediction}

        if n_segments > 1:
            # Compute inflexion points.
            inflexion_points = [float(data_prediction[np.where(index_prediction > breaks[i + 1])[0][0]].round(3)) for i
                                in range(n_segments - 1)]
            fitting_dict["inflexion_points_index"] = breaks[1:-1].round(3).tolist()
            fitting_dict["inflexion_points_data"] = inflexion_points

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
