import numpy as np


class SpatialAnalyser:
    """Class grouping all functions intended to analyse certain spatial properties of a 3D spatio-temporal data array.

    Each of these functions is initially created to be used for analysis within a MacularAnalysisDataframes.
    """
    @staticmethod
    def activation_time_computing(data_array, index_array, threshold):
        """Calculation of the activation time from which the response exceeds a given threshold.

        Parameters
        ----------
        data_array : np.array
            3D array containing activity data.

        index_array : np.array
            One-dimensional array containing the time index to be used.

        threshold : floats
            Activation threshold used to calculate the activation time.

        Returns
        ----------
        activation_time_array : np.array()
            2D array containing the activation times of the 3D array given as input.
        """
        # Loops at each horizontal and vertical position of the 3D array.
        activation_time_array = np.empty((data_array.shape[0], data_array.shape[1]))

        # Set up a filter based on the activation threshold.
        data_array_test_threshold = (data_array > threshold)

        # Loops at each horizontal and vertical position of the 3D array.
        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                # Cases where an activation time exists.
                try:
                    activation_time_index = np.argwhere(data_array_test_threshold[i][j])[0][0]
                    activation_time_array[i][j] = index_array[activation_time_index]
                # Cases where no activation time was found.
                except IndexError:
                    activation_time_array[i][j] = np.nan

        return activation_time_array




