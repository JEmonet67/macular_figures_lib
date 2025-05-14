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
        # Initialises a 2D array with the same size as the spatial dimensions of the data array.
        activation_time_array = np.empty((data_array.shape[0], data_array.shape[1]))

        # Set up a filter based on the activation threshold.
        data_array_test_threshold = (data_array > threshold)

        # Loops at each horizontal and vertical position of the 3D array.
        for i in range(data_array.shape[1]):
            for j in range(data_array.shape[0]):
                # Cases where an activation time exists.
                try:
                    activation_time_index = np.argwhere(data_array_test_threshold[j][i])[0][0]
                    activation_time_array[j][i] = round(index_array[activation_time_index], 3)
                # Cases where no activation time was found.
                except IndexError:
                    activation_time_array[j][i] = np.nan

        return activation_time_array

    @staticmethod
    def latency_computing(data_array, index_array, threshold, axis):
        """Calculation of the latency from which the response exceeds a given threshold.

        Parameters
        ----------
        data_array : np.array
            3D array containing activity data.

        index_array : list of np.array
            One-dimensional array containing the time index to be used.

        threshold : floats
            Activation threshold used to calculate the activation time.

        axis : str
            Axis of the object's movement ("horizontal" or "vertical").

        Returns
        ----------
        activation_time_array : np.array()
            2D array containing the activation times of the 3D array given as input.
        """
        # Initialises a 2D array with the same size as the spatial dimensions of the data array.
        latency_array = np.empty((data_array.shape[0], data_array.shape[1]))

        # Set up a filter based on the activation threshold.
        data_array_test_threshold = (data_array > threshold)

        # Loops at each horizontal and vertical position of the 3D array.
        for i in range(data_array.shape[1]):
            if axis == "horizontal":
                current_index = index_array[i]
            for j in range(data_array.shape[0]):
                if axis == "vertical":
                    current_index = index_array[j]
                # Cases where an activation time exists.
                try:
                    latency_index = np.argwhere(data_array_test_threshold[j][i])[0][0]
                    latency_array[j][i] = round(current_index[latency_index], 3)
                # Cases where no activation time was found.
                except IndexError:
                    latency_array[j][i] = np.nan

        return latency_array

    @staticmethod
    def time_to_peak_computing(data_array, index_array):
        """Calculation of the time to peak from which the response is maximal.

        Parameters
        ----------
        data_array : np.array
            3D array containing activity data.

        index_array : list of np.array
            One-dimensional array containing the time index to be used.

        Returns
        ----------
        time_to_peak_array : np.array()
            2D array containing the time to peak of the 3D array given as input.
        """
        # Initialises a 2D array with the same size as the spatial dimensions of the data array.
        time_to_peak_array = np.empty((data_array.shape[0], data_array.shape[1]))

        # Loops at each horizontal and vertical position of the 3D array.
        for i in range(data_array.shape[1]):
            for j in range(data_array.shape[0]):
                # Cases where a peak exists.
                try:
                    time_to_peak_index = data_array[j][i].argmax()
                    time_to_peak_array[j][i] = round(index_array[time_to_peak_index], 3)
                # Cases where no peak was found.
                except IndexError:
                    time_to_peak_array[j][i] = np.nan

        return time_to_peak_array

    @staticmethod
    def latency_computing(data_array, index_array, threshold, axis):
        """Calculation of the latency from which the response exceeds a given threshold.

        Parameters
        ----------
        data_array : np.array
            3D array containing activity data.

        index_array : list of np.array
            One-dimensional array containing the time index to be used.

        threshold : floats
            Activation threshold used to calculate the latency.

        axis : str
            Axis of the object's movement ("horizontal" or "vertical").

        Returns
        ----------
        latency_array : np.array()
            2D array containing the latency of the 3D array given as input.
        """
        # Initialises a 2D array with the same size as the spatial dimensions of the data array.
        latency_array = np.empty((data_array.shape[0], data_array.shape[1]))

        # Set up a filter based on the activation threshold.
        data_array_test_threshold = (data_array > threshold)

        # Loops at each horizontal and vertical position of the 3D array.
        for i in range(data_array.shape[1]):
            if axis == "horizontal":
                current_index = index_array[i]
            for j in range(data_array.shape[0]):
                if axis == "vertical":
                    current_index = index_array[j]
                # Cases where an activation time exists.
                try:
                    latency_index = np.argwhere(data_array_test_threshold[j][i])[0][0]
                    latency_array[j][i] = round(current_index[latency_index], 3)
                # Cases where no activation time was found.
                except IndexError:
                    latency_array[j][i] = np.nan

        return latency_array


    @staticmethod
    def peak_delay_computing(data_array, index_array, axis):
        """Calculation of the delay to peak from which the response is maximal.

        Parameters
        ----------
        data_array : np.array
            3D array containing activity data.

        index_array : list of np.array
            One-dimensional array containing the time index to be used.

        axis : str
            Axis of the object's movement ("horizontal" or "vertical").

        Returns
        ----------
        peak_delay_array : np.array()
            2D array containing the delay to peak of the 3D array given as input.
        """
        # Initialises a 2D array with the same size as the spatial dimensions of the data array.
        peak_delay_array = np.empty((data_array.shape[0], data_array.shape[1]))

        # Loops at each horizontal and vertical position of the 3D array.
        for i in range(data_array.shape[1]):
            if axis == "horizontal":
                current_index = index_array[i]
            for j in range(data_array.shape[0]):
                if axis == "vertical":
                    current_index = index_array[j]
                # Cases where an peak exists.
                try:
                    peak_delay_index = data_array[j][i].argmax()
                    peak_delay_array[j][i] = round(current_index[peak_delay_index], 3)
                # Cases where no peak was found.
                except IndexError:
                    peak_delay_array[j][i] = np.nan

        return peak_delay_array
