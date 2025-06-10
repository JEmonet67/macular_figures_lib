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
        data_array : np.ndarray
            3D array containing activity data.

        index_array : np.ndarray
            One-dimensional array containing the time index to be used.

        threshold : floats or np.ndarray
            Activation threshold used to calculate the activation time.

        Returns
        ----------
        activation_time_array : np.ndarray
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
    def dynamic_threshold_computing(data_array, threshold_ratio):
        """Calculation of an array containing different dynamic thresholds proportional to the maximum local activity
        value present in each array contained at each position of the main and secondary axes of a 3D array.

        A threshold is calculated for each X and Y position of the 3D array by multiplying the maximum value of the
        array contained there by the threshold ratio. This dynamic 2D threshold array is repeated as many times as the
        size of the tertiary axis of the data array so that it can serve as a filter of the same size.

        Parameters
        ----------
        data_array : np.ndarray
            3D array containing activity data.

        threshold_ratio : floats
            Dynamic threshold ratio to be applied to each local maximum.

        Returns
        ----------
        dynamic_threshold : np.ndarray
            3D array containing a different dynamic threshold value for each position in X and Y.
        """
        # Initialisation of an empty array to hold the dynamic threshold.
        dynamic_threshold = np.empty((data_array.shape[0], data_array.shape[1]))

        # Loop through each position in the data array.
        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                # Calculation of the dynamic threshold of the current position.
                dynamic_threshold[i][j] = data_array[i][j].max() * threshold_ratio

        dynamic_threshold_3d = np.repeat(dynamic_threshold[:, :, np.newaxis], data_array.shape[2], axis=2)

        return dynamic_threshold_3d


    @staticmethod
    def latency_computing(data_array, index_array, threshold, axis):
        """Calculation of the latency from which the response exceeds a given threshold.

        Parameters
        ----------
        data_array : np.ndarray
            3D array containing activity data.

        index_array : list of np.ndarray
            One-dimensional array containing the time index to be used.

        threshold : floats
            Activation threshold used to calculate the activation time.

        axis : str
            Axis of the object's movement ("horizontal" or "vertical").

        Returns
        ----------
        activation_time_array : np.ndarray
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
        data_array : np.ndarray
            3D array containing activity data.

        index_array : list of np.ndarray
            One-dimensional array containing the time index to be used.

        Returns
        ----------
        time_to_peak_array : np.ndarray
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
        data_array : np.ndarray
            3D array containing activity data.

        index_array : list of np.ndarray
            One-dimensional array containing the time index to be used.

        threshold : floats
            Activation threshold used to calculate the latency.

        axis : str
            Axis of the object's movement ("horizontal" or "vertical").

        Returns
        ----------
        latency_array : np.ndarray
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
        data_array : np.ndarray
            3D array containing activity data.

        index_array : list of np.ndarray
            One-dimensional array containing the time index to be used.

        axis : str
            Axis of the object's movement ("horizontal" or "vertical").

        Returns
        ----------
        peak_delay_array : np.ndarray
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

    @staticmethod
    def peak_amplitude_computing(data_array):
        """Calculation of the peak amplitude from the response.

        Parameters
        ----------
        data_array : np.ndarray
            3D array containing activity data.

        Returns
        ----------
        amplitude_array : np.ndarray
            2D array containing the amplitude of the 3D array given as input.
        """
        # Initialises a 2D array with the same size as the spatial dimensions of the data array.
        amplitude_array = np.empty((data_array.shape[0], data_array.shape[1]))

        # Loops at each horizontal and vertical position of the 3D array.
        for i in range(data_array.shape[1]):
            for j in range(data_array.shape[0]):
                # Cases where a peak exists.
                try:
                    amplitude_array[j][i] = data_array[j][i].max().round(3)
                # Cases where no peak was found.
                except IndexError:
                    amplitude_array[j][i] = np.nan

        return amplitude_array

    @staticmethod
    def initial_amplitude_computing(data_array):
        """Calculation of the initial amplitude of the response.

        Parameters
        ----------
        data_array : np.ndarray
            3D array containing activity data.

        Returns
        ----------
        amplitude_array : np.ndarray
            2D array containing the initial amplitude of the 3D array given as input.
        """
        # Initialises a 2D array with the same size as the spatial dimensions of the data array.
        amplitude_array = np.empty((data_array.shape[0], data_array.shape[1]))

        # Loops at each horizontal and vertical position of the 3D array.
        for i in range(data_array.shape[1]):
            for j in range(data_array.shape[0]):
                amplitude_array[j][i] = data_array[j][i][0].round(3)

        return amplitude_array
