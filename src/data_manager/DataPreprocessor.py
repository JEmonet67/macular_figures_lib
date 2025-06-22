import numpy as np


class DataPreprocessor:
    @staticmethod
    def array_slicing(array, slicer_indices):
        """Slicing an array into an array of smaller dimensions or a single float.

        Slicing is allowed by a tuple of the same size as the array, which allows each of these dimensions to be sliced
        or not. If the value corresponding to the dimension is an int, slicing is performed on that dimension, otherwise
        the dimension remains intact.

        Parameters
        ----------
        array : np.ndarray
            3D, 2D or 1D array to be sliced.

        slicer_indices : tuple
            Indices according to which each dimension of the array should be cut. The tuple therefore has the same
            length as the number of dimensions in the array. If a dimension should not be cut, simply enter any non-int
            object. For clarity, we recommend using ‘None’.

        Returns
        ----------
        array : np.ndarray or float
            Array that has been cut down to a smaller size than the original array or a simple floats number.
        """
        # Convert non-int slice indices to empty slice objects.
        slicer_indices = tuple([i if isinstance(i, int) else slice(None) for i in slicer_indices])

        # 3D array slicing.
        if array.ndim == 3:
            return array[slicer_indices]
        # 2D array slicing.
        elif array.ndim == 2:
            return array[slicer_indices]
        # 1D array slicing.
        elif array.ndim == 1:
            return array[slicer_indices]

    @staticmethod
    def vsdi_computing(macular_dict_array_data):
        # Set the average excitatory and inhibitory voltages.
        exc_mean_voltage = macular_dict_array_data["muVn_CorticalExcitatory"]
        inh_mean_voltage = macular_dict_array_data["muVn_CorticalInhibitory"]

        # Set of initial average voltages.
        initial_exc_mean_voltage = exc_mean_voltage[:, :, 0]
        initial_inh_mean_voltage = inh_mean_voltage[:, :, 0]

        # Calculation of the VSDI of the excitatory and inhibitory populations.
        vsdi_exc = -DataPreprocessor.array_normalization(exc_mean_voltage, initial_exc_mean_voltage)
        vsdi_inh = -DataPreprocessor.array_normalization(inh_mean_voltage, initial_inh_mean_voltage)

        # Calculation of the VSDI of the cortical column.
        vsdi = vsdi_exc * 0.8 + vsdi_inh * 0.2

        return vsdi

    @staticmethod
    def array_normalization(array_to_normalize, baseline):
        """Normalise an array by a single value, a 2D or 3D array."""
        if type(baseline) == int:
            normalized_array = (array_to_normalize - baseline) / baseline

        elif type(baseline) == np.ndarray:
            if len(array_to_normalize.shape) == 1:
                normalized_array = (array_to_normalize - float(baseline)) / float(baseline)
            elif len(array_to_normalize.shape) == 2:
                normalized_array = (array_to_normalize - baseline) / baseline
            elif len(array_to_normalize.shape) == 3:
                # Extension of the initial voltages to have the same time dimension as the data to be normalised.
                baseline_3d = np.repeat(baseline[:, :, np.newaxis],
                                        array_to_normalize.shape[-1], axis=2)
                normalized_array = (array_to_normalize - baseline_3d) / baseline_3d

        return normalized_array

    @staticmethod
    def computing_binning_parameters(index, bin_time):
        # Number of indices to be averaged within the binning time.
        bin_size = int(bin_time / round(index[1] - index[0], 6))
        # Number of binning intervals to cover the index.
        n_bin = int((index[-1] - index[0]) / bin_time)

        return bin_size, n_bin

    @staticmethod
    def binning_unidimensional(index, bin_size, n_bin):
        return index[:bin_size * n_bin].reshape((n_bin, bin_size)).mean(axis=-1).round(5)

    @staticmethod
    def binning_tridimensional(array, bin_size, n_bin):
        return array[:, :, :bin_size * n_bin].reshape(
            (array.shape[0], array.shape[1], n_bin, bin_size)).mean(axis=-1)

    @staticmethod
    def temporal_centering(index, list_time_center):
        # Incrementing the list of time indexes re-centred for each cell in the bar axis.
        list_index_centered = []

        for time_center in list_time_center:
            list_index_centered.append(index - time_center)

        return np.array(list_index_centered)

    @staticmethod
    def spatial_centering(index, n_cells):
        # Odd n_cells.
        if n_cells // 2:
            return index - index[int(n_cells / 2)]
        # Even n_cells.
        else:
            return index - (index[int(n_cells / 2)] + index[int(n_cells / 2) - 1]) / 2

    @staticmethod
    def derivative_computing_3d_array(array, index, n=1):
        """Function to compute derivative of a 3D array to be derived. The derivative can depend
        on n_value or to be instant."""
        df_dxdt = np.zeros(array.shape)

        # Go through temporal index
        for i_derivate in range(array.shape[2]):
            # Case where the time index is too close to the lower limit for the window to fit.
            if i_derivate < n:
                df_dxdt[:, :, i_derivate] = (array[:, :, i_derivate + n] - array[:, :, 0])
                df_dxdt[:, :, i_derivate] = df_dxdt[:, :, i_derivate] / (index[i_derivate + n] - index[0])

            # Case where the time index is too close to the upper limit for the window to fit.
            elif i_derivate >= array.shape[2] - n:
                df_dxdt[:, :, i_derivate] = (array[:, :, array.shape[2] - 1] - array[:, :, i_derivate - n])
                df_dxdt[:, :, i_derivate] = df_dxdt[:, :, i_derivate] / (
                        index[array.shape[2] - 1] - index[i_derivate - n])
            # Intermediate case where the n window fits within the neighbourhood of the current time index.
            else:
                df_dxdt[:, :, i_derivate] = (array[:, :, i_derivate + n] - array[:, :, i_derivate - n])
                df_dxdt[:, :, i_derivate] = df_dxdt[:, :, i_derivate] / (index[i_derivate + n] - index[i_derivate - n])

        return df_dxdt

    @staticmethod
    def conversion_specific_arrays_unit_dict_array(dict_array, pattern, suffix_array, ratio):
        """
        Function that converts all arrays in a dictionary whose names contain a given pattern.

        The conversion is done by multiplying each array by a ratio. The converted arrays are stored in new key-value
        pairs in the dictionary, where the key is followed by a suffix indicating the type of conversion performed.
        """
        dict_array_new_units = dict_array.copy()
        for name_array in dict_array_new_units.copy():
            if pattern in name_array:
                dict_array_new_units[f"{name_array}_{suffix_array}"] = dict_array_new_units[name_array] * ratio

        return dict_array_new_units

    @staticmethod
    def array_edge_cropping(array, cropping_dict={}):
        """Function to crop the edges of each axis of a 3D array.

        Parameters
        ----------
        array : np.ndarray
            3D array to be cropped.

        cropping_dict : dict
            Dictionary of cropping settings.

            It can contain a key for each extremity of each axis. We have ‘x_min_edge’ and ‘x_max_edge’ for the X
            spatial dimension, ‘y_min_edge’ and ‘y_max_edge’ for the Y spatial dimension, “t_min_edge” and ‘t_max_edge’
            for the T temporal dimension.

        Returns
        ----------
        array : np.ndarray
            3D array cropped based on the edges of the dictionary.
        """
        # Initialisation of the default dictionary.
        default_cropping_dict = {"x_min_edge": 0, "x_max_edge": 0, "y_min_edge": 0, "y_max_edge": 0,
                                 "t_min_edge": 0, "t_max_edge": 0}

        # Merging of default dictionary keys not present in the crop dictionary provided as input.
        cropping_dict = {key: cropping_dict[key] if key in cropping_dict else default_cropping_dict[key] for key in
                         default_cropping_dict}

        # Cropping each axis of the array with its new respective edges.
        return array[cropping_dict["y_min_edge"]:array.shape[0] - cropping_dict["y_max_edge"],
           cropping_dict["x_min_edge"]:array.shape[1] - cropping_dict["x_max_edge"],
           cropping_dict["t_min_edge"]:array.shape[2] - cropping_dict["t_max_edge"]]

    @staticmethod
    def crop_imbricated_array(imbricated_array, cropping_type, cropping_dict):
        """Function to use differents ways of cropping an imbricated array.

        The crop of a imbricated array corresponds to cropping the isolated 1D array present at each position of the 2D
        array. The purpose of this structure is to allow 1D arrays of different sizes to coexist after a certain
        cropping method.

        The different crop methods implemented are:
        - The ‘fixed_edge’ crop removes edges of a fixed size specified by the user.
        - The ‘threshold’ crop removes all positions below a given threshold.
        - The ‘max_ratio_threshold’ crop removes all positions whose activity is less than a certain fraction of the
        maximum response of the array to be cropped.

        Parameters
        ----------
        imbricated_array : np.ndarray of np.ndarray
            2D array containing isolated 1D arrays to be cropped.

        cropping_type : str
            Name of the cropping type to be applied to the isolated array of the imbricated array.

        cropping_dict : dict
            Dictionary of cropping settings. Settings depend on the type of cropping selected.

        Returns
        ----------
        imbricated_array : np.ndarray of np.ndarray
            Imbricated array whose isolated array has been cropped or not.
        """
        if cropping_type == "fixed_edge":
            return DataPreprocessor.fixed_edge_cropping(imbricated_array, cropping_dict.copy())

        elif cropping_type == "threshold":
            return DataPreprocessor.threshold_cropping(imbricated_array, cropping_dict["threshold"])

        elif cropping_type == "max_ratio_threshold":
            return DataPreprocessor.max_ratio_threshold_cropping(imbricated_array, cropping_dict["ratio_threshold"])
        else:
            return imbricated_array

    @staticmethod
    def fixed_edge_cropping(imbricated_array, cropping_dict={}):
        """Function to crop the edges of one axis of an imbricated array.

        Parameters
        ----------
        imbricated_array : np.ndarray of np.ndarray
            2D array containing isolated 1D arrays to be cropped.

        cropping_dict : dict
            Dictionary of cropping settings.

            It can hold a key for each end of the axis to crop. We have ‘edge_start’ and ‘edge_end’ for the beginning
            and end of the array on the axis, respectively.

        Returns
        ----------
        imbricated_array : np.ndarray of np.ndarray
            2D array containing isolated and cropped 1D arrays.
        """
        # Initialisation of the default dictionary.
        default_cropping_dict = {"edge_start": 0, "edge_end": 0}

        # Merging of default dictionary keys not present in the crop dictionary provided as input.
        for key, value in default_cropping_dict.items():
            if key not in cropping_dict:
                cropping_dict[key] = value

        # Cropping of all isolated arrays contained in the first 2D array.
        for i in range(imbricated_array.shape[0]):
            for j in range(imbricated_array.shape[1]):
                imbricated_array[i, j] = imbricated_array[i, j][cropping_dict["edge_start"]:
                                                                imbricated_array[i, j].shape[0] -
                                                                cropping_dict["edge_end"]]

        return imbricated_array


    @staticmethod
    def threshold_cropping(imbricated_array, threshold):
        """Function that removes all values from an imbricated array that are less than a threshold.

        Parameters
        ----------
        imbricated_array : np.ndarray of np.ndarray
            2D array containing isolated 1D arrays to be cropped.

        threshold : int or float
            Threshold value to apply for array filtering.

        Returns
        ----------
        imbricated_array : np.ndarray of np.ndarray
            2D array containing isolated and threshold cropped 1D arrays.
        """
        # Initialises an array of the same size as the first imbricated array.
        cropped_imbricated_array = np.empty((imbricated_array.shape[0], imbricated_array.shape[1]), dtype=object)

        # Loop the coordinates on the two axes of the first imbricated array.
        for i in range(imbricated_array.shape[0]):
            for j in range(imbricated_array.shape[1]):
                # Retrieves the 1D array from the current position to be cropped.
                array_to_threshold = imbricated_array[i, j].astype(float)
                # Crop of the current 1D array.
                cropped_imbricated_array[i, j] = array_to_threshold[array_to_threshold >= threshold]

        return cropped_imbricated_array

    @staticmethod
    def max_ratio_threshold_cropping(imbricated_array, ratio_threshold):
        """Function that removes all values from an imbricated array that are less than a dynamic threshold calculated
        from a fraction of the local maximum.

        The threshold is calculated by multiplying the threshold ratio by the maximum value present within each isolated
        array in the imbricated array. This ensures that only the local maximum present in a given axis is taken into
        account.

        Parameters
        ----------
        imbricated_array : np.ndarray of np.ndarray
            2D array containing isolated 1D arrays to be cropped.

        ratio_threshold : float
            Threshold value corresponding to the local maximum ratio used as the effective threshold.

        Returns
        ----------
        imbricated_array : np.ndarray of np.ndarray
            2D array containing isolated and max ratio threshold cropped 1D arrays.
        """
        # Initialises an array of the same size as the first imbricated array.
        cropped_imbricated_array = np.empty((imbricated_array.shape[0], imbricated_array.shape[1]), dtype=object)

        # Loop the coordinates on the two axes of the first imbricated array.
        for i in range(imbricated_array.shape[0]):
            for j in range(imbricated_array.shape[1]):
                # Retrieves the 1D array from the current position to be cropped.
                array_to_threshold = imbricated_array[i, j]
                # Calculation of the dynamic threshold depending on the maximum present in the array to be cropped.
                threshold = ratio_threshold * array_to_threshold.max()
                # Crop of the current 1D array.
                cropped_imbricated_array[i, j] = array_to_threshold[array_to_threshold >= threshold]

        return cropped_imbricated_array

    @staticmethod
    def transform_3d_array_to_imbricated_arrays(array, axis):
        """Function that transforms a 3D array into a double imbricated array.

        The imbricated array retains all the data from the 3D array but transforms its structure to isolate one of the
        axes within a 1D array. This 1D array of the axis to be isolated is stored within a larger 2D array with the
        same dimensions as the other two axes of the original array.

        This process allows you to isolate the values of an array along a given axis, then crop them to create arrays
        of variable sizes, and finally perform calculations on them, such as averaging.

        Parameters
        ----------
        array : np.ndarray
            3d array of data to convert into an imbricated array.

        axis : str
            Axis along which the process is performed, this is the axis included in the last array of imbricated array.

        Returns
        ----------
        imbricated_array : np.ndarray of np.ndarray
            Imbricated array reorganising values to isolate a given axis.
        """
        # Vertical case.
        if axis == "vertical":
            # Initialise the array with an appropriate size and allow it to store array.
            imbricated_array = np.empty((array.shape[1], array.shape[2]), dtype=object)
            # Loop the coordinates on the other two axes.
            for x in range(array.shape[1]):
                for t in range(array.shape[2]):
                    # Fill the current position with the corresponding array represented according to the desired axis.
                    imbricated_array[x, t] = array[:, x, t]

        # Horizontal case.
        elif axis == "horizontal":
            # Initialise the array with an appropriate size and allow it to store array.
            imbricated_array = np.empty((array.shape[0], array.shape[2]), dtype=object)
            # Loop the coordinates on the other two axes.
            for y in range(array.shape[0]):
                for t in range(array.shape[2]):
                    # Fill the current position with the corresponding array represented according to the desired axis.
                    imbricated_array[y, t] = array[y, :, t]

        # Temporal case.
        elif axis == "temporal":
            # Initialise the array with an appropriate size and allow it to store array.
            imbricated_array = np.empty((array.shape[0], array.shape[1]), dtype=object)
            # Loop the coordinates on the other two axes.
            for y in range(array.shape[0]):
                for x in range(array.shape[1]):
                    # Fill the current position with the corresponding array represented according to the desired axis.
                    imbricated_array[y, x] = array[y, x, :]

        return imbricated_array

    @staticmethod
    def imbricated_arrays_axis_averaging(imbricated_array):
        """Function for calculating average sections for a specific axis in an imbricated array.

        Parameters
        ----------
        imbricated_array : np.ndarray of np.ndarray
            2D array containing isolated 1D arrays to be averaged.

        Returns
        ----------
        mean_section : np.ndarray
            2D array corresponding to the average section along the axis contained in the isolated 1D array.
        """
        # Initialisation of the average section array with the two axes of the first imbricated array.
        mean_section = np.empty((imbricated_array.shape[0], imbricated_array.shape[1]), dtype=object)

        # Loop on both axes of the first imbricated array.
        for i in range(imbricated_array.shape[0]):
            for j in range(imbricated_array.shape[1]):
                # Retrieves the 1D array from the current position to be cropped.
                array_to_mean = imbricated_array[i, j]
                # Average of the current 1D array.
                mean_section[i, j] = array_to_mean.mean().round(4)

        return mean_section.astype(float)
