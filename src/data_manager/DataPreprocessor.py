import numpy as np


class DataPreprocessor:
    """Summary

        Explanation

        Note
        ----------


        Parameters
        ----------
        param1 : type
            Summary param1

            Explanation param1

        Example
        ----------
        >> instruction
        result instruction

    """

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
            return index - index[int(n_cells/2)]
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
    def crop_edge(array, x_left_edge, x_right_edge, y_bottom_edge, y_top_edge):
        """Function to crop parts of an array."""
        return array[y_bottom_edge:array.shape[0]-y_top_edge, x_left_edge:array.shape[1]-x_right_edge, :]

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
