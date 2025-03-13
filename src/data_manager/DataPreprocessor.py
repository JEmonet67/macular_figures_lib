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
        print("VSDI computing...", end="")
        exc_mean_voltage = macular_dict_array_data["muVn_CorticalExcitatory"]
        inh_mean_voltage = macular_dict_array_data["muVn_CorticalInhibitory"]

        initial_exc_mean_voltage = exc_mean_voltage[:, :, 0]
        initial_exc_mean_voltage = np.repeat(initial_exc_mean_voltage[:, :, np.newaxis],
                                             exc_mean_voltage.shape[-1], axis=2)
        initial_inh_mean_voltage = inh_mean_voltage[:, :, 0]
        initial_inh_mean_voltage = np.repeat(initial_inh_mean_voltage[:, :, np.newaxis],
                                             inh_mean_voltage.shape[-1], axis=2)

        vsdi_exc = (-(exc_mean_voltage - initial_exc_mean_voltage) / initial_exc_mean_voltage)
        vsdi_inh = (-(inh_mean_voltage - initial_inh_mean_voltage) / initial_inh_mean_voltage)

        vsdi = vsdi_exc * 0.8 + vsdi_inh * 0.2

        return vsdi


    @staticmethod
    def computing_binning_parameters(index, bin_time):
        print(f"Binning {bin_time}s...", end="")
        bin_size = int(bin_time / round(index[1] - index[0], 6))
        n_bin = int((index[-1] - index[0]) / bin_time)

        return bin_size, n_bin

    @staticmethod
    def binning_index(index, bin_size, n_bin):
        return index[:bin_size * n_bin].reshape((n_bin, bin_size)).mean(axis=-1)

    @staticmethod
    def binning_data_array(array, bin_size, n_bin):
        return array[:, :, :bin_size * n_bin].reshape(
            (array.shape[0], array.shape[1], n_bin, bin_size)).mean(axis=-1)

    @staticmethod
    def temporal_centering(index, list_time_center):
        print("Temporal centering...", end="")
        list_index_centered = []
        for time_center in list_time_center:
            list_index_centered.append(index - time_center)

        return list_index_centered

    @staticmethod
    def derivative_computing_3d_array(array, index, n=1):
        """Function to compute derivative of a 3D array to be derived. The derivative can depend
        on n_value or to be instant."""
        print("Derivating...", end="")
        df_dxdt = np.zeros(array.shape)
        for i_derivate in range(array.shape[2]):
            if i_derivate < n:
                df_dxdt[:, :, i_derivate] = (array[:, :, i_derivate + n] - array[:, :, 0])
                df_dxdt[:, :, i_derivate] = df_dxdt[:, :, i_derivate] / (index[i_derivate + n] - index[0])
            elif i_derivate >= array.shape[2] - n:
                df_dxdt[:, :, i_derivate] = (array[:, :, array.shape[2] - 1] - array[:, :, i_derivate - n])
                df_dxdt[:, :, i_derivate] = df_dxdt[:, :, i_derivate] / (
                            index[array.shape[2] - 1] - index[i_derivate - n])
            else:
                df_dxdt[:, :, i_derivate] = (array[:, :, i_derivate + n] - array[:, :, i_derivate - n])
                df_dxdt[:, :, i_derivate] = df_dxdt[:, :, i_derivate] / (index[i_derivate + n] - index[i_derivate - n])

        return df_dxdt