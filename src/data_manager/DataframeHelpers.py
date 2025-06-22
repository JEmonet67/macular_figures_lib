import re


class DataframeHelpers:
    def __init__(self):
        # Extraction of the name, value and unit of a condition.
        self.name_value_unit_cond_reg = re.compile("(^[A-Za-z]+)(-?[0-9]{1,4},?[0-9]{0,4})([A-Za-z]+$)")

    @property
    def name_value_unit_cond_reg(self):
        return self.name_value_unit_cond_reg

    @name_value_unit_cond_reg.setter
    def name_value_unit_cond_reg(self, name_value_unit_cond_reg):
        self._name_value_unit_cond_reg = name_value_unit_cond_reg

    @staticmethod
    def crop_dataframe_rows(dataframe, min_index, max_index):
        # Definition of maximal time.
        if max_index == "max":
            max_index = dataframe.index[-1]

        print(f"Dataframe cropping from : {min(max(dataframe.index[0], min_index), max_index)}s "
              f"to {min(dataframe.index[-1], max_index)}s")
        # Cropping of the dataframe between the minimum and maximum indicated.
        dataframe = dataframe[(dataframe.index >= min_index) & (dataframe.index <= max_index)]
        # Re-centring of the index.
        dataframe.index = dataframe.index - min_index

        return dataframe

    def sort_macular_dataframe(self, dataframe):
        # Sorting of the conditions index according to condition name and then condition value.
        print("Sorting...", end="")
        dataframe = dataframe.sort_index(axis=1, key=lambda x: [
            (self.name_value_unit_cond_reg.findall(elt)[0][0],
             float(self.name_value_unit_cond_reg.findall(elt)[0][1]))
            for elt in x.tolist()])
        print("Done!")

        return dataframe
