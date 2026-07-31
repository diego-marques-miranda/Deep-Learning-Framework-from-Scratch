import numpy as np


class MedianImputer:
    """
    Imputer for completing missing values (NaNs) using column-wise medians.

    Learns statistical medians from training feature sets and applies them
    to transform training and testing datasets independently, preventing
    data leakage.
    """

    def fit(self, data, cols=None):
        """
        Computes the median for specified or all columns from the dataset.

        Args:
            data (array-like): 2D array containing features to compute medians from.
            cols (list of int, optional): Column indices to calculate medians for.
                If None, computes medians for all columns in the dataset.
        """
        data = np.asarray(data)
        self.cols = cols

        if self.cols is None:
            self.medians = np.nanmedian(data, axis=0)
        else:
            self.medians = np.full(data.shape[1], np.nan)
            self.medians[self.cols] = np.nanmedian(data[:, self.cols], axis=0)

    def transform(self, data):
        """
        Fills missing values (NaNs) in the input data using learned medians.

        Args:
            data (array-like): 2D array with potential missing values.

        Returns:
            np.ndarray: A copy of the input data with NaNs replaced by medians.
        """
        data = np.asarray(data).copy()

        if self.cols is None:
            return np.where(np.isnan(data), self.medians, data)
        else:
            sub_data = data[:, self.cols]
            sub_medians = self.medians[self.cols]
            data[:, self.cols] = np.where(
                np.isnan(sub_data), sub_medians, sub_data
            )
            return data