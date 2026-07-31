import numpy as np
from abc import ABC, abstractmethod


class Base_Scaler(ABC):
    """Abstract base class defining the standard interface for feature scaling."""

    @abstractmethod
    def fit(self, data):
        """Computes scaling parameters from the dataset."""
        pass

    @abstractmethod
    def transform(self, data):
        """Applies the scaling transformation using the fitted parameters."""
        pass

    @abstractmethod
    def inverse_transform(self, data):
        """Reverts scaled data back to its original physical scale."""
        pass


class MinMaxScaler(Base_Scaler):
    """Standard Min-Max scaler mapping features to a normalized range [0, 1]."""

    def __init__(self):
        """Initializes feature minimum and maximum bounds."""
        self.min = 0
        self.max = 0

    def fit(self, data):
        """
        Learns the minimum and maximum values along each feature axis.

        Args:
            data (array-like): Feature matrix used to compute min and max bounds.
        """
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        """
        Normalizes data using learned min/max bounds.

        Adds a small numerical epsilon (1e-8) in the denominator 
        to prevent division by zero when bounds are equal.

        Args:
            data (array-like): Feature matrix or array to normalize.

        Returns:
            np.ndarray: Scaled array with matching dimensions.
        """
        return (data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        """
        Denormalizes data back to the original physical scale.

        Args:
            data (array-like): Normalized feature array or model predictions.

        Returns:
            np.ndarray: Array rescaled to original feature bounds.
        """
        return data * (self.max - self.min) + self.min