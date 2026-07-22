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

class MinMaxScaler(Base_Scaler):
    """Standard Min-Max scaler mapping features to a normalized range with stability epsilon."""
    def __init__(self):
        self.min = 0
        self.max = 0
    
    def fit(self, data):
        """Learns the minimum and maximum values along each feature axis."""
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        """Normalizes data using the learned min/max bounds, adding a small epsilon to prevent division by zero."""
        return (data - self.min) / (self.max - self.min) + 1e-8