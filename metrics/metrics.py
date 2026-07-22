import numpy as np
from abc import ABC, abstractmethod

class Base_Metric(ABC):
    """Abstract base class defining the standard interface for performance metrics."""
    @abstractmethod
    def update(self, y_true, y_pred):
        """Accumulates metric state based on batch targets and predictions."""
        pass

    @abstractmethod
    def result(self):
        """Computes and returns the final metric value."""
        pass

    @abstractmethod
    def reset(self):
        """Clears the accumulated metric states."""
        pass

class MAPE(Base_Metric):
    """Mean Absolute Percentage Error metric tracker."""
    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        """Updates cumulative absolute percentage errors with batch data."""
        batch_mape = np.abs((y_true - y_pred) / (y_true + 1e-8)) # epsilon added for division safety
        self.sum_mape += np.sum(batch_mape)
        self.count += len(y_true)

    def result(self):
        """Returns the overall MAPE as a percentage."""
        return self.sum_mape / self.count * 100
    
    def reset(self):
        """Resets the sum and sample counters to zero."""
        self.sum_mape = 0
        self.count = 0