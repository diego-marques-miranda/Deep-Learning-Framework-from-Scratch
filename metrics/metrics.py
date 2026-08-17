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

class R2(Base_Metric):
    """R-squared (Coefficient of Determination) metric tracker."""
    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        """Updates cumulative statistics for R² calculation with batch data."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        self.ss_res += np.sum((y_true - y_pred) ** 2)
        
        self.sum_y += np.sum(y_true)
        self.sum_y_sq += np.sum(y_true ** 2)
        self.count += len(y_true)

    def result(self):
        """Computes and returns the overall R-squared score."""
        if self.count == 0:
            return 0.0
            
        # SS_tot = Σ(y^2) - (Σy)^2 / n
        ss_tot = self.sum_y_sq - (self.sum_y ** 2) / self.count
        
        if ss_tot == 0:
            return 0.0
            
        return 1.0 - (self.ss_res / ss_tot)
    
    def reset(self):
        """Resets the accumulated statistics and counters to zero."""
        self.ss_res = 0.0
        self.sum_y = 0.0
        self.sum_y_sq = 0.0
        self.count = 0