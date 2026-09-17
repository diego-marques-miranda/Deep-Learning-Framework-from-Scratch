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

class Accuracy(Base_Metric):
    """Classification accuracy metric tracker."""
    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        """Updates cumulative correct predictions with batch data."""
        # Converts predicted probabilities/logits to class indices
        correct_pred = np.argmax(y_pred, axis=1)

        self.samples += len(y_true)
        self.correct_pred += np.sum(y_true == correct_pred)

    def result(self):
        """Computes and returns the overall accuracy score."""
        if self.samples == 0:
            return 0.0
        return self.correct_pred / self.samples

    def reset(self):
        """Resets the samples and correct prediction counters to zero."""
        self.samples = 0
        self.correct_pred = 0

class MacroPrecision(Base_Metric):
    """Macro-averaged precision for multiclass classification."""

    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        """
        Accumulates true positives and false positives for each class.
        ``y_pred`` is expected to contain class probabilities or scores.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        num_classes = y_pred.shape[1]
        predicted_classes = np.argmax(y_pred, axis=1)

        if self.true_positives is None:
            self.num_classes = num_classes
            self.true_positives = np.zeros(num_classes, dtype=np.int64)
            self.false_positives = np.zeros(num_classes, dtype=np.int64)

        if num_classes != self.num_classes:
            raise ValueError(
                "Number of classes in y_pred changed between updates."
            )

        for class_index in range(self.num_classes):
            predicted_positive = predicted_classes == class_index
            actual_positive = y_true == class_index

            self.true_positives[class_index] += np.sum(
                predicted_positive & actual_positive
            )

            self.false_positives[class_index] += np.sum(
                predicted_positive & ~actual_positive
            )

    def result(self):
        if self.num_classes is None:
            return 0.0

        denominators = self.true_positives + self.false_positives

        precision_per_class = np.divide(
            self.true_positives,
            denominators,
            out=np.zeros(self.num_classes, dtype=float),
            where=denominators != 0
        )

        return np.mean(precision_per_class)

    def reset(self):
        self.num_classes = None
        self.true_positives = None
        self.false_positives = None

class MacroRecall(Base_Metric):
    """Macro-averaged recall for multiclass classification."""

    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        """
        Accumulates true positives and false negatives for each class.
        ``y_pred`` is expected to contain class probabilities or scores.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        num_classes = y_pred.shape[1]
        predicted_classes = np.argmax(y_pred, axis=1)

        if self.num_classes is None:
            self.num_classes = num_classes
            self.true_positives = np.zeros(
                num_classes,
                dtype=np.int64
            )
            self.false_negatives = np.zeros(
                num_classes,
                dtype=np.int64
            )

        if num_classes != self.num_classes:
            raise ValueError(
                "Number of classes in y_pred changed between updates."
            )

        for class_index in range(self.num_classes):
            actual_positive = y_true == class_index
            predicted_positive = predicted_classes == class_index

            self.true_positives[class_index] += np.sum(
                actual_positive & predicted_positive
            )

            self.false_negatives[class_index] += np.sum(
                actual_positive & ~predicted_positive
            )

    def result(self):
        if self.num_classes is None:
            return 0.0

        denominators = (
            self.true_positives
            + self.false_negatives
        )

        recall_per_class = np.divide(
            self.true_positives,
            denominators,
            out=np.zeros(self.num_classes, dtype=float),
            where=denominators != 0
        )

        return np.mean(recall_per_class)

    def reset(self):
        self.num_classes = None
        self.true_positives = None
        self.false_negatives = None