import sys
import numpy as np
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from metrics.metrics import MacroPrecision


def test_macro_precision():
    print("--- Running MacroPrecision Validation Test ---")

    metric = MacroPrecision()

    # --------------------------------------------------------
    # Batch 1
    # --------------------------------------------------------
    #
    # True classes:      0, 1, 2
    # Predicted classes: 0, 1, 2
    #
    # Every class has:
    # TP = 1
    # FP = 0
    #
    # Precision per class:
    # Class 0 = 1.0
    # Class 1 = 1.0
    # Class 2 = 1.0
    #
    # MacroPrecision = 1.0
    # --------------------------------------------------------

    y_true_batch_1 = np.array([0, 1, 2])

    y_pred_batch_1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    metric.update(y_true_batch_1, y_pred_batch_1)

    result_batch_1 = metric.result()

    assert np.isclose(result_batch_1, 1.0), (
        f"Unexpected MacroPrecision after batch 1: "
        f"{result_batch_1}"
    )

    print("Batch 1: PASSED")


    # --------------------------------------------------------
    # Batch 2
    # --------------------------------------------------------
    #
    # True classes:
    #   0 appears once
    #   1 appears 10 times
    #   2 appears 10 times
    #
    # Predictions:
    #   Class 0: 1 correct + 9 false positives
    #   Class 1: 10 correct
    #   Class 2: 10 correct
    #
    # Therefore:
    #
    # Precision class 0 = 1 / (1 + 9) = 0.1
    # Precision class 1 = 10 / 10 = 1.0
    # Precision class 2 = 10 / 10 = 1.0
    #
    # If we calculated the metric independently for this batch:
    #
    # MacroPrecision = (0.1 + 1.0 + 1.0) / 3
    #                = 0.7
    #
    # But the metric must ACCUMULATE the counts from batch 1.
    #
    # Across both batches:
    #
    # Class 0:
    #   TP = 1 + 1 = 2
    #   FP = 0 + 9 = 9
    #   Precision = 2 / 11
    #
    # Class 1:
    #   TP = 1 + 10 = 11
    #   FP = 0
    #   Precision = 1
    #
    # Class 2:
    #   TP = 1 + 10 = 11
    #   FP = 0
    #   Precision = 1
    #
    # Final MacroPrecision:
    #
    # ((2 / 11) + 1 + 1) / 3
    # = 0.727272...
    # --------------------------------------------------------

    y_true_batch_2 = np.array(
        [0]
        + [1] * 10
        + [2] * 10
    )

    predicted_classes_batch_2 = np.array(
        [0]
        + [0] * 9
        + [1]
        + [2] * 10
    )

    y_pred_batch_2 = np.zeros(
        (len(predicted_classes_batch_2), 3)
    )

    y_pred_batch_2[
        np.arange(len(predicted_classes_batch_2)),
        predicted_classes_batch_2
    ] = 1.0

    metric.update(y_true_batch_2, y_pred_batch_2)

    expected_result = ((2 / 11) + 1.0 + 1.0) / 3
    actual_result = metric.result()

    assert np.isclose(actual_result, expected_result), (
        f"MacroPrecision mismatch after accumulation: "
        f"expected {expected_result}, got {actual_result}"
    )

    print("Batch accumulation: PASSED")

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    metric.reset()

    assert metric.num_classes is None
    assert metric.true_positives is None
    assert metric.false_positives is None

    print("Reset: PASSED")

    print("\nAll MacroPrecision assertions passed successfully.")


if __name__ == "__main__":
    test_macro_precision()