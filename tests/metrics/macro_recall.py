import sys
import numpy as np
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from metrics.metrics import MacroRecall


def test_macro_recall():
    print("--- Running MacroRecall Validation Test ---")

    metric = MacroRecall()

    # --------------------------------------------------------
    # Batch 1
    # --------------------------------------------------------
    #
    # True classes:      0, 1, 2
    # Predicted classes: 0, 1, 2
    #
    # Every class has:
    # TP = 1
    # FN = 0
    #
    # Recall per class:
    # Class 0 = 1.0
    # Class 1 = 1.0
    # Class 2 = 1.0
    #
    # MacroRecall = 1.0
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
        f"Unexpected MacroRecall after batch 1: "
        f"{result_batch_1}"
    )

    print("Batch 1: PASSED")

    # --------------------------------------------------------
    # Batch 2
    # --------------------------------------------------------
    #
    # True classes:
    #   Class 0 appears once
    #   Class 1 appears 10 times
    #   Class 2 appears 10 times
    #
    # Predictions:
    #
    #   Class 0:
    #       1 sample -> predicted as Class 1
    #       TP = 0
    #       FN = 1
    #
    #   Class 1:
    #       1 sample -> predicted as Class 0
    #       9 samples -> predicted correctly
    #       TP = 9
    #       FN = 1
    #
    #   Class 2:
    #       10 samples -> predicted correctly
    #       TP = 10
    #       FN = 0
    #
    # After accumulating Batch 1:
    #
    # Class 0:
    #   TP = 1 + 0 = 1
    #   FN = 0 + 1 = 1
    #   Recall = 1 / 2 = 0.5
    #
    # Class 1:
    #   TP = 1 + 9 = 10
    #   FN = 0 + 1 = 1
    #   Recall = 10 / 11
    #
    # Class 2:
    #   TP = 1 + 10 = 11
    #   FN = 0
    #   Recall = 1.0
    #
    # Final MacroRecall:
    #
    # (0.5 + 10/11 + 1.0) / 3
    # = 0.803030...
    # --------------------------------------------------------

    y_true_batch_2 = np.array(
        [0]
        + [1] * 10
        + [2] * 10
    )

    predicted_classes_batch_2 = np.array(
        [1]
        + [0]
        + [1] * 9
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

    expected_result = (
        0.5
        + (10 / 11)
        + 1.0
    ) / 3

    actual_result = metric.result()

    assert np.isclose(actual_result, expected_result), (
        f"MacroRecall mismatch after accumulation: "
        f"expected {expected_result}, got {actual_result}"
    )

    print("Batch accumulation: PASSED")

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    metric.reset()

    assert metric.num_classes is None
    assert metric.true_positives is None
    assert metric.false_negatives is None

    print("Reset: PASSED")

    print("\nAll MacroRecall assertions passed successfully.")


if __name__ == "__main__":
    test_macro_recall()