import sys
import numpy as np
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from metrics.metrics import MacroF1


def test_macro_f1():
    print("--- Running MacroF1 Validation Test ---")

    metric = MacroF1()

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
    # FN = 0
    #
    # F1 per class:
    # Class 0 = 1.0
    # Class 1 = 1.0
    # Class 2 = 1.0
    #
    # MacroF1 = 1.0
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
        f"Unexpected MacroF1 after batch 1: "
        f"{result_batch_1}"
    )

    print("Batch 1: PASSED")

    # --------------------------------------------------------
    # Batch 2
    # --------------------------------------------------------
    #
    # True classes:
    #   Class 0 appears once
    #   Class 1 appears twice
    #   Class 2 appears three times
    #
    # Predictions:
    #
    #   Samples:
    #       True 0 -> Pred 1
    #       True 1 -> Pred 1
    #       True 1 -> Pred 0
    #       True 2 -> Pred 2
    #       True 2 -> Pred 0
    #       True 2 -> Pred 2
    #
    # Batch 2 counts:
    #
    # Class 0:
    #   TP = 0
    #   FP = 2
    #   FN = 1
    #
    # Class 1:
    #   TP = 1
    #   FP = 1
    #   FN = 1
    #
    # Class 2:
    #   TP = 2
    #   FP = 0
    #   FN = 1
    #
    # After accumulating Batch 1:
    #
    # Class 0:
    #   TP = 1
    #   FP = 2
    #   FN = 1
    #
    #   F1 = 2*1 / (2*1 + 2 + 1)
    #      = 2/5
    #
    # Class 1:
    #   TP = 2
    #   FP = 1
    #   FN = 1
    #
    #   F1 = 2*2 / (2*2 + 1 + 1)
    #      = 4/6
    #      = 2/3
    #
    # Class 2:
    #   TP = 3
    #   FP = 0
    #   FN = 1
    #
    #   F1 = 2*3 / (2*3 + 0 + 1)
    #      = 6/7
    #
    # Final MacroF1:
    #
    # (2/5 + 2/3 + 6/7) / 3
    # = 0.682539...
    # --------------------------------------------------------

    y_true_batch_2 = np.array(
        [0, 1, 1, 2, 2, 2]
    )

    predicted_classes_batch_2 = np.array(
        [1, 1, 0, 2, 0, 2]
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
        (2 / 5)
        + (2 / 3)
        + (6 / 7)
    ) / 3

    actual_result = metric.result()

    assert np.isclose(actual_result, expected_result), (
        f"MacroF1 mismatch after accumulation: "
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
    assert metric.false_negatives is None

    print("Reset: PASSED")

    print("\nAll MacroF1 assertions passed successfully.")


if __name__ == "__main__":
    test_macro_f1()