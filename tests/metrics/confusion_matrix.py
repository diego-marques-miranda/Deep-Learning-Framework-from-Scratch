import sys
import numpy as np
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from metrics.visualization import ConfusionMatrix


def test_confusion_matrix():
    print("--- Running ConfusionMatrix Validation Test ---")

    metric = ConfusionMatrix()

    # --------------------------------------------------------
    # Batch 1
    # --------------------------------------------------------
    #
    # True classes:      0, 1, 2, 0, 1
    # Predicted classes: 0, 1, 2, 1, 2
    #
    # Expected confusion matrix:
    #
    #                 Predicted
    #                 0  1  2
    #
    # Actual 0        1  1  0
    #        1        0  1  1
    #        2        0  0  1
    #
    # --------------------------------------------------------

    y_true_batch_1 = np.array([
        0, 1, 2, 0, 1
    ])

    y_pred_batch_1 = np.array([
        [1.0, 0.0, 0.0],  # -> 0
        [0.0, 1.0, 0.0],  # -> 1
        [0.0, 0.0, 1.0],  # -> 2
        [0.0, 1.0, 0.0],  # -> 1
        [0.0, 0.0, 1.0]   # -> 2
    ])

    metric.update(y_true_batch_1, y_pred_batch_1)

    expected_matrix_batch_1 = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1]
    ])

    actual_matrix_batch_1 = metric.result()

    assert np.array_equal(
        actual_matrix_batch_1,
        expected_matrix_batch_1
    ), (
        "Confusion matrix mismatch after batch 1:\n"
        f"Expected:\n{expected_matrix_batch_1}\n"
        f"Got:\n{actual_matrix_batch_1}"
    )

    print("Batch 1: PASSED")

    # --------------------------------------------------------
    # Batch 2
    # --------------------------------------------------------
    #
    # True classes:      0, 0, 1, 2, 2
    # Predicted classes: 0, 2, 0, 2, 1
    #
    # Batch 2 matrix:
    #
    #                 Predicted
    #                 0  1  2
    #
    # Actual 0        1  0  1
    #        1        1  0  0
    #        2        0  1  1
    #
    # After accumulating Batch 1:
    #
    #                 Predicted
    #                 0  1  2
    #
    # Actual 0        2  1  1
    #        1        1  1  1
    #        2        0  1  2
    #
    # --------------------------------------------------------

    y_true_batch_2 = np.array([
        0, 0, 1, 2, 2
    ])

    y_pred_batch_2 = np.array([
        [1.0, 0.0, 0.0],  # -> 0
        [0.0, 0.0, 1.0],  # -> 2
        [1.0, 0.0, 0.0],  # -> 0
        [0.0, 0.0, 1.0],  # -> 2
        [0.0, 1.0, 0.0]   # -> 1
    ])

    metric.update(y_true_batch_2, y_pred_batch_2)

    expected_matrix_final = np.array([
        [2, 1, 1],
        [1, 1, 1],
        [0, 1, 2]
    ])

    actual_matrix_final = metric.result()

    assert np.array_equal(
        actual_matrix_final,
        expected_matrix_final
    ), (
        "Confusion matrix mismatch after accumulation:\n"
        f"Expected:\n{expected_matrix_final}\n"
        f"Got:\n{actual_matrix_final}"
    )

    print("Batch accumulation: PASSED")

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    metric.reset()

    assert metric.num_classes is None
    assert metric.matrix is None

    print("Reset: PASSED")

    print("\nAll ConfusionMatrix assertions passed successfully.")


if __name__ == "__main__":
    test_confusion_matrix()