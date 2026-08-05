import sys
import numpy as np
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent
sys.path.append(str(PROJECT_ROOT))

from layers import Layer_Dense
from optimizers import SGD_Momentum


def test_sgd_momentum():
    print("--- Running SGD Momentum Validation Test ---")

    layer = Layer_Dense(n_inputs=2, n_nodes=1)
    
    # Overriding initial params with fixed values
    layer.weights = np.array([[1.0], [2.0]], dtype=np.float64)
    layer.biases = np.array([[0.5]], dtype=np.float64)

    optimizer = SGD_Momentum(learning_rate=0.01, beta=0.9)

    # Step 1 (t = 1)
    layer.dweights = np.array([[0.5], [1.0]], dtype=np.float64)
    layer.dbiases = np.array([[0.2]], dtype=np.float64)

    optimizer.update(layer)

    # Expected analytical updates (t = 1):
    # v_w1 = 0.9 * 0 + 0.1 * [0.5, 1.0] = [0.05, 0.10]
    # v_b1 = 0.9 * 0 + 0.1 * 0.2        = [0.02]
    # W1   = [1.0, 2.0] - 0.01 * [0.05, 0.10] = [0.9995, 1.9990]
    # b1   = [0.5] - 0.01 * [0.02]            = [0.4998]

    expected_weights_t1 = np.array([[0.9995], [1.9990]])
    expected_biases_t1 = np.array([[0.4998]])

    assert np.allclose(layer.weights, expected_weights_t1), f"Weight mismatch at t=1: {layer.weights}"
    assert np.allclose(layer.biases, expected_biases_t1), f"Bias mismatch at t=1: {layer.biases}"
    print("Step 1 (t=1): PASSED")

    # Step 2 (t = 2)
    layer.dweights = np.array([[0.8], [-0.5]], dtype=np.float64)
    layer.dbiases = np.array([[0.1]], dtype=np.float64)

    optimizer.update(layer)

    # Expected analytical updates (t = 2):
    # v_w2 = 0.9 * [0.05, 0.10] + 0.1 * [0.8, -0.5] = [0.125, 0.04]
    # v_b2 = 0.9 * [0.02]       + 0.1 * [0.1]       = [0.028]
    # W2   = [0.9995, 1.9990] - 0.01 * [0.125, 0.04]   = [0.99825, 1.99860]
    # b2   = [0.4998] - 0.01 * [0.028]                = [0.49952]

    expected_weights_t2 = np.array([[0.99825], [1.99860]])
    expected_biases_t2 = np.array([[0.49952]])

    assert np.allclose(layer.weights, expected_weights_t2), f"Weight mismatch at t=2: {layer.weights}"
    assert np.allclose(layer.biases, expected_biases_t2), f"Bias mismatch at t=2: {layer.biases}"
    print("Step 2 (t=2): PASSED")

    print("\nAll SGD Momentum assertions passed successfully.")


if __name__ == "__main__":
    test_sgd_momentum()