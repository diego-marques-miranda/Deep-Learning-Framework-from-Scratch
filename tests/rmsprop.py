import sys
import numpy as np
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent
sys.path.append(str(PROJECT_ROOT))

from layers import Layer_Dense
from optimizers import RMSprop


def test_rmsprop():
    print("--- Running RMSprop Validation Test ---")

    layer = Layer_Dense(n_inputs=2, n_nodes=1)

    # Overriding initial params with fixed values
    layer.weights = np.array([[1.0], [2.0]], dtype=np.float64)
    layer.biases = np.array([[0.5]], dtype=np.float64)

    optimizer = RMSprop(learning_rate=0.01, beta=0.9)

    # Step 1 (t = 1)
    layer.dweights = np.array([[0.5], [1.0]], dtype=np.float64)
    layer.dbiases = np.array([[0.2]], dtype=np.float64)

    optimizer.update(layer)

    # Expected analytical updates (t = 1):
    # cache_w1 = 0.9 * 0 + 0.1 * [0.5^2, 1.0^2] = [0.025, 0.100]
    # cache_b1 = 0.9 * 0 + 0.1 * [0.2^2]       = [0.004]
    # W1 = [1.0, 2.0] - 0.01 * [0.5, 1.0] / (sqrt([0.025, 0.100]) + 1e-8) = [0.96837723, 1.96837722]
    # b1 = [0.5] - 0.01 * [0.2] / (sqrt([0.004]) + 1e-8)                = [0.46837723]

    expected_weights_t1 = np.array([[0.96837723], [1.96837722]])
    expected_biases_t1 = np.array([[0.46837723]])

    assert np.allclose(layer.weights, expected_weights_t1), f"Weight mismatch at t=1: {layer.weights}"
    assert np.allclose(layer.biases, expected_biases_t1), f"Bias mismatch at t=1: {layer.biases}"
    print("Step 1 (t=1): PASSED")

    # Step 2 (t = 2)
    layer.dweights = np.array([[0.8], [-0.5]], dtype=np.float64)
    layer.dbiases = np.array([[0.1]], dtype=np.float64)

    optimizer.update(layer)

    # Expected analytical updates (t = 2):
    # cache_w2 = 0.9 * [0.025, 0.100] + 0.1 * [0.8^2, (-0.5)^2] = [0.0865, 0.1150]
    # cache_b2 = 0.9 * [0.004]        + 0.1 * [0.1^2]         = [0.0046]
    # W2 = [0.96837723, 1.96837722] - 0.01 * [0.8, -0.5] / (sqrt([0.0865, 0.1150]) + 1e-8) = [0.94117641, 1.98312142]
    # b2 = [0.46837723] - 0.01 * [0.1] / (sqrt([0.0046]) + 1e-8)                          = [0.45363303]

    expected_weights_t2 = np.array([[0.94117641], [1.98312142]])
    expected_biases_t2 = np.array([[0.45363303]])

    assert np.allclose(layer.weights, expected_weights_t2), f"Weight mismatch at t=2: {layer.weights}"
    assert np.allclose(layer.biases, expected_biases_t2), f"Bias mismatch at t=2: {layer.biases}"
    print("Step 2 (t=2): PASSED")

    print("\nAll RMSprop assertions passed successfully.")


if __name__ == "__main__":
    test_rmsprop()