import sys
import numpy as np
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent
sys.path.append(str(PROJECT_ROOT))

from layers import Layer_Dense
from optimizers import Adam


def test_adam():
    print("--- Running Adam Validation Test ---")

    layer = Layer_Dense(n_inputs=2, n_nodes=1)

    # Overriding initial params with fixed values
    layer.weights = np.array([[1.0], [2.0]], dtype=np.float64)
    layer.biases = np.array([[0.5]], dtype=np.float64)

    optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

    # Step 1 (t = 1)
    layer.dweights = np.array([[0.5], [1.0]], dtype=np.float64)
    layer.dbiases = np.array([[0.2]], dtype=np.float64)

    optimizer.update(layer)

    # Expected analytical updates (t = 1):
    # m_w1 = 0.1 * [0.5, 1.0] = [0.05, 0.10],   m_hat_w1 = m_w1 / (1 - 0.9^1) = [0.5, 1.0]
    # v_w1 = 0.001 * [0.25, 1.0] = [0.00025, 0.0010], v_hat_w1 = v_w1 / (1 - 0.999^1) = [0.25, 1.0]
    # W1 = [1.0, 2.0] - 0.001 * [0.5, 1.0] / (sqrt([0.25, 1.0]) + 1e-8) = [0.9990, 1.9990]
    # b1 = [0.5] - 0.001 * [0.2] / (sqrt([0.04]) + 1e-8) = [0.4990]

    expected_weights_t1 = np.array([[0.9990], [1.9990]])
    expected_biases_t1 = np.array([[0.4990]])

    assert np.allclose(layer.weights, expected_weights_t1), f"Weight mismatch at t=1: {layer.weights}"
    assert np.allclose(layer.biases, expected_biases_t1), f"Bias mismatch at t=1: {layer.biases}"
    print("Step 1 (t=1): PASSED")

    # Step 2 (t = 2)
    layer.dweights = np.array([[0.8], [-0.5]], dtype=np.float64)
    layer.dbiases = np.array([[0.1]], dtype=np.float64)

    optimizer.update(layer)

    # Expected analytical updates (t = 2):
    # m_w2 = 0.9 * [0.05, 0.10] + 0.1 * [0.8, -0.5] = [0.125, 0.04]
    # m_hat_w2 = [0.125, 0.04] / (1 - 0.9^2) = [0.65789474, 0.21052632]
    # v_w2 = 0.999 * [0.00025, 0.001] + 0.001 * [0.64, 0.25] = [0.00088975, 0.001249]
    # v_hat_w2 = v_w2 / (1 - 0.999^2) = [0.44509755, 0.62481241]
    # W2 = [0.9990, 1.9990] - 0.001 * m_hat_w2 / (sqrt(v_hat_w2) + 1e-8) = [0.99801388, 1.99873366]
    # b2 = [0.4990] - 0.001 * m_hat_b2 / (sqrt(v_hat_b2) + 1e-8) = [0.49806782]

    expected_weights_t2 = np.array([[0.99801388], [1.99873366]])
    expected_biases_t2 = np.array([[0.49806782]])

    assert np.allclose(layer.weights, expected_weights_t2), f"Weight mismatch at t=2: {layer.weights}"
    assert np.allclose(layer.biases, expected_biases_t2), f"Bias mismatch at t=2: {layer.biases}"
    print("Step 2 (t=2): PASSED")

    print("\nAll Adam assertions passed successfully.")


if __name__ == "__main__":
    test_adam()