import unittest

import numpy as np

from .kernel import Matern
from .miscellaneous_math import Math


class TestMaternKernel(unittest.TestCase):
    def test_euclidean_distance_calculation(self):
        x = np.array([[5.0, 2.0], [9.0, 3.0]])
        y = np.array([[8.0, 3.0], [5.0, 12.0]])

        distances = Math.euclidean(x, y)
        expected_distances = np.array([[3.162278, 10], [1, 9.848858]])

        np.testing.assert_array_almost_equal(distances, expected_distances, )

    def test_kernel_covariance_calculation(self):
        length_scales = np.array([0.1, 0.5])

        kernel = Matern(length_scales, constant=0.5)

        x = np.array([[5.0, 2.0], [9.0, 3.0]])
        y = np.array([[8.0, 3.0], [5.0, 12.0]])

        covariances = kernel(x, y)
        expected_covariances = np.array([[6.414748e-22, 1.608859e-14], [2.752368e-07, 3.893699e-32]])

        np.testing.assert_array_almost_equal(covariances, expected_covariances)

    def test_kernel_hyperparameter_gradients(self):
        length_scales = np.array([0.1, 0.5, 0.8])

        kernel = Matern(length_scales, constant=0.5)

        x = np.array([[5.0, 2.0, 3.0], [9.0, 3.0, 5.0]])

        gradients = kernel.calculate_hyperparameter_gradients(x)

        self.assertEqual(gradients.shape, (2, 2, 4))

    def test_kernel_input_gradients(self):
        length_scales = np.array([0.1, 0.5, 0.8])

        kernel = Matern(length_scales, constant=0.5)

        x = np.array([[5.0, 2.0, 3.0], [9.0, 3.0, 5.0]])
        y = np.array([[8.0, 3.0, 5.0], [5.0, 12.0, 6.0], [7.0, 2.0, 9.0]])

        gradients = kernel.calculate_input_gradients(x, y)

        self.assertEqual(gradients.shape, (2, 3, 3))
