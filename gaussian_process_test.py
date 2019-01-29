import unittest

import numpy as np

from .gaussian_process import GaussianProcess
from .kernel import Matern


class TestGaussianProcess(unittest.TestCase):
    @staticmethod
    def objective(x: np.ndarray) -> np.ndarray:
        return -0.5 * np.exp(-0.5 * (x - 2) ** 2) - 0.5 * np.exp(-0.5 * (x + 2.1) ** 2 / 5) + 0.3

    def test_predictions(self):
        inputs = np.linspace(-5, 5, 10 + 1)[1:][:, np.newaxis]
        targets = self.objective(inputs)

        length_scales = np.array([0.5])
        constant = 0.5
        sigma = 1e-10

        kernel = Matern(length_scales, constant)

        optimized_kernel = GaussianProcess.optimize_kernel(kernel, inputs, targets, sigma)

        good_input = inputs[np.argmax(targets[:, 0]), np.newaxis] + 1e-5
        bad_input = inputs[np.argmin(targets[:, 0]), np.newaxis] + 1e-5

        good_prediction, _ = GaussianProcess.predict(good_input, optimized_kernel, sigma, inputs, targets)
        bad_prediction, _ = GaussianProcess.predict(bad_input, optimized_kernel, sigma, inputs, targets)

        self.assertGreater(good_prediction, bad_prediction)

    def test_kernel_optimization(self):
        inputs = np.linspace(-5, 5, 10 + 1)[1:][:, np.newaxis]
        targets = self.objective(inputs)

        length_scales = np.array([0.5])
        constant = 0.5
        sigma = 1e-10

        kernel = Matern(length_scales, constant)

        likelihood = GaussianProcess.log_marginal_likelihood(kernel, inputs, targets, sigma)

        optimized_kernel = GaussianProcess.optimize_kernel(kernel, inputs, targets, sigma)

        optimized_likelihood = GaussianProcess.log_marginal_likelihood(optimized_kernel, inputs, targets, sigma)

        self.assertGreater(optimized_likelihood, likelihood)

    def test_kernel_optimization_multiple_targets(self):
        inputs = np.linspace(-5, 5, 10 + 1)[1:][:, np.newaxis]
        objective = self.objective(inputs)
        targets = np.concatenate([objective, 1 - objective, 5. * objective], axis=-1)

        length_scales = np.array([0.5])
        constant = 0.5
        sigma = 1e-10

        kernel = Matern(length_scales, constant)

        likelihood = GaussianProcess.log_marginal_likelihood(kernel, inputs, targets, sigma)

        optimized_kernel = GaussianProcess.optimize_kernel(kernel, inputs, targets, sigma)

        optimized_likelihood = GaussianProcess.log_marginal_likelihood(optimized_kernel, inputs, targets, sigma)

        self.assertGreater(optimized_likelihood, likelihood)

    def test_input_gradients(self):
        inputs = np.linspace(-5, 5, 10 + 1)[1:][:, np.newaxis]
        targets = self.objective(inputs)

        length_scales = np.array([0.5])
        constant = 0.5
        sigma = 1e-10

        kernel = Matern(length_scales, constant)

        mean_gradients, covariance_gradients = GaussianProcess.calculate_input_gradients(
            inputs[2, np.newaxis],
            kernel,
            sigma,
            inputs,
            targets)

        self.assertEqual(mean_gradients.shape, (1, 1, 1))
        self.assertEqual(covariance_gradients.shape, (1, 1, 1))
