import unittest

import numpy as np

from .bayesian_optimizer import BayesianOptimizer
from .gaussian_process import GaussianProcess
from .kernel import Matern


class TestGaussianProcess(unittest.TestCase):
    @staticmethod
    def objective(x: np.ndarray) -> np.ndarray:
        return -0.5 * np.exp(-0.5 * (x - 2) ** 2) - 0.5 * np.exp(-0.5 * (x + 2.1) ** 2 / 5) + 0.3

    def test_expected_improvement_input_gradients(self):
        inputs = np.random.random((4, 2))
        targets = np.array([[10.], [5.], [1.], [3.]])

        length_scales = np.array([0.5, 0.1])
        constant = 0.5
        sigma = 1e-10
        candidate_bounds = [[-5, 5], [-5, 5]]

        kernel = Matern(length_scales, constant)
        optimized_kernel = GaussianProcess.optimize_kernel(kernel, inputs, targets, sigma)
        bayesian_optimizer = BayesianOptimizer(optimized_kernel, sigma, candidate_bounds)

        candidates = np.random.random((3, 2))

        gradients = bayesian_optimizer.calculate_expected_improvement_input_gradients(candidates, inputs, targets)

        self.assertEqual(gradients.shape, (3, 2))

    def test_expected_improvement_optimization(self):
        inputs = np.linspace(-5, 5, 10 + 1)[1:][:, np.newaxis]
        targets = self.objective(inputs)

        length_scales = np.array([1.0])
        constant = 1.0
        sigma = 1e-10
        candidate_bounds = [(-20., 20.)]

        kernel = Matern(length_scales, constant)
        optimized_kernel = GaussianProcess.optimize_kernel(kernel, inputs, targets, sigma)
        bayesian_optimizer = BayesianOptimizer(optimized_kernel, sigma, candidate_bounds)

        candidate = np.array([-10.])
        candidate_ei = bayesian_optimizer.calculate_expected_improvement(candidate[np.newaxis, :], inputs, targets)[0]

        optimized_candidate = bayesian_optimizer.optimize_candidate(candidate, inputs, targets)
        optimized_candidate_ei = bayesian_optimizer.calculate_expected_improvement(optimized_candidate[np.newaxis, :], inputs, targets)[0]

        self.assertGreater(optimized_candidate_ei, candidate_ei)
