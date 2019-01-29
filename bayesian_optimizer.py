import warnings
from typing import Sequence, Tuple

import numpy as np
import scipy.optimize as sp_optimize
import scipy.special as sp_special

from .gaussian_process import GaussianProcess
from .kernel import Kernel
from .sobol import new_sobol_sequence_generator


class BayesianOptimizer:
    def __init__(self, kernel: Kernel, sigma: float, input_bounds: Sequence[Tuple[float, float]]):
        """

        :param kernel:
        :param sigma:
        :param input_bounds:
        """
        self.kernel = kernel
        self.sigma = sigma
        self.input_bounds = np.array(input_bounds)
        self.sequence_generator = new_sobol_sequence_generator()

    def calculate_expected_improvement(self, candidates: np.ndarray, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """

        :param candidates:
        :param inputs:
        :param targets:
        :return:
        """
        means, covariances = GaussianProcess.predict(candidates, self.kernel, self.sigma, inputs, targets)

        best = np.min(targets, axis=0)

        standard_deviations = np.sqrt(np.diag(covariances))[:, np.newaxis]

        z_scores = (best - means) / standard_deviations

        cdf = 0.5 * (1.0 + sp_special.erf(z_scores / np.sqrt(2)))

        pdf = np.exp(-0.5 * z_scores ** 2) / np.sqrt(2 * np.pi)

        ei = (best - means) * cdf + standard_deviations * pdf

        ei_target_sum = np.sum(ei, axis=-1)

        return ei_target_sum

    def calculate_expected_improvement_input_gradients(self, candidates: np.ndarray, inputs: np.ndarray, targets: np.ndarray):
        """

        :param candidates:
        :param inputs:
        :param targets:
        :return:
        """
        means, covariances = GaussianProcess.predict(candidates, self.kernel, self.sigma, inputs, targets)

        mean_gradients, covariance_gradients = GaussianProcess.calculate_input_gradients(candidates, self.kernel, self.sigma, inputs, targets)

        best = np.min(targets, axis=0)

        standard_deviations = np.sqrt(np.diag(covariances))[:, np.newaxis]

        z_scores = ((best - means) / standard_deviations)

        cdf = 0.5 * (1.0 + sp_special.erf(z_scores / np.sqrt(2)))

        pdf = np.exp(-0.5 * z_scores ** 2) / np.sqrt(2 * np.pi)

        variance_gradients = np.diagonal(covariance_gradients).T

        gradients = -1. * mean_gradients * cdf[:, np.newaxis, :]
        gradients += 0.5 * pdf[:, np.newaxis, :] * variance_gradients[:, np.newaxis, :] / standard_deviations[:, np.newaxis, :]

        gradients_target_sum = np.sum(gradients, axis=1)

        return gradients_target_sum

    def generate_candidates(self, num_features: int, num_points: int) -> np.ndarray:
        """
        Get candidates with `num_points` for `num_features`.
        Uses Sobol sequences to generate properly distributed points.
        Sobol sequences lie in the unit hypercube so we scale them using bound configuration.
        :param num_features:
        :param num_points:
        :return:
        """
        points = self.sequence_generator.generate_sequence(num_features, num_points)

        scaled_points = points * (self.input_bounds[1] - self.input_bounds[0]) - self.input_bounds[0]

        return scaled_points

    def optimize_candidate(self, initial_candidate: np.ndarray, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """

        :param initial_candidate:
        :param inputs:
        :param targets:
        :return:
        """
        assert initial_candidate.ndim == 1

        def objective_function(candidate):
            expected_improvement = self.calculate_expected_improvement(candidate[np.newaxis, :], inputs, targets)

            gradients = self.calculate_expected_improvement_input_gradients(candidate[np.newaxis, :], inputs, targets)

            return -expected_improvement, -np.squeeze(gradients, axis=0)

        optimized_candidate, _, info = sp_optimize.fmin_l_bfgs_b(
            objective_function,
            initial_candidate,
            bounds=self.input_bounds)

        if info['warnflag'] != 0:
            warnings.warn(f'fmin_l_bfgs_b terminated abnormally with the state: {info}')

        return optimized_candidate

    def suggest(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            num_candidates: int = 1e4,
            num_optimized_candidates: int = 20) -> np.ndarray:
        """

        :param inputs:
        :param targets:
        :param num_candidates:
        :param num_optimized_candidates:
        :return:
        """
        _, num_features = np.shape(inputs)

        initial_candidates = self.generate_candidates(num_features, num_candidates)

        initial_acquisitions = self.calculate_expected_improvement(initial_candidates, inputs, targets)

        best_candidates = initial_candidates[np.argsort(initial_acquisitions)[-num_optimized_candidates:]]

        optimized_candidates = [self.optimize_candidate(candidate, inputs, targets) for candidate in best_candidates]

        candidates = np.concatenate([best_candidates, np.stack(optimized_candidates, axis=0)], axis=0)

        acquisitions = self.calculate_expected_improvement(candidates, inputs, targets)

        suggestion = candidates[np.argmax(acquisitions)]

        return suggestion
