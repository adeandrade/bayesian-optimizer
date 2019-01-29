import warnings
from typing import Optional, TypeVar, Union

import numpy as np
import scipy.linalg as sp_linalg
import scipy.optimize as sp_optimize

from .kernel import Kernel, Tuple, Matern

KernelType = TypeVar('KernelType', Kernel, Matern)


class GaussianProcess:
    """
    Functions for a gaussian process regressor.
    Assumes the mean prior is zero.
    Based on Algorithm 2.1 of "Gaussian Processes for Machine Learning" (C. Rasmussen, C. Williams, R. Sutton et al., 2006).
    Gradients of the likelihood with respect to the kernel hyperparameters is Equation 5.9 of the same document.
    """
    @staticmethod
    def _decompose(kernel: Kernel, inputs: np.ndarray, targets: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Cholesky representation of the kernel evaluated on the inputs (`l_matrix`) and
        the matrix multiplication of the inverse of the distance matrix and the `targets` (`alpha`).
        :param kernel: An object of type `Kernel` representing the kernel.
        :param inputs: A matrix of size (number of vectors, number of dimensions) representing the inputs of the model.
        :param targets: A matrix of size (number of vectors, number of targets) representing the target values of the model.
        :param sigma: Spherical covariance to add to the kernel covariance function for stability.
        :return: A tuple of numpy arrays with the Cholesky matrix and the alpha vector respectively.
        """
        distances = kernel(inputs)

        distances[np.diag_indices_from(distances)] += sigma

        l_matrix = sp_linalg.cholesky(distances, lower=True)

        alpha = sp_linalg.cho_solve((l_matrix, True), targets)

        return l_matrix, alpha

    @staticmethod
    def predict(
            test_inputs: np.ndarray,
            kernel: Kernel,
            sigma: float,
            inputs: Optional[np.ndarray] = None,
            targets: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the means of each `input_test` vector and the covariances between all vectors in `input_test`.
        Handles the case where data is not present.
        :param test_inputs: A matrix of size (number of vectors, number of dimensions) representing the testing inputs of the model.
        :param kernel: An object of type `Kernel` representing the kernel.
        :param sigma: Spherical covariance to add to the kernel covariance function for stability.
        :param inputs: An optional matrix of size (number of vectors, number of dimensions) representing the inputs of the model.
        :param targets: An optional matrix of size (number of vectors, number of targets) representing the target values of the model.
        :return: A tuple with a matrix of size (number of vectors in `test_inputs`, number of targets) representing means
        and a matrix of size (size number of vectors in `test_inputs`, size number of vectors in `test_inputs`) with
        covariances between all vectors.
        """
        if inputs is None or targets is None:
            mean = np.zeros(test_inputs.shape[0], dtype=test_inputs.dtype)
            covariance = kernel(test_inputs, test_inputs)

            return mean, covariance

        l_matrix, alpha = GaussianProcess._decompose(kernel, inputs, targets, sigma)

        cross_distances = kernel(test_inputs, inputs)

        v = sp_linalg.cho_solve((l_matrix, True), cross_distances.T)

        means = cross_distances.dot(alpha)

        covariances = kernel(test_inputs) - cross_distances.dot(v)

        return means, covariances

    @staticmethod
    def calculate_input_gradients(
            input_test: np.ndarray,
            kernel: Kernel,
            sigma: float,
            inputs: np.ndarray,
            targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the gradients of the mean function and the covariance function with respect to `input_test`.
        :param input_test: A matrix of size (number of vectors, number of dimensions) representing the testing inputs of the model.
        :param kernel: An object of type `Kernel` representing the kernel.
        :param sigma: Spherical covariance to add to the kernel covariance function for stability.
        :param inputs: A matrix of size (number of vectors, number of dimensions) representing the inputs of the model.
        :param targets: A matrix of size (number of vectors, number of targets) representing the target values of the model.
        :return: A tuple with a tensor of size (number of vectors in `input_test`, number of targets, number of dimensions)
        representing means and a tensor of size (number of vectors in `input_test`, number of vectors in `input_test`, number of dimensions)
        representing covariances.
        """
        l_matrix, alpha = GaussianProcess._decompose(kernel, inputs, targets, sigma)

        distance_gradients = kernel.calculate_input_gradients(input_test, inputs)

        cross_distances = kernel(input_test, inputs)

        v = sp_linalg.cho_solve((l_matrix, True), cross_distances.T)

        mean_gradients = np.transpose(np.tensordot(distance_gradients, alpha, [1, 0]), [0, 2, 1])

        covariance_gradients = -2. * np.tensordot(v, distance_gradients, axes=[0, 1])

        return mean_gradients, covariance_gradients

    @staticmethod
    def log_marginal_likelihood(
            kernel: Kernel,
            inputs: np.ndarray,
            targets: np.ndarray,
            sigma: float,
            eval_gradient=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Computes the log-likelihood of the `targets` given the `inputs` and the `kernel`.
        The flag `eval_gradient` will optionally return the gradients with respect to the model hyperparameters.
        :param kernel: An object of type `Kernel` representing the kernel.
        :param inputs: A matrix of size (number of vectors, number of dimensions) representing the inputs of the model.
        :param targets: A matrix of size (number of vectors, number of targets) representing the target values of the model.
        :param sigma: Spherical covariance to add to the kernel covariance function for stability.
        :param eval_gradient: A boolean variable to specify whether to return gradients or not.
        :return: A scalar value representing the log-likelihood and optionally a vector of gradients for each hyperparameter.
        """
        num_samples = inputs.shape[0]

        l_matrix, alpha = GaussianProcess._decompose(kernel, inputs, targets, sigma)

        log_likelihood = -0.5 * np.einsum('ik,ik->k', targets, alpha)
        log_likelihood -= np.sum(np.log(np.diag(l_matrix)))
        log_likelihood -= num_samples / 2. * np.log(2. * np.pi)
        log_likelihood = np.sum(log_likelihood, axis=-1)

        if not eval_gradient:
            return log_likelihood

        kernel_gradients = kernel.calculate_hyperparameter_gradients(inputs)

        log_likelihood_gradient = alpha[:, np.newaxis] * alpha
        log_likelihood_gradient -= sp_linalg.cho_solve((l_matrix, True), np.eye(num_samples))[:, :, np.newaxis]
        log_likelihood_gradient = 0.5 * np.einsum('ijl,ijk->kl', log_likelihood_gradient, kernel_gradients)
        log_likelihood_gradient = np.sum(log_likelihood_gradient, axis=-1)

        return log_likelihood, log_likelihood_gradient

    @staticmethod
    def optimize_kernel(kernel: KernelType, inputs: np.ndarray, targets: np.ndarray, sigma: float) -> KernelType:
        """
        Optimize kernel hyperparameters to maximize the gaussian process likelihood.
        :param kernel: A subclass of `Kernel`.
        :param inputs: A matrix of size (number of vectors, number of dimensions) representing the inputs of the model.
        :param targets: A matrix of size (number of vectors, number of targets) representing the target values of the model.
        :param sigma: Spherical covariance to add to the kernel covariance function for stability.
        :return: A new kernel of type `KernelType` with the optimized hyperparameters.
        """
        def objective_function(hyperparameters):
            new_kernel = kernel.from_hyperparameters(hyperparameters)

            likelihood, gradient = GaussianProcess.log_marginal_likelihood(new_kernel, inputs, targets, sigma, eval_gradient=True)

            return -likelihood, -gradient

        optimized_hyperparameters, _, info = sp_optimize.fmin_l_bfgs_b(
            objective_function,
            kernel.get_hyperparameters(),
            bounds=kernel.get_hyperparameter_bounds())

        if info['warnflag'] != 0:
            warnings.warn(f'fmin_l_bfgs_b terminated abnormally with the state: {info}')

        return kernel.from_hyperparameters(optimized_hyperparameters)
