import math
from typing import Tuple, Optional

import numpy as np

from .miscellaneous_math import Math


class Kernel:
    def __call__(self, x_matrix: np.ndarray, y_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the covariances under the kernel between all vectors in `x_matrix` and `y_matrix`, or between all pairs in `x_matrix`.
        :param x_matrix: A matrix of size (number of vectors, number of dimensions).
        :param y_matrix: A matrix of size (number of vectors, number of dimensions).
        :return: A matrix of size (number of points in `x_matrix`, number of points in `y_matrix`), or a squared matrix of number of points in `x_matrix`.
        """
        raise NotImplementedError

    def calculate_input_gradients(self, x_matrix: np.ndarray, y_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the gradients of the kernel function with respect to each vector dimension of `x_matrix`.
        :param x_matrix: A matrix of size (number of vectors, number of dimensions).
        :param y_matrix: A matrix of size (number of vectors, number of dimensions).
        :return: A three-dimensional tensor of size (number of points in `x_matrix`, number of points in `y_matrix`, number of dimensions) with the derivatives.
        """
        raise NotImplementedError

    def calculate_hyperparameter_gradients(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the gradients of the kernel evaluates on `matrix` with respect to the hyperparameters.
        We only worry about the case where we compute the kernels with a single matrix (all the combinations of `matrix`)
        :param matrix: A matrix of size (number of vectors, number of dimensions).
        :return: A three-dimensional tensor of size (number of vectors, number of vectors, number of hyperparameters) with the derivatives.
        """
        raise NotImplementedError

    def get_hyperparameters(self) -> np.ndarray:
        """
        Obtains a vector representation of the hyperparameters values.
        :return: A vector of hyperparameter values.
        """
        raise NotImplementedError

    def get_hyperparameter_bounds(self) -> np.ndarray:
        """
        Obtains the bounds for each hyperparameter.
        Must be aligned with `get_hyperparameters`.
        :return: A matrix of size (number of hyperparameters, 2) with the lower and upper bounds for each hyperparameter.
        """
        raise NotImplementedError

    @staticmethod
    def from_hyperparameters(hyperparameters: np.ndarray) -> 'Kernel':
        """
        Creates a kernel from a vector representing hyperparameters.
        :hyperparameters: A vector of hyperparameter values.
        :return: A subclass of `Kernel`.
        """
        raise NotImplementedError


class Matern(Kernel):
    def __init__(self, length_scales: np.ndarray, constant: Optional[float] = None, bounds: Optional[Tuple[float, float]] = None):
        """
        An ARD Matern 3/2 kernel.
        :param length_scales: A vector with the length scale for each input dimension.
        :param constant: A constant value multiplying the kernel
        :param bounds: The minimum and maximum value the hyperparameters can take.
        """
        self.length_scales = length_scales
        self.constant = constant if constant else 1.0
        self.bounds = bounds if bounds else (1e-5, 1e5)

    def __call__(self, x_matrix: np.ndarray, y_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        assert x_matrix.shape[1] == self.length_scales.size and (y_matrix is None or y_matrix.shape[1] == self.length_scales.size)

        x_scaled = x_matrix / self.length_scales

        y_scaled = y_matrix / self.length_scales if y_matrix is not None else x_scaled

        distances = Math.euclidean(x_scaled, y_scaled)

        k_matrix = distances * math.sqrt(3)
        k_matrix = self.constant * (1. + k_matrix) * np.exp(-k_matrix)

        return k_matrix

    def calculate_input_gradients(self, x_matrix: np.ndarray, y_matrix: np.ndarray) -> np.ndarray:
        scaled_distances = Math.euclidean(x_matrix / self.length_scales, y_matrix / self.length_scales)
        scaled_distance_gradients = Math.subtractions(x_matrix, y_matrix) / (self.length_scales ** 2)

        kernel_derivative = -3 * self.constant * np.exp(-np.sqrt(3) * scaled_distances)
        kernel_derivative = kernel_derivative[:, :, np.newaxis] * scaled_distance_gradients

        return kernel_derivative

    def _calculate_constant_gradient(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the kernel evaluated on matrix with respect to the kernel constant.
        :param matrix: A matrix of size (number of vectors, number of dimensions).
        :return: A squared matrix of size (number of vectors, number of vectors).
        """
        scaled = matrix / self.length_scales

        distances = Math.euclidean(scaled, scaled)

        gradient = distances * math.sqrt(3)

        gradient = (1. + gradient) * np.exp(-gradient)

        return gradient

    def _calculate_length_scale_gradient(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the kernel evaluated on matrix with respect to each scale length.
        :param matrix: A matrix of size (number of vectors, number of dimensions).
        :return: A three-dimensional tensor of size (number of vectors, number of vectors, number of vector dimensions).
        """
        distances = (matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]) ** 2 / (self.length_scales ** 2)

        derivative = self.constant * 3 * distances * np.exp(-np.sqrt(3 * distances.sum(-1)))[..., np.newaxis]

        return derivative

    def calculate_hyperparameter_gradients(self, matrix: np.ndarray) -> np.ndarray:
        assert matrix.shape[1] == self.length_scales.size

        constant_derivative = self._calculate_constant_gradient(matrix)

        length_scale_derivative = self._calculate_length_scale_gradient(matrix)

        return np.concatenate((constant_derivative[..., np.newaxis], length_scale_derivative), axis=2)

    def get_hyperparameters(self) -> np.ndarray:
        return np.concatenate(([self.constant], self.length_scales))

    def get_hyperparameter_bounds(self) -> np.ndarray:
        return np.tile(self.bounds, (self.length_scales.size + 1, 1))

    @staticmethod
    def from_hyperparameters(hyperparameters) -> 'Matern':
        return Matern(hyperparameters[1:], hyperparameters[0])
