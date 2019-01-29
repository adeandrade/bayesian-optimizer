import numba
import numpy as np


class Math:
    @staticmethod
    @numba.njit
    def euclidean(x_matrix: np.ndarray, y_matrix: np.ndarray) -> np.ndarray:
        """
        Computes the euclidean distance between all vectors in `x_matrix` and all vectors in `y_matrix`.
        :param x_matrix: A matrix of size (number of vectors, number of dimensions).
        :param y_matrix: A matrix of size (number of vectors, number of dimensions).
        :return: A matrix of size (number of vectors in `x_matrix`, number of vectors in `y_matrix`).
        """
        distances = np.empty((x_matrix.shape[0], y_matrix.shape[0]), dtype=x_matrix.dtype)

        for index in range(x_matrix.shape[0]):
            distances[index, :] = np.sqrt(np.sum(np.square(y_matrix - x_matrix[index]), axis=1))

        return distances

    @staticmethod
    @numba.njit
    def subtractions(x_matrix: np.ndarray, y_matrix: np.ndarray) -> np.ndarray:
        """
        Computes the x - y for all vectors between `x_matrix` and `y_matrix`.
        :param x_matrix: A matrix of size (number of vectors, number of dimensions).
        :param y_matrix: A matrix of size (number of vectors, number of dimensions).
        :return: A tensor of size (number of vectors in `x_matrix`, number of vectors in `y_matrix`, number of dimensions).
        """
        subtractions = np.empty((x_matrix.shape[0], y_matrix.shape[0], x_matrix.shape[1]), dtype=x_matrix.dtype)

        for index in range(x_matrix.shape[0]):
            subtractions[index, :, :] = x_matrix[index] - y_matrix

        return subtractions
