import os

import numba
import numpy as np


@numba.jitclass([
    ('accumulated_degrees', numba.uint32[:]),
    ('coefficients', numba.uint32[:]),
    ('direction_numbers', numba.uint32[:])])
class SobolSequenceGenerator:
    def __init__(self, accumulated_degrees, coefficients, direction_numbers):
        """
        A Sobol sequence generator that produces points that covers an unit hypercube evenly.
        Based on "Notes on Generating Sobol Sequences" (S. Joe, F.Y. Kuo, 2008).
        Primitive polynomials and direction numbers produced by the same authors: http://www.maths.unsw.edu.au/~fkuo/sobol
        :param accumulated_degrees: A vector of accumulated polynomial degrees.
        :param coefficients: An integer representation of the primitive polynomials.
        :param direction_numbers: A vector of direction numbers sorted by polynomial and polynomial coefficients.
        """
        self.accumulated_degrees = accumulated_degrees
        self.coefficients = coefficients
        self.direction_numbers = direction_numbers

    def calculate_direction_numbers(self, num_dimensions: int, num_points: int) -> np.ndarray:
        """
        Calculate direction numbers for all dimensions.
        Each dimension uses a different primitive polynomial.
        We select them from the available ones incrementally.
        :param num_dimensions: The number of dimensions to consider.
        :param num_points: The number of points to obtain.
        :return: A matrix of size (number of dimensions, number of bits) representing direction numbers.
        """
        num_bits = int(np.ceil(np.log2(num_points)))

        directions_numbers = np.empty((num_dimensions, num_bits), dtype=np.uint32)

        directions_numbers[0, :] = 1 << uint32_arange(31, 31 - num_bits, -1)

        for dimension_index in range(1, num_dimensions):
            start_index = self.accumulated_degrees[dimension_index - 1]
            end_index = self.accumulated_degrees[dimension_index]

            degree = end_index - start_index
            coefficients = self.coefficients[dimension_index - 1]
            initial_direction_numbers = self.direction_numbers[start_index:end_index]

            directions_numbers[dimension_index, :] = calculate_dimension_direction_numbers(degree, coefficients, initial_direction_numbers, num_bits)

        return directions_numbers

    def generate_sequence(self, num_dimensions: int, num_points: int) -> np.ndarray:
        """
        Generates a sequence of multidimensional points in the unit hypercube distributed evenly.
        The density of the points is given by `num_points`.
        :param num_dimensions: The number of dimensions to consider.
        :param num_points: The number of points to obtain.
        :return: A matrix of size (number of points, number of dimensions) with float values in the [0, 1] range.
        """
        first_zero_indices = get_first_zero_indices(num_points)

        direction_numbers = self.calculate_direction_numbers(num_dimensions, num_points)

        sequence = np.zeros((num_points, num_dimensions), dtype=np.uint32)

        for point_index in range(1, num_points):
            sequence[point_index, :] = sequence[point_index - 1, :] ^ direction_numbers[:, first_zero_indices[point_index - 1] - 1]

        sequence = sequence / float(1 << 32)

        return sequence


@numba.njit()
def get_first_zero_indices(num_points: int) -> np.ndarray:
    """
    Get the index of the first zero digit from the right for each binary representation of a range of `num_points`.
    Indices start at 1.
    :param num_points: The number of points to get the index from.
    :return: A vector of integers ordered by point indices in ascending order.
    """
    first_zero_indices = np.empty(num_points, dtype=np.int32)

    for index in range(num_points):
        zero_index, shifted_index = 1, index

        while shifted_index & 1 == 1:
            shifted_index = shifted_index >> 1
            zero_index += 1

        first_zero_indices[index] = zero_index

    return first_zero_indices


@numba.njit()
def calculate_dimension_direction_numbers(degree: int, coefficients: int, initial_direction_numbers: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Calculate the missing direction numbers for a specific dimension using a primitive polynomial.
    The produced direction numbers are represented as decimals by placing them at the beginning of a 32-bit sequence.
    :param degree: The degree of the primitive polynomial.
    :param coefficients: An integer representation of the primitive polynomial coefficients.
    :param initial_direction_numbers: The initial direction numbers given by the degree.
    :param num_bits: The maximum number of bits required to represent all point indices.
    :return: A vector of integers representing direction numbers up to `num_bits`.
    """
    if num_bits <= degree:
        direction_numbers = initial_direction_numbers << uint32_arange(31, 31 - num_bits, -1)

    else:
        direction_numbers = np.empty(num_bits, dtype=np.uint32)

        direction_numbers[:degree] = initial_direction_numbers << uint32_arange(31, 31 - degree, -1)

        for direction_index in range(degree, num_bits):
            direction_numbers[direction_index] = direction_numbers[direction_index - degree] ^ (direction_numbers[direction_index - degree] >> degree)

            for coefficient_index in range(degree - 1):
                coefficient = (coefficients >> (degree - coefficient_index - 2)) & 1

                direction_numbers[direction_index] ^= coefficient * direction_numbers[direction_index - coefficient_index - 1]

    return direction_numbers


@numba.njit()
def uint32_arange(start: int, stop: int, step: int = 1) -> np.ndarray:
    """
    Implements a Numba version of np.arange that supports uint32 types.
    :param start: The initial value of the array.
    :param stop: The last stop of the array.
    :param step: The value to sum to the previous value to create the current.
    :return: A vector of type `uint32` with the specified range.
    """
    length = (stop - start) // step

    if length <= 0:
        return np.empty(0, dtype=np.uint32)

    array_range = np.empty(length, dtype=np.uint32)

    for index in range(length):
        array_range[index] = start
        start += step

    return array_range


def new_sobol_sequence_generator() -> SobolSequenceGenerator:
    """
    Loads `sobol.npz` from the location of this script and creates an instance of `SobolSequenceGenerator`.
    The file contains a series of vectors with the primitive polynomials and directions numbers.
    :return: An instance of `SobolSequenceGenerator`.
    """
    script_dir = os.path.dirname(__file__)

    data = np.load(os.path.join(script_dir, 'resources', 'sobol.npz'))

    sobol_sequence_generator = SobolSequenceGenerator(
        data['accumulated_degrees'],
        data['coefficients'],
        data['direction_numbers'])

    return sobol_sequence_generator
