import unittest

import numpy as np

from .sobol import uint32_arange, get_first_zero_indices, calculate_dimension_direction_numbers, SobolSequenceGenerator


class TestSobolSequences(unittest.TestCase):
    def test_numba_uint32_arange(self):
        custom_range = uint32_arange(31, 31 - 5, -1)
        expected = np.arange(31, 31 - 5, -1, np.int32)

        np.testing.assert_array_equal(custom_range, expected)

    def test_get_first_zero_indices(self):
        num_points = 7

        zero_indices = get_first_zero_indices(num_points)

        expected = np.array([1, 2, 1, 3, 1, 2, 1])

        np.testing.assert_array_equal(zero_indices, expected)

    def test_dimension_direction_numbers(self):
        degree = 3
        coefficients = 1
        initial_direction_numbers = np.array([1, 3, 7], dtype=np.uint32)

        direction_numbers = calculate_dimension_direction_numbers(degree, coefficients, initial_direction_numbers, num_bits=5)
        direction_numbers = direction_numbers / float(1 << 32)

        expected = np.array([0.5, 0.75, 0.875, 0.3125, 0.21875], dtype=np.float32)

        np.testing.assert_array_equal(direction_numbers, expected)

    def test_sequence_generations(self):
        accumulated_degrees = np.array([0, 1, 4], dtype=np.uint32)
        coefficients = np.array([0, 1], dtype=np.uint32)
        direction_numbers = np.array([1, 1, 3, 7], dtype=np.uint32)

        generator = SobolSequenceGenerator(accumulated_degrees, coefficients, direction_numbers)

        sequence = np.squeeze(generator.generate_sequence(3, 5))
        expected = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.375, 0.375, 0.125]], dtype=np.float32)

        np.testing.assert_array_equal(sequence, expected)
