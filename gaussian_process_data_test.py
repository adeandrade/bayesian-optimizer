import unittest

from .gaussian_process_data import GaussianProcessData, Input, ValueBounds, CompletedTrial


class TestGaussianProcessData(unittest.TestCase):
    def test_predictions(self):
        data = GaussianProcessData(
            id='experiment_id',
            inputs=[
                Input('popularity', ValueBounds(0., 1.0)),
                Input('virality', ValueBounds(0., 1.0))],
            target_names=['ctr'],
            sigma=1e-10,
            kernel_constant=1.0,
            length_scales=[1.0, 1.0],
            length_scale_bounds=ValueBounds(-1e-3, 1e3),
            completed_trials=[
                CompletedTrial([0.5, 0.5], [5.0]),
                CompletedTrial([1.0, 0.1], [10.0])])

        serialized = data.to_bytes()

        deserialized = GaussianProcessData.from_bytes(serialized)

        self.assertEqual(data, deserialized)
