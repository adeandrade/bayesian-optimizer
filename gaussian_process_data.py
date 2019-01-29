import dataclasses
import operator
from typing import List, Tuple, Mapping

import numpy as np

from .gaussian_process_data_pb2 import GaussianProcessData as GaussianProcessDataPB


@dataclasses.dataclass(frozen=True)
class ValueBounds:
    lower: float
    upper: float


@dataclasses.dataclass(frozen=True)
class Input:
    name: str
    bounds: ValueBounds


@dataclasses.dataclass(frozen=True)
class CompletedTrial:
    inputs: List[float]
    targets: List[float]


@dataclasses.dataclass(frozen=True)
class GaussianProcessData:
    id: str
    inputs: List[Input]
    target_names: List[str]
    sigma: float
    kernel_constant: float
    length_scales: List[float]
    length_scale_bounds: ValueBounds
    completed_trials: List[CompletedTrial]

    @property
    def num_features(self) -> int:
        return len(self.inputs)

    def get_length_scale_bounds(self) -> Tuple[float, float]:
        return self.length_scale_bounds.lower, self.length_scale_bounds.upper

    def get_feature_bounds(self) -> List[Tuple[float, float]]:
        return [(feature.bounds.lower, feature.bounds.upper) for feature in self.inputs]

    def get_inputs(self) -> np.ndarray:
        return np.array(trial.inputs for trial in self.completed_trials)

    def get_targets(self) -> np.ndarray:
        return np.array(trial.targets for trial in self.completed_trials)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_inputs(), self.get_targets()

    def get_length_scales(self) -> np.ndarray:
        return np.array(self.length_scales)

    def with_completed_numpy_trial(self, inputs: np.ndarray, targets: np.ndarray) -> 'GaussianProcessData':
        return self.with_completed_trial(list(inputs), list(targets))

    def with_completed_trial(self, inputs: List[float], targets: List[float]) -> 'GaussianProcessData':
        completed_trials = self.completed_trials + [CompletedTrial(inputs, targets)]

        return dataclasses.replace(self, completed_trials=completed_trials)

    def get_ordered_inputs(self, inputs: Mapping[str, float]) -> List[float]:
        input_indices = {data_input.name: index for index, data_input in enumerate(self.inputs)}

        indexed_inputs = ((value, input_indices[key]) for key, value in inputs.items())

        sorted_inputs = [value for value, _ in sorted(indexed_inputs, key=operator.itemgetter(1))]

        return sorted_inputs

    def get_ordered_targets(self, targets: Mapping[str, float]) -> List[float]:
        target_indices = {index: target_name for index, target_name in enumerate(self.target_names)}

        indexed_targets = [(value, target_indices[key]) for key, value in targets.items()]

        sorted_targets = [value for value, _ in sorted(indexed_targets, key=operator.itemgetter(1))]

        return sorted_targets

    def to_bytes(self) -> str:
        pb_gaussian_process_data = GaussianProcessDataPB()

        pb_gaussian_process_data.id = self.id

        for data_input in self.inputs:
            pb_input = pb_gaussian_process_data.inputs.add()

            pb_input.name = data_input.name

            pb_input.bounds.lower = data_input.bounds.lower
            pb_input.bounds.upper = data_input.bounds.upper

        pb_gaussian_process_data.target_names.extend(self.target_names)
        pb_gaussian_process_data.sigma = self.sigma
        pb_gaussian_process_data.kernel_constant = self.kernel_constant
        pb_gaussian_process_data.length_scales.extend(self.length_scales)
        pb_gaussian_process_data.length_scale_bounds.lower = self.length_scale_bounds.lower
        pb_gaussian_process_data.length_scale_bounds.upper = self.length_scale_bounds.upper

        for trial in self.completed_trials:
            pb_completed_trial = pb_gaussian_process_data.completed_trials.add()

            pb_completed_trial.inputs.extend(trial.inputs)
            pb_completed_trial.targets.extend(trial.targets)

        return pb_gaussian_process_data.SerializeToString()

    @staticmethod
    def from_bytes(data: str) -> 'GaussianProcessData':
        pb_gaussian_process_data = GaussianProcessDataPB()
        pb_gaussian_process_data.ParseFromString(data)

        data_inputs = [
            Input(data_input.name, ValueBounds(data_input.bounds.lower, data_input.bounds.upper))
            for data_input in pb_gaussian_process_data.inputs]

        length_scale_bounds = ValueBounds(
            pb_gaussian_process_data.length_scale_bounds.lower,
            pb_gaussian_process_data.length_scale_bounds.upper)

        completed_trials = [
            CompletedTrial(completed_trial.inputs, completed_trial.targets)
            for completed_trial in pb_gaussian_process_data.completed_trials]

        gaussian_process_data = GaussianProcessData(
            id=pb_gaussian_process_data.id,
            inputs=data_inputs,
            target_names=pb_gaussian_process_data.target_names,
            sigma=pb_gaussian_process_data.sigma,
            kernel_constant=pb_gaussian_process_data.kernel_constant,
            length_scales=pb_gaussian_process_data.length_scales,
            length_scale_bounds=length_scale_bounds,
            completed_trials=completed_trials)

        return gaussian_process_data
