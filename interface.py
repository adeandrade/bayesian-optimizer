import dataclasses
from typing import Mapping, Tuple, List, Sequence, Union

from sklearn.gaussian_process.kernels import Matern

from .bayesian_optimizer import BayesianOptimizer
from .gaussian_process import GaussianProcess
from .gaussian_process_data import GaussianProcessData, ValueBounds, Input
from .kernel import Matern
from .store import Store


class Interface:
    @staticmethod
    def create_new_model(
            model_id: str,
            inputs: Sequence[Tuple[str, float, float]],
            target_names: List[str],
            sigma: float = 1e-10,
            kernel_constant: float = 1.0,
            length_scales: Union[float, List[float]] = 1.0,
            length_scale_bounds: Tuple[float, float] = (1e-5, 1e5)) -> None:
        """

        :param model_id:
        :param inputs:
        :param target_names:
        :param sigma:
        :param kernel_constant:
        :param length_scales:
        :param length_scale_bounds:
        :return:
        """
        boxed_inputs = [Input(data_input[0], ValueBounds(data_input[1], data_input[2])) for data_input in inputs]

        boxed_length_scale_bounds = ValueBounds(length_scale_bounds[0], length_scale_bounds[1])

        boxed_length_scales = [length_scales for _ in range(len(inputs))] if type(length_scales) is float else length_scales

        data = GaussianProcessData(
            id=model_id,
            inputs=boxed_inputs,
            target_names=target_names,
            sigma=sigma,
            kernel_constant=kernel_constant,
            length_scales=boxed_length_scales,
            length_scale_bounds=boxed_length_scale_bounds,
            completed_trials=[])

        Store.set_model(data)

    @staticmethod
    def get_trial_inputs(model_id: str, version: str) -> Mapping[str, float]:
        """

        :param model_id:
        :param version:
        :return:
        """
        return Store.get_trial_inputs(model_id, version)

    @staticmethod
    def complete_trial(model_id: str, version: str, targets: Mapping[str, float]) -> None:
        """
        The targets we are trying to minimize
        :param model_id:
        :param version:
        :param targets:
        :return:
        """
        state = Store.get_model(model_id)

        trial_inputs = Interface.get_trial_inputs(model_id, version)

        sorted_inputs = state.get_ordered_inputs(trial_inputs)

        sorted_targets = state.get_ordered_targets(targets)

        new_state = state.with_completed_trial(sorted_inputs, sorted_targets)

        Store.set_model(new_state)

    @staticmethod
    def get_next_trial(model_id: str, version: str) -> Mapping[str, float]:
        """

        :param model_id:
        :param version:
        :return:
        """
        state = Store.get_model(model_id)

        inputs, targets = state.get_inputs(), state.get_targets()

        kernel = Matern(state.get_length_scales(), state.kernel_constant, state.get_length_scale_bounds())

        optimized_kernel = GaussianProcess.optimize_kernel(kernel, inputs, targets, state.sigma)

        bayesian_optimizer = BayesianOptimizer(optimized_kernel, state.sigma, state.get_feature_bounds())

        suggestion = bayesian_optimizer.suggest(inputs, targets)

        trial_inputs = {data_input.name: suggestion[inputs] for index, data_input in enumerate(state.inputs)}

        new_state = dataclasses.replace(state, length_scales=optimized_kernel.length_scales, kernel_constant=optimized_kernel.constant)

        Store.set_model(new_state)

        Store.set_trial_inputs(model_id, version, trial_inputs)

        return trial_inputs
