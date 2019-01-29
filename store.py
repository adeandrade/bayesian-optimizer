import dataclasses
import os
import urllib.parse as url_parse
from typing import List, Mapping

import boto3

from .gaussian_process_data import GaussianProcessData
from .trial_inputs_pb2 import TrialInputs as TrialInputsPB


storage_location = 's3://{bucket}/bayesian-optimizer'


@dataclasses.dataclass(frozen=True)
class TrialInputs:
    version: str
    inputs: List[float]

    def to_map(self, model: GaussianProcessData) -> Mapping[str, float]:
        return {data_input.name: self.inputs[index] for index, data_input in enumerate(model.inputs)}

    def to_bytes(self) -> str:
        pb_trian_inputs = TrialInputsPB()

        pb_trian_inputs.version = self.version
        pb_trian_inputs.inputs.extend(self.inputs)

        return pb_trian_inputs.SerializeToString()

    @staticmethod
    def from_map(version: str, inputs: Mapping[str, float], model: GaussianProcessData) -> 'TrialInputs':
        sorted_inputs = model.get_ordered_inputs(inputs)

        return TrialInputs(version, sorted_inputs)

    @staticmethod
    def from_bytes(data: str) -> 'TrialInputs':
        pb_trial_inputs = TrialInputsPB()
        pb_trial_inputs.ParseFromString(data)

        return TrialInputs(pb_trial_inputs.version, pb_trial_inputs.inputs)


class Store:
    aws_client = boto3.client('s3')

    @classmethod
    def set_model(cls, model: GaussianProcessData) -> None:
        key = os.path.join(cls.storage_location, 'models', f'{model.id}.pb')

        url = url_parse.urlparse(key)

        cls.aws_client.put_object(url.netloc, url.path, model.to_bytes())

    @classmethod
    def get_model(cls, model_id: str) -> GaussianProcessData:
        key = os.path.join(cls.storage_location, 'models', f'{model_id}.pb')

        url = url_parse.urlparse(key)

        aws_object = cls.aws_client.get_object(url.netloc, url.path)

        return GaussianProcessData.from_bytes(aws_object['Body'].read())

    @classmethod
    def set_trial_inputs(cls, model_id: str, version: str, inputs: Mapping[str, float]) -> None:
        trial_inputs = TrialInputs.from_map(version, inputs, cls.get_model(model_id))

        key = os.path.join(cls.storage_location, 'trial_inputs', model_id, f'{version}.pb')

        url = url_parse.urlparse(key)

        cls.aws_client.put_object(url.netloc, url.path, trial_inputs.to_bytes())

    @classmethod
    def get_trial_inputs(cls, model_id: str, version: str) -> Mapping[str, float]:
        key = os.path.join(cls.storage_location, 'trial_inputs', model_id, f'{version}.pb')

        url = url_parse.urlparse(key)

        aws_object = cls.aws_client.get_object(url.netloc, url.path)

        trial_inputs = TrialInputs.from_bytes(aws_object['Body'].read())

        return trial_inputs.to_map(cls.get_model(model_id))
