#! /opt/conda/bin/python3
""" File containing keras Model class extension """

# Copyright 2018 FAU-iPAT (http://ipat.uni-erlangen.de/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Union, List, Dict, Any
import keras.backend as backend  # type: ignore
from keras.models import Model as KerasModel  # type: ignore
from keras.models import model_from_config as keras_model_from_config
from keras.models import model_from_json as keras_model_from_json
import numpy as np


# noinspection PyClassHasNoInit
class Model(KerasModel):
    """
    Class extending the base keras.models.Model class

    Extension of the base class to insert some additional
    methods with functionality that was often required.
    """

    def get_activations(
            self,
            inputs: Union[np.ndarray, List[np.ndarray]],
            layer_names: Union[str, List[str]] = None
    ) -> List[np.ndarray]:
        """
        Method to track the activation of layers due to the inputs

        :param inputs: Input to track activation from
        :param layer_names: List of layer names to get activation for
        :return: List of activations
        """
        model_inputs = self.input if isinstance(self.input, list) else [self.input, ]
        input_values = [*inputs, 0.] if isinstance(inputs, list) else [inputs, 0.]
        layer_names = [layer_names, ] if layer_names is not None and not isinstance(layer_names, list) else layer_names
        requested = [layer.output for layer in self.layers if layer.name in layer_names or layer_names is None]
        functions = [backend.function(model_inputs + [backend.learning_phase()], [req]) for req in requested]
        return [func(input_values)[0] for func in functions]

    def get_sorted_weights(self, include_bias: bool = False) -> np.ndarray:
        """
        Method to get sorted list of all weights

        :param include_bias: Whether to include bias values
        :return: Sorted list of all layer weights
        """
        results = np.zeros((0,))
        for layer in self.layers:
            weights = layer.get_weights()
            if weights:
                results = np.concatenate((results, weights[0].flatten()))
            if len(weights) > 1 and include_bias is True:
                results = np.concatenate((results, weights[1].flatten()))
        return np.sort(results)


def model_from_config(
        config: str,
        custom_objects: Dict[str, Any] = None
) -> Model:
    """
    Model to load model from configuration data (with custom layers)

    :param config: Configuration data
    :param custom_objects: Dictionary of additional custom layers
    :return: Loaded model
    """
    all_custom_objects = {
        'Model': Model,
    }
    if custom_objects is not None:
        all_custom_objects.update(custom_objects)
    return keras_model_from_config(config, all_custom_objects)


def model_from_json(
        json: str,
        custom_objects: Dict[str, Any] = None
) -> Model:
    """
    Method to load a model from json (with custom layers)

    :param json: JSON string describing the model
    :param custom_objects: Dictionary of additional custom layers
    :return: Loaded model
    """
    all_custom_objects = {
        'Model': Model,
    }
    if custom_objects is not None:
        all_custom_objects.update(custom_objects)
    return keras_model_from_json(json, all_custom_objects)
