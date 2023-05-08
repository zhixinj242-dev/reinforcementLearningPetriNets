from skrl.models.torch.base import Model
from skrl.utils.model_instantiators import _generate_sequential
from agents import CDeterministicMixin
from typing import Union, Tuple, Optional, Sequence, Mapping, Any

import gym
import gymnasium
from enum import Enum

import torch
import torch.nn as nn

class Shape(Enum):
    """
    Enum to select the shape of the model's inputs and outputs
    """
    ONE = 1
    STATES = 0
    OBSERVATIONS = 0
    ACTIONS = -1
    STATES_ACTIONS = -2
def c_deterministic_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                        device: Optional[Union[str, torch.device]] = None,
                        clip_actions: bool = False,
                        input_shape: Shape = Shape.STATES,
                        hiddens: list = [256, 256],
                        hidden_activation: list = ["relu", "relu"],
                        output_shape: Shape = Shape.ACTIONS,
                        output_activation: Optional[str] = "tanh",
                        output_scale: float = 1.0) -> Model:
    class CDeterministicModel(CDeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions, metadata):
            Model.__init__(self, observation_space, action_space, device)
            CDeterministicMixin.__init__(self, clip_actions)

            self.instantiator_output_scale = metadata["output_scale"]
            self.instantiator_input_type = metadata["input_shape"].value

            self.net = _generate_sequential(model=self,
                                            input_shape=metadata["input_shape"],
                                            hiddens=metadata["hiddens"],
                                            hidden_activation=metadata["hidden_activation"],
                                            output_shape=metadata["output_shape"],
                                            output_activation=metadata["output_activation"],
                                            output_scale=metadata["output_scale"])

        def compute(self, inputs, role=""):
            if self.instantiator_input_type == 0:
                output = self.net(inputs["states"])
            elif self.instantiator_input_type == -1:
                output = self.net(inputs["taken_actions"])
            elif self.instantiator_input_type == -2:
                output = self.net(torch.cat((inputs["states"], inputs["taken_actions"]), dim=1))

            return output * self.instantiator_output_scale, {}

    metadata = {"input_shape": input_shape,
                "hiddens": hiddens,
                "hidden_activation": hidden_activation,
                "output_shape": output_shape,
                "output_activation": output_activation,
                "output_scale": output_scale}

    return CDeterministicModel(observation_space=observation_space,
                                  action_space=action_space,
                                  device=device,
                                  clip_actions=clip_actions,
                                  metadata=metadata)