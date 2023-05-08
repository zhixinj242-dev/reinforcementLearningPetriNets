from skrl.models.torch.deterministic import DeterministicMixin
from typing import Union, Mapping, Tuple, Any
import gymnasium
import torch


class CDeterministicMixin(DeterministicMixin):
    def act(self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act deterministically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is ``None``. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dictionary

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, _, outputs = model.act({"states": states})
            >>> print(actions.shape, outputs)
            torch.Size([4096, 1]) {}
        """
        # map from observations/states to actions
        actions, outputs = self.compute(inputs, role)

        # clip actions
        if self._d_clip_actions[role] if role in self._d_clip_actions else self._d_clip_actions[""]:
            if self._backward_compatibility:
                actions = torch.max(torch.min(actions, self.clip_actions_max), self.clip_actions_min)
            else:
                actions = torch.clamp(actions, min=self.clip_actions_min, max=self.clip_actions_max)

        return actions, None, outputs