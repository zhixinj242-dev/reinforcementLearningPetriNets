from skrl.agents.torch.dqn import DQN
from typing import Union, Tuple, Dict, Any, Optional, Mapping

import gym, gymnasium
import copy
import math

import torch
import torch.nn.functional as F

from skrl.memories.torch import Memory
from skrl.models.torch import Model

from skrl.agents.torch import Agent

import numpy as np
import random

class CDQN(DQN):

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy
        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        :return: Actions
        :rtype: torch.Tensor
        """
        states = self._state_preprocessor(states)

        # calculate safe sets for optimal deterministic policy extraction
        #transitions = states.numpy()[0][:self.action_space.n]
        #valid_transitions = np.where(transitions == 1)[0]

        # for evaluation purposes (fully exploit)
        if not self._exploration_timesteps:
            return torch.argmax(self.q_network.act({"states": states}, role="q_network")[0], dim=1,
                                keepdim=True), None, None

        # sample random actions
        actions = self.random_act({"states": states}, role="q_network")[0]
        if timestep < self._random_timesteps:
            return actions, None, None

        # sample actions with epsilon-greedy policy
        epsilon = self._exploration_final_epsilon + (
                    self._exploration_initial_epsilon - self._exploration_final_epsilon) \
                  * math.exp(-1.0 * timestep / self._exploration_timesteps)

        indexes = (torch.rand(states.shape[0], device=self.device) >= epsilon).nonzero().view(-1)
        if indexes.numel():
            actions[indexes] = torch.argmax(self.q_network.act({"states": states[indexes]}, role="q_network")[0], dim=1,
                                            keepdim=True)

        # record epsilon
        self.track_data("Exploration / Exploration epsilon", epsilon)

        return actions, None, None
    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # sample a batch from memory
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = \
            self.memory.sample(names=self.tensors_names, batch_size=self._batch_size)[0]

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            sampled_states = self._state_preprocessor(sampled_states, train=not gradient_step)
            sampled_next_states = self._state_preprocessor(sampled_next_states)

            transitions = sampled_states[:,:self.action_space.n]

            # compute target values
            with torch.no_grad():
                next_q_values, _, _ = self.target_q_network.act({"states": sampled_next_states},
                                                                role="target_q_network")

                # calculate safe sets S_C for all constraints
                # TODO: somethings wrong here, doesnt really learn, see tensorboard!
                next_valid_q_values = torch.mul(next_q_values, transitions)
                target_q_values = torch.max(next_valid_q_values, dim=-1, keepdim=True)[0]
                target_values = sampled_rewards + self._discount_factor * sampled_dones.logical_not() * target_q_values

            # compute Q-network loss
            q_values = torch.gather(self.q_network.act({"states": sampled_states}, role="q_network")[0],
                                    dim=1, index=sampled_actions.long())

            q_network_loss = F.mse_loss(q_values, target_values)

            # optimize Q-network
            self.optimizer.zero_grad()
            q_network_loss.backward()
            self.optimizer.step()

            # update target network
            if not timestep % self._target_update_interval:
                self.target_q_network.update_parameters(self.q_network, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.scheduler.step()

            # record data
            self.track_data("Loss / Q-network loss", q_network_loss.item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            if self._learning_rate_scheduler:
                self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

    def random_act(self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = "") \
            -> Tuple[torch.Tensor, None, Mapping[str, Union[torch.Tensor, Any]]]:
        """Act randomly according to the action space

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises NotImplementedError: Unsupported action space

        :return: Model output. The first component is the action to be taken by the agent
        :rtype: tuple of torch.Tensor, None, and dictionary
        """
        # calculate safe sets and chose random action
        transitions = inputs["states"].numpy()[0][:self.action_space.n]
        valid_transitions = np.where(transitions == 1)[0]
        chosen_random_action = random.choice(valid_transitions)

        # discrete action space (Discrete)
        if issubclass(type(self.action_space), gym.spaces.Discrete) or issubclass(type(self.action_space), gymnasium.spaces.Discrete):
             return torch.Tensor([[chosen_random_action]], device=self.device).type(torch.int64), None, {}
        # continuous action space (Box)
        elif issubclass(type(self.action_space), gym.spaces.Box) or issubclass(type(self.action_space), gymnasium.spaces.Box):
            if self._random_distribution is None:
                self._random_distribution = torch.distributions.uniform.Uniform(
                    low=torch.tensor(self.action_space.low[0], device=self.device, dtype=torch.float32),
                    high=torch.tensor(self.action_space.high[0], device=self.device, dtype=torch.float32))

            return self._random_distribution.sample(sample_shape=(inputs["states"].shape[0], self.num_actions)), None, {}
        else:
            raise NotImplementedError("Action space type ({}) not supported".format(type(self.action_space)))

