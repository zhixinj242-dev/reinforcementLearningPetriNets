from skrl.agents.torch.dqn import DQN
from typing import Union, Tuple, Dict, Any, Optional, Mapping

import gymnasium as gym
import copy
import math

import torch
import torch.nn.functional as F

from skrl.memories.torch import Memory
from skrl.models.torch import Model

from skrl.agents.torch import Agent

import numpy as np
import random

from utils.log_manager import LogManager
from environment import JunctionPetriNetEnv

class CDQN(DQN):

    def _masked_random_act(self, states: torch.Tensor) -> torch.Tensor:
        """
        【新增辅助函数】：受约束的随机采样。
        只在 Mask=1 的合法动作中进行随机选择，而不是在所有动作中瞎选。
        """
        batch_size = states.shape[0]
        actions = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        
        # 提取 Mask (前 N 个状态位)
        masks = states[:, 0:self.action_space.n]  # Shape: [batch, n_actions]
        
        for i in range(batch_size):
            # 找出当前样本所有合法的动作索引
            # 严格判断：mask值必须等于1.0才认为是合法动作
            valid_indices = torch.nonzero(masks[i] == 1.0).view(-1)
            
            if valid_indices.numel() > 0:
                # 如果有合法动作，从中随机选一个
                choice_idx = torch.randint(0, valid_indices.numel(), (1,), device=self.device)
                actions[i] = valid_indices[choice_idx]
            else:
                # 理论上不该发生（死锁），如果发生了就选择"什么都不做"动作（最后一个动作）
                actions[i] = torch.tensor(self.action_space.n - 1, device=self.device)
                
        return actions

    def __init__(self, models, memory, observation_space, action_space, device, cfg=None, log_manager=None, raw_env=None):
        # 按照skrl库的正确参数顺序调用父类__init__
        super().__init__(models, memory, observation_space, action_space, device, cfg)
        
        self.algorithm_type = "CDQN"
        self.log_manager = log_manager
        self.last_action_info = {}
        self.raw_env = raw_env  # 保存原始环境的引用

    def __del__(self):
        # 关闭日志文件
        if hasattr(self, "log_manager") and self.log_manager:
            self.log_manager.close()

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
        # 确保states是二维的
        if states.dim() == 1:
            states = states.unsqueeze(0)
        elif states.dim() == 0:
            states = states.unsqueeze(0).unsqueeze(0)
        
        # 【关键修复】：在状态预处理之前保存原始状态，确保车辆观测不被标准化
        original_states = states.clone()
        original_state = original_states.cpu().numpy()[0].tolist()
        
        # 提取掩码（从原始状态中提取）
        original_masks = None
        if original_states.shape[1] >= self.action_space.n:
            original_masks = original_states[:, 0:self.action_space.n].clone()
        
        # 提取合法动作索引
        valid_indices = []
        masks_list = []
        if original_masks is not None:
            masks_list = original_masks[0].tolist()
            valid_indices = [i for i, mask in enumerate(masks_list) if mask == 1.0]
        
        # 对状态进行预处理（仅用于Q值网络）
        states = self._state_preprocessor(states)

        # 初始化日志信息字典
        log_info = {
            "original_state": original_state,
            "mask": masks_list,
            "legal_actions": valid_indices,
            "exploration_type": "",
            "selected_action": -1,
            "selected_action_name": "Unknown",
            "action_is_legal": False,
            "q_values": [],
            "masked_q_values": [],
            "environment_acceptance": None,
            "environment_legal_actions": [],
            "environment_legal_state": None
        }

        # 2. 评估模式（完全贪婪策略）
        if not self._exploration_timesteps:
            q_actions = self.q_network.act({"states": states}, role="debug")[0]
            
            # 提取掩码，确保评估模式也严格遵循掩码约束
            masks = original_masks if original_masks is not None else states[:, 0:self.action_space.n]
            
            # 只在合法动作中选择贪婪动作
            masked_q_actions = q_actions.clone()
            masked_q_actions[masks == 0.0] = 0.0
            
            # 计算贪婪动作
            actions = torch.argmax(masked_q_actions, dim=1, keepdim=True)
            
            # 更新日志信息
            log_info["exploration_type"] = "贪婪利用（评估模式）"
            log_info["q_values"] = q_actions[0].tolist()
            log_info["masked_q_values"] = masked_q_actions[0].tolist()

        # 3. 训练初期的完全随机探索
        elif timestep < self._random_timesteps:
            # 使用原始掩码进行随机动作选择
            if original_masks is not None:
                # 创建一个临时状态，使用原始掩码
                temp_states = states.clone()
                if temp_states.shape[1] >= self.action_space.n:
                    temp_states[:, 0:self.action_space.n] = original_masks
                actions = self._masked_random_act(temp_states)
            else:
                actions = self._masked_random_act(states)
            
            # 更新日志信息
            log_info["exploration_type"] = "纯随机探索"

        # 4. Epsilon-Greedy 探索
        else:
            # 计算 epsilon
            epsilon = self._exploration_final_epsilon + (
                self._exploration_initial_epsilon - self._exploration_final_epsilon) \
                * math.exp(-1.0 * timestep / self._exploration_timesteps)

            # 计算最佳动作（利用）
            q_actions = self.q_network.act({"states": states}, role="debug")[0]
            
            # 提取掩码，确保贪婪选择也严格遵循掩码约束
            masks = original_masks if original_masks is not None else states[:, 0:self.action_space.n]
            
            # 只在合法动作中选择贪婪动作
            masked_q_actions = q_actions.clone()
            masked_q_actions[masks == 0.0] = 0.0
            
            # 计算贪婪动作
            greedy_actions = torch.argmax(masked_q_actions, dim=1, keepdim=True)
            
            # 生成随机动作
            if original_masks is not None:
                # 创建一个临时状态，使用原始掩码
                temp_states = states.clone()
                if temp_states.shape[1] >= self.action_space.n:
                    temp_states[:, 0:self.action_space.n] = original_masks
                random_actions = self._masked_random_act(temp_states)
            else:
                random_actions = self._masked_random_act(states)
            
            # 混合
            random_filter = torch.rand((states.shape[0], 1), device=self.device) < epsilon
            actions = torch.where(random_filter, random_actions, greedy_actions)

            # 更新日志信息
            log_info["exploration_type"] = "Epsilon-Greedy探索" if timestep < self._exploration_timesteps else "贪婪利用"
            log_info["q_values"] = q_actions[0].tolist()
            log_info["masked_q_values"] = masked_q_actions[0].tolist()

        # 【新增】：确保选择的动作总是合法的
        if original_masks is not None:
            masks = original_masks
            for i in range(actions.shape[0]):
                action_idx = actions[i].item()
                # 检查动作是否合法
                if action_idx < masks.shape[1]:
                    mask_value = masks[i, action_idx].item()
                    is_legal = (mask_value == 1.0)
                    
                    # 如果选择了非法动作，强制选择一个合法动作
                    if not is_legal:
                        # 找出所有合法动作
                        valid_indices = torch.nonzero(masks[i] == 1.0).view(-1)
                        if valid_indices.numel() > 0:
                            # 从合法动作中随机选择一个
                            choice_idx = torch.randint(0, valid_indices.numel(), (1,), device=self.device)
                            actions[i] = valid_indices[choice_idx]

        # 更新最终选择的动作信息
        action_idx = actions[0].item()
        log_info["selected_action"] = action_idx
        
        # 记录动作名称
        action_name = "Unknown"
        if hasattr(self, 'actions_to_transitions') and action_idx < len(self.actions_to_transitions):
            action_name = self.actions_to_transitions[action_idx]
        log_info["selected_action_name"] = action_name
        
        # 检查动作是否合法
        if original_masks is not None and action_idx < original_masks.shape[1]:
            mask_value = original_masks[0, action_idx].item()
            log_info["action_is_legal"] = (mask_value == 1.0)

        # 存储动作选择的信息，以便在记录转换时关联环境反馈信息
        self.last_action_info[timestep] = log_info

        # 暂时不记录日志，等待环境反馈信息后再一起记录
        # if self.log_manager:
        #     self.log_manager.log_step(timestep, log_info)

        return actions, None, None

    def _masked_argmax(self, q_values: torch.Tensor, transitions: torch.Tensor) -> torch.Tensor:
        """
        【内部辅助函数】：安全的 Argmax。
        因为我们在 Model 层已经通过“Shift-then-Mask”策略把非法动作的 Q 值变成了极小的负数 (-offset)，
        所以这里不需要再做复杂的 Shift，直接 Argmax 即可！
        """
        # 由于 Model 层已经保证了非法动作的值 << 合法动作的值（因为非法动作被减去了 offset，而合法动作是原始值）
        # 所以直接 argmax 就能选出合法的最大值。
        return torch.argmax(q_values, dim=1, keepdim=True)

    def record_transition(self, *args, **kwargs):
        """
        记录转换并更新日志，关联环境反馈信息
        
        支持多种参数格式，以兼容不同版本的skrl库
        """
        # 从kwargs中提取参数
        states = kwargs.get('states', args[0] if len(args) > 0 else None)
        actions = kwargs.get('actions', args[1] if len(args) > 1 else None)
        rewards = kwargs.get('rewards', args[2] if len(args) > 2 else None)
        next_states = kwargs.get('next_states', args[3] if len(args) > 3 else None)
        terminated = kwargs.get('terminated', args[4] if len(args) > 4 else None)
        truncated = kwargs.get('truncated', args[5] if len(args) > 5 else None)
        infos = kwargs.get('infos', args[6] if len(args) > 6 else None)
        timestep = kwargs.get('timestep', args[7] if len(args) > 7 else None)
        timesteps = kwargs.get('timesteps', args[8] if len(args) > 8 else None)
        
        # 确保所有必要的参数都有值
        if states is None:
            states = kwargs.get('observations')
        if next_states is None:
            next_states = kwargs.get('next_observations')
        if terminated is None:
            # 尝试从多个来源获取terminated
            terminated = kwargs.get('dones', kwargs.get('done', False))
        if truncated is None:
            # 尝试从多个来源获取truncated
            truncated = kwargs.get('truncated', kwargs.get('done', False))
        if infos is None:
            infos = kwargs.get('info', {})
        if timestep is None:
            timestep = kwargs.get('step')
        if timesteps is None:
            # 使用默认值3000
            timesteps = 3000
        
        # 确保必要的参数有值
        if terminated is None:
            terminated = False
        if truncated is None:
            truncated = False
        if infos is None:
            infos = {}
        if timestep is None:
            timestep = 0
        if timesteps is None:
            timesteps = 3000
        
        # 调用父类的record_transition方法
        try:
            # 检查父类方法的签名，确保传递正确的参数
            # 直接使用位置参数传递所有必要的参数
            super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
        except Exception as e:
            # 如果父类调用失败，至少记录日志
            print(f"Error calling parent record_transition: {e}")
            # 尝试使用kwargs方式传递
            try:
                # 构建一个包含所有必要参数的字典，确保使用正确的键名
                transition_kwargs = {
                    'states': states,
                    'actions': actions,
                    'rewards': rewards,
                    'next_states': next_states,
                    'terminated': terminated,
                    'truncated': truncated,
                    'infos': infos,
                    'timestep': timestep,
                    'timesteps': timesteps
                }
                # 传递kwargs
                super().record_transition(**transition_kwargs)
            except Exception as e2:
                print(f"Error calling parent record_transition with kwargs: {e2}")
                # 尝试使用不同的参数组合
                try:
                    # 尝试传递更少的参数
                    if len(args) >= 5:
                        super().record_transition(*args)
                    else:
                        # 只传递最基本的参数
                        super().record_transition(states, actions, rewards, next_states, terminated)
                except Exception as e3:
                    print(f"Error calling parent record_transition with reduced args: {e3}")
                    # 如果所有尝试都失败，忽略错误，继续训练
                    pass
        
        # 如果有日志管理器和环境信息，更新日志
        if self.log_manager and timestep in self.last_action_info and infos:
            # 获取动作选择的信息
            action_info = self.last_action_info[timestep]
            
            # 更新环境反馈信息
            action_info["environment_acceptance"] = infos.get("environment_acceptance", None)
            action_info["environment_legal_actions"] = infos.get("environment_legal_actions", [])
            action_info["environment_legal_state"] = infos.get("environment_legal_state", None)
            
            # 【关键修复】：从环境信息中提取原始观测，替换预处理后的状态
            environment_legal_state = infos.get("environment_legal_state", None)
            if environment_legal_state is not None:
                # 使用环境的flatten_observation方法确保顺序一致（现在是实例方法）
                if self.raw_env is not None:
                    flattened_original_state = self.raw_env.flatten_observation(environment_legal_state)
                else:
                    flattened_original_state = JunctionPetriNetEnv.flatten_observation(environment_legal_state)
                
                # 将numpy数组转换为float
                flattened_original_state = [
                    float(value[0]) if hasattr(value, 'ndim') and value.ndim > 0 else float(value)
                    for value in flattened_original_state
                ]
                
                # 替换原始状态
                action_info["original_state"] = flattened_original_state
            
            # 记录完整的日志条目，包括动作选择信息和环境反馈信息
            self.log_manager.log_step(timestep, action_info)
            
            # 移除已处理的动作信息
            del self.last_action_info[timestep]
        elif self.log_manager and timestep in self.last_action_info:
            # 如果没有环境信息，至少记录动作选择信息
            action_info = self.last_action_info[timestep]
            self.log_manager.log_step(timestep, action_info)
            del self.last_action_info[timestep]

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
                                                                role="debug")

                # calculate safe sets S_C for all constraints
                # TODO: somethings wrong here, doesnt really learn, see tensorboard!
                next_valid_q_values = torch.mul(next_q_values, transitions)
                target_q_values = torch.max(next_valid_q_values, dim=-1, keepdim=True)[0]
                target_values = sampled_rewards + self._discount_factor * sampled_dones.logical_not() * target_q_values

            # compute Q-network loss
            q_values = torch.gather(self.q_network.act({"states": sampled_states}, role="debug")[0],
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

