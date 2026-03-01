from skrl.agents.torch.dqn import DQN
from skrl.utils.model_instantiators import deterministic_model, Shape
from agents import c_deterministic_model, CDQN
from utils.log_manager import LogManager
from environment import JunctionPetriNetEnv
import torch
import math


class LoggedDQN(DQN):
    """
    带日志记录的DQN代理
    """
    def __init__(self, models, memory, observation_space, action_space, device, cfg=None, log_manager=None, raw_env=None):
        super().__init__(models, memory, observation_space, action_space, device, cfg)
        # 从log_manager获取正确的algorithm_type
        if log_manager is not None:
            self.algorithm_type = log_manager.algorithm_type
        else:
            self.algorithm_type = "DQN"
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
        masks_list = []
        valid_indices = []
        if original_states.shape[1] >= self.action_space.n:
            original_masks = original_states[:, 0:self.action_space.n].clone()
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
            log_info["masked_q_values"] = masked_q_actions[0].tolist()
            
            # 更新日志信息
            log_info["exploration_type"] = "贪婪利用（评估模式）"
            log_info["q_values"] = q_actions[0].tolist()
            log_info["masked_q_values"] = q_actions[0].tolist()

        # 3. 训练初期的完全随机探索
        elif timestep < self._random_timesteps:
            # 生成随机动作
            actions = torch.randint(0, self.action_space.n, (states.shape[0], 1), device=self.device)
            
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
            random_actions = torch.randint(0, self.action_space.n, (states.shape[0], 1), device=self.device)
            
            # 混合
            random_filter = torch.rand((states.shape[0], 1), device=self.device) < epsilon
            actions = torch.where(random_filter, random_actions, greedy_actions)

            # 更新日志信息
            log_info["exploration_type"] = "Epsilon-Greedy探索" if timestep < self._exploration_timesteps else "贪婪利用"
            log_info["q_values"] = q_actions[0].tolist()
            log_info["masked_q_values"] = masked_q_actions[0].tolist()

        # 更新最终选择的动作信息
        action_idx = actions[0].item()
        log_info["selected_action"] = action_idx
        
        # 记录动作名称
        action_name = "Unknown"
        if hasattr(self, 'actions_to_transitions') and action_idx < len(self.actions_to_transitions):
            action_name = self.actions_to_transitions[action_idx]
        log_info["selected_action_name"] = action_name
        
        # 检查动作是否合法
        if original_masks is not None and action_idx < self.action_space.n:
            mask_value = original_masks[0, action_idx].item()
            log_info["action_is_legal"] = (mask_value == 1.0)

        # 存储动作选择的信息，以便在记录转换时关联环境反馈信息
        self.last_action_info[timestep] = log_info

        # 暂时不记录日志，等待环境反馈信息后再一起记录
        # if self.log_manager:
        #     self.log_manager.log_step(timestep, log_info)

        return actions, None, None

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


def get_dqn_model(env, memory, cfg, constrained=False, log_manager=None, device=None, raw_env=None):
    models_dqn = {}
    # 使用传递的device参数，如果没有传递则使用env.device
    target_device = device if device is not None else getattr(env, 'device', 'cpu')
    print(f"[get_dqn_model] 使用设备: {target_device}", flush=True)
    if not constrained:
        models_dqn["q_network"] = deterministic_model(observation_space=env.observation_space,
                                                  action_space=env.action_space,
                                                  device=target_device,
                                                  clip_actions=False,
                                                  input_shape=Shape.OBSERVATIONS,
                                                  hiddens=[64, 64],
                                                  hidden_activation=["relu", "relu"],
                                                  output_shape=Shape.ACTIONS,
                                                  output_activation=None,
                                                  output_scale=1.0)
        models_dqn["target_q_network"] = deterministic_model(observation_space=env.observation_space,
                                                         action_space=env.action_space,
                                                         device=target_device,
                                                         clip_actions=False,
                                                         input_shape=Shape.OBSERVATIONS,
                                                         hiddens=[64, 64],
                                                         hidden_activation=["relu", "relu"],
                                                         output_shape=Shape.ACTIONS,
                                                         output_activation=None,
                                                         output_scale=1.0)
    else:
        models_dqn["q_network"] = c_deterministic_model(observation_space=env.observation_space,
                                                  action_space=env.action_space,
                                                  device=target_device,
                                                  clip_actions=False,
                                                  input_shape=Shape.OBSERVATIONS,
                                                  hiddens=[64, 64],
                                                  hidden_activation=["relu", "relu"],
                                                  output_shape=Shape.ACTIONS,
                                                  output_activation=None,
                                                  output_scale=1.0)
        models_dqn["target_q_network"] = c_deterministic_model(observation_space=env.observation_space,
                                                         action_space=env.action_space,
                                                         device=target_device,
                                                         clip_actions=False,
                                                         input_shape=Shape.OBSERVATIONS,
                                                         hiddens=[64, 64],
                                                         hidden_activation=["relu", "relu"],
                                                         output_shape=Shape.ACTIONS,
                                                         output_activation=None,
                                                         output_scale=1.0)

    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    for model in models_dqn.values():
            model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    agent = CDQN(models=models_dqn, memory=memory, observation_space=env.observation_space,
                action_space=env.action_space, device=target_device, cfg=cfg, log_manager=log_manager, raw_env=raw_env) if constrained \
        else LoggedDQN(models=models_dqn, memory=memory, observation_space=env.observation_space,
                action_space=env.action_space, device=target_device, cfg=cfg, log_manager=log_manager, raw_env=raw_env)
    
    # 传递动作名称映射给代理，以便在日志中显示动作名称
    if hasattr(env, 'actions_to_transitions'):
        agent.actions_to_transitions = env.actions_to_transitions
    
    return agent




