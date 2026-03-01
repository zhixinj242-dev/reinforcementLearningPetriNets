from skrl.models.torch.base import Model
from skrl.utils.model_instantiators import _generate_sequential
from agents import CDeterministicMixin
from typing import Union, Tuple, Optional, Sequence, Mapping, Any

import gymnasium as gym
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
def c_deterministic_model(observation_space: Optional[Union[int, Tuple[int], gym.Space]] = None,
                        action_space: Optional[Union[int, Tuple[int], gym.Space]] = None,
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
            
            # 初始化日志文件
            import os
            os.makedirs("debug_logs", exist_ok=True)
            self.log_file = open("debug_logs/model_debug.log", "a", encoding="utf-8")

        def __del__(self):
            # 关闭日志文件
            if hasattr(self, "log_file") and not self.log_file.closed:
                self.log_file.close()

        def compute(self, inputs, role=""):
            if self.instantiator_input_type == 0:
                output = self.net(inputs["states"])
            elif self.instantiator_input_type == -1:
                output = self.net(inputs["taken_actions"])
            elif self.instantiator_input_type == -2:
                output = self.net(torch.cat((inputs["states"], inputs["taken_actions"]), dim=1))

            # 约束：确保非法动作Q值=0，合法动作Q值>0
            # 1. 正确提取每个样本的 Mask
            mask = inputs["states"][:, 0:self.action_space.n]
            if mask.device != output.device:
                mask = mask.to(output.device)

            # 2. 非法动作设为0
            masked_output = output * mask

            # 3. 找出batch中所有合法动作的最小Q值
            # 将非法动作设为极大值，这样min不会受非法动作影响
            masked_with_large = masked_output + (1 - mask) * 1e9
            min_valid_q = torch.min(masked_with_large, dim=1, keepdim=True)[0]

            # 4. 如果最小Q值<=0，则平移该样本的所有合法动作使其最小值=1.0
            shift = torch.where(min_valid_q <= 0, 1.0 - min_valid_q, torch.tensor(0.0, device=output.device))
            final_output = masked_output + shift

            # 5. 非法动作强制设为0
            final_output = final_output * mask
            
            # 添加调试信息：记录Q值分布和掩码情况到文件
            if role == "debug":
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.log_file.write(f"[{timestamp}] Q值范围: min={torch.min(output).item()}, max={torch.max(output).item()}, mean={torch.mean(output).item()}\n")
                self.log_file.write(f"[{timestamp}] Q值 <= 0 的数量: {torch.sum(output <= 0).item()}/{output.numel()}\n")
                self.log_file.write(f"[{timestamp}] 掩码前输出: {output}\n")
                self.log_file.write(f"[{timestamp}] 掩码: {mask}\n")
                self.log_file.write(f"[{timestamp}] 掩码后输出: {final_output}\n")
                self.log_file.write(f"[{timestamp}] 掩码后合法动作Q值: {final_output * mask}\n")
                self.log_file.flush()  # 确保数据写入文件
            
            return final_output * self.instantiator_output_scale, {}

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